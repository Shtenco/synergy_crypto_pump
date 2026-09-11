#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from crypto_pump.v2.branches import CatBoostRankerBranch, MultiHorizonTabularBranch, SocialBranch
from crypto_pump.v2.config import DEFAULT_HORIZONS
from crypto_pump.v2.ensemble import CalibratedStacker
from crypto_pump.v2.features import social_feature_columns, v2_numeric_feature_columns
from crypto_pump.v2.pipeline import V2Ensemble


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _diag(stage: str, spec: str, rows: int, features: int) -> None:
    payload = {"stage": stage, "horizon": spec, "rows": int(rows), "features": int(features)}
    print(json.dumps(payload), flush=True)


def _save_or_load_classifier(
    train: pd.DataFrame,
    features: list[str],
    spec,
    path: Path,
    iterations: int,
    depth: int,
    learning_rate: float,
    threads: int,
    resume: bool,
):
    from catboost import CatBoostClassifier

    if resume and path.exists():
        model = CatBoostClassifier()
        model.load_model(path)
        print(f"RESUME classifier {spec.name} <- {path}", flush=True)
        return model
    use = train[spec.target_col].notna().to_numpy()
    y = train.loc[use, spec.target_col].astype(np.int8).to_numpy()
    _diag("classifier", spec.name, int(use.sum()), len(features))
    x = train.loc[use, features].astype(np.float32)
    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function="Logloss",
        eval_metric="PRAUC",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=max(1, iterations // 5),
        allow_writing_files=False,
        thread_count=threads,
        border_count=128,
        task_type="CPU",
    )
    model.fit(x, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)
    del x, y
    gc.collect()
    return model


def _save_or_load_ranker(
    train: pd.DataFrame,
    features: list[str],
    spec,
    path: Path,
    iterations: int,
    depth: int,
    learning_rate: float,
    threads: int,
    resume: bool,
):
    from catboost import CatBoostRanker, Pool

    if resume and path.exists():
        model = CatBoostRanker()
        model.load_model(path)
        print(f"RESUME ranker {spec.name} <- {path}", flush=True)
        return model
    use = train[spec.mfe_col].notna().to_numpy()
    cols = ["timestamp", "symbol", spec.mfe_col, *features]
    df = train.loc[use, cols].sort_values(["timestamp", "symbol"], kind="mergesort")
    _diag("ranker", spec.name, len(df), len(features))
    relevance = df.groupby("timestamp", sort=False)[spec.mfe_col].rank(pct=True, ascending=True).to_numpy(np.float32)
    group_id = pd.factorize(df["timestamp"], sort=False)[0].astype(np.int32, copy=False)
    x = df[features].astype(np.float32)
    pool = Pool(x, label=relevance, group_id=group_id)
    model = CatBoostRanker(
        loss_function="YetiRank:mode=NDCG",
        eval_metric="NDCG:top=10",
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        border_count=128,
        random_seed=42,
        verbose=max(1, iterations // 5),
        allow_writing_files=False,
        thread_count=threads,
        task_type="CPU",
    )
    model.fit(pool)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)
    del df, x, pool, relevance, group_id
    gc.collect()
    return model


def _predict_classifier(model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    x = frame[features].astype(np.float32)
    p = model.predict_proba(x)[:, 1]
    del x
    gc.collect()
    return np.asarray(p, dtype=np.float32)


def _predict_ranker(model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    x = frame[features].astype(np.float32)
    raw = np.asarray(model.predict(x), dtype=np.float32)
    del x
    gc.collect()
    tmp = pd.DataFrame({"timestamp": frame["timestamp"].to_numpy(), "raw": raw}, index=frame.index)
    return tmp.groupby("timestamp", sort=False)["raw"].rank(pct=True).to_numpy(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fault-tolerant staged final V2 CatBoost + YetiRank trainer")
    ap.add_argument("source")
    ap.add_argument("output_dir")
    ap.add_argument("--model-path", default="crypto_pump_v2_large.joblib")
    ap.add_argument("--validation-fraction", type=float, default=0.15)
    ap.add_argument("--classifier-iterations", type=int, default=250)
    ap.add_argument("--classifier-depth", type=int, default=6)
    ap.add_argument("--ranker-iterations", type=int, default=220)
    ap.add_argument("--ranker-depth", type=int, default=5)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    stage_report_path = outdir / "stage_report.json"
    report: dict[str, object] = {"status": "running", "stages": [], "errors": []}

    d = pd.read_parquet(args.source)
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
    times = np.array(sorted(d["timestamp"].unique()))
    cut = max(1, int(len(times) * (1.0 - args.validation_fraction)))
    purge = max(s.horizon_bars for s in DEFAULT_HORIZONS)
    if cut <= purge:
        raise RuntimeError("not enough timestamps for 24h purge before validation")
    train_times = set(times[: cut - purge])
    val_times = set(times[cut:])
    train = d[d["timestamp"].isin(train_times)].copy()
    val = d[d["timestamp"].isin(val_times)].copy()
    del d
    gc.collect()

    features = v2_numeric_feature_columns(train)
    social_features = social_feature_columns(train)
    report.update({
        "rows_train": int(len(train)),
        "rows_validation": int(len(val)),
        "timestamps_total": int(len(times)),
        "train_timestamp_max": str(max(train_times)),
        "validation_timestamp_min": str(min(val_times)),
        "purge_bars": int(purge),
        "feature_count": int(len(features)),
        "social_feature_count": int(len(social_features)),
        "classifier_iterations": args.classifier_iterations,
        "ranker_iterations": args.ranker_iterations,
    })
    _write_json(stage_report_path, report)

    tabular = MultiHorizonTabularBranch(kind="catboost", features=features)
    ranker = CatBoostRankerBranch(kind="catboost", features=features)
    social = SocialBranch(kind="catboost", features=[])

    for spec in DEFAULT_HORIZONS:
        path = outdir / "classifiers" / f"{spec.name}.cbm"
        try:
            model = _save_or_load_classifier(
                train, features, spec, path,
                args.classifier_iterations, args.classifier_depth,
                args.learning_rate, args.threads, args.resume,
            )
            tabular.models[spec.target_col] = model
            report["stages"].append({"stage": "classifier", "horizon": spec.name, "status": "ok", "path": str(path)})
        except Exception as exc:
            report["errors"].append({"stage": "classifier", "horizon": spec.name, "error": repr(exc), "traceback": traceback.format_exc()})
        _write_json(stage_report_path, report)
        gc.collect()

    for spec in DEFAULT_HORIZONS:
        path = outdir / "rankers" / f"{spec.name}.cbm"
        try:
            model = _save_or_load_ranker(
                train, features, spec, path,
                args.ranker_iterations, args.ranker_depth,
                args.learning_rate, args.threads, args.resume,
            )
            ranker.models[spec.target_col] = model
            ranker.model_kinds[spec.target_col] = "catboost_yetirank"
            report["stages"].append({"stage": "ranker", "horizon": spec.name, "status": "ok", "path": str(path)})
        except Exception as exc:
            report["errors"].append({"stage": "ranker", "horizon": spec.name, "error": repr(exc), "traceback": traceback.format_exc()})
        _write_json(stage_report_path, report)
        gc.collect()

    missing = [
        spec.name for spec in DEFAULT_HORIZONS
        if spec.target_col not in tabular.models or spec.target_col not in ranker.models
    ]
    if missing:
        report["status"] = "partial"
        report["missing_horizons"] = missing
        _write_json(stage_report_path, report)
        print(json.dumps(report, indent=2), flush=True)
        sys.exit(2)

    scores = pd.DataFrame(index=val.index)
    for spec in DEFAULT_HORIZONS:
        try:
            scores[f"tabular__{spec.target_col}"] = _predict_classifier(tabular.models[spec.target_col], val, features)
            scores[f"ranker__{spec.target_col}"] = _predict_ranker(ranker.models[spec.target_col], val, features)
            report["stages"].append({"stage": "validation_scores", "horizon": spec.name, "status": "ok"})
        except Exception as exc:
            report["errors"].append({"stage": "validation_scores", "horizon": spec.name, "error": repr(exc), "traceback": traceback.format_exc()})
        _write_json(stage_report_path, report)
    if "bocd_cp_prob" in val:
        scores["bocd_cp_prob"] = pd.to_numeric(val["bocd_cp_prob"], errors="coerce").astype(np.float32)

    if any(f"tabular__{s.target_col}" not in scores or f"ranker__{s.target_col}" not in scores for s in DEFAULT_HORIZONS):
        report["status"] = "partial"
        _write_json(stage_report_path, report)
        sys.exit(3)

    stacker = CalibratedStacker().fit(scores, val, DEFAULT_HORIZONS)
    joblib.dump(stacker, outdir / "stacker.joblib", compress=3)
    report["stages"].append({"stage": "stacker", "status": "ok"})
    _write_json(stage_report_path, report)

    ensemble = V2Ensemble(model_kind="catboost")
    ensemble.tabular = tabular
    ensemble.ranker = ranker
    ensemble.social = social
    ensemble.stacker = stacker
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = outdir / model_path
    ensemble.save(model_path)
    report["stages"].append({"stage": "ensemble", "status": "ok", "path": str(model_path)})
    report["status"] = "complete"
    _write_json(stage_report_path, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
