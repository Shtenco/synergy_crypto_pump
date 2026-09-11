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
from crypto_pump.v2.features import v2_numeric_feature_columns
from crypto_pump.v2.pipeline import V2Ensemble, evaluate_v2_predictions
from crypto_pump.v2.splits import purged_walk_forward_folds


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _fit_or_load_classifier(train, features, spec, path, iterations, depth, lr, threads, resume):
    from catboost import CatBoostClassifier

    if resume and path.exists():
        model = CatBoostClassifier()
        model.load_model(path)
        print(json.dumps({"resume": "classifier", "horizon": spec.name, "path": str(path)}), flush=True)
        return model
    use = train[spec.target_col].notna().to_numpy()
    y = train.loc[use, spec.target_col].astype(np.int8).to_numpy()
    print(json.dumps({"stage": "classifier", "horizon": spec.name, "rows": int(use.sum()), "features": len(features)}), flush=True)
    x = train.loc[use, features].astype(np.float32)
    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=lr,
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


def _fit_or_load_ranker(train, features, spec, path, iterations, depth, lr, threads, resume):
    from catboost import CatBoostRanker, Pool

    if resume and path.exists():
        model = CatBoostRanker()
        model.load_model(path)
        print(json.dumps({"resume": "ranker", "horizon": spec.name, "path": str(path)}), flush=True)
        return model
    use = train[spec.mfe_col].notna().to_numpy()
    meta = train.loc[use, ["timestamp", spec.mfe_col]]
    relevance = meta.groupby("timestamp", sort=False)[spec.mfe_col].rank(pct=True, ascending=True).to_numpy(np.float32)
    group_id = pd.factorize(meta["timestamp"], sort=False)[0].astype(np.int32, copy=False)
    print(json.dumps({"stage": "ranker", "horizon": spec.name, "rows": int(use.sum()), "features": len(features), "groups": int(group_id[-1] + 1) if len(group_id) else 0}), flush=True)
    x = train.loc[use, features].astype(np.float32)
    pool = Pool(x, label=relevance, group_id=group_id)
    model = CatBoostRanker(
        loss_function="YetiRank:mode=NDCG",
        eval_metric="NDCG:top=10",
        iterations=iterations,
        depth=depth,
        learning_rate=lr,
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
    del meta, relevance, group_id, x, pool
    gc.collect()
    return model


def _classifier_scores(models, frame, features):
    out = pd.DataFrame(index=frame.index)
    for spec in DEFAULT_HORIZONS:
        x = frame[features].astype(np.float32)
        out[f"tabular__{spec.target_col}"] = np.asarray(models[spec.target_col].predict_proba(x)[:, 1], dtype=np.float32)
        del x
        gc.collect()
    return out


def _ranker_scores(models, frame, features):
    out = pd.DataFrame(index=frame.index)
    for spec in DEFAULT_HORIZONS:
        x = frame[features].astype(np.float32)
        raw = np.asarray(models[spec.target_col].predict(x), dtype=np.float32)
        del x
        tmp = pd.DataFrame({"timestamp": frame["timestamp"].to_numpy(), "raw": raw}, index=frame.index)
        out[f"ranker__{spec.target_col}"] = tmp.groupby("timestamp", sort=False)["raw"].rank(pct=True).to_numpy(np.float32)
        del tmp, raw
        gc.collect()
    return out


def _base_scores(tabular, ranker, frame, features):
    out = pd.concat([
        _classifier_scores(tabular.models, frame, features),
        _ranker_scores(ranker.models, frame, features),
    ], axis=1)
    if "bocd_cp_prob" in frame and frame["bocd_cp_prob"].notna().any():
        out["bocd_cp_prob"] = pd.to_numeric(frame["bocd_cp_prob"], errors="coerce").astype(np.float32)
    return out


def _make_prediction_frame(test, base_scores, final, fold_no):
    pred = pd.concat([base_scores, final], axis=1)
    pred["fold"] = int(fold_no)
    pred["timestamp"] = test["timestamp"].to_numpy()
    pred["symbol"] = test["symbol"].to_numpy()
    for spec in DEFAULT_HORIZONS:
        pred[spec.target_col] = test[spec.target_col].to_numpy()
        pred[spec.mfe_col] = test[spec.mfe_col].to_numpy()
        pred[spec.mae_col] = test[spec.mae_col].to_numpy()
    recent = [c for c in ("current_candle_return", "ret_1", "ret_3", "ret_6", "ret_12") if c in test.columns]
    for c in recent:
        pred[c] = test[c].to_numpy()
    return pred.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Checkpointable purged walk-forward CatBoost + YetiRank V2")
    ap.add_argument("source")
    ap.add_argument("output_dir")
    ap.add_argument("--n-splits", type=int, default=2)
    ap.add_argument("--min-train-fraction", type=float, default=0.45)
    ap.add_argument("--validation-fraction", type=float, default=0.10)
    ap.add_argument("--purge-bars", type=int, default=288)
    ap.add_argument("--embargo-bars", type=int, default=12)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--classifier-iterations", type=int, default=160)
    ap.add_argument("--classifier-depth", type=int, default=6)
    ap.add_argument("--ranker-iterations", type=int, default=100)
    ap.add_argument("--ranker-depth", type=int, default=5)
    ap.add_argument("--learning-rate", type=float, default=0.06)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    report_path = outdir / "stage_report.json"
    report: dict[str, object] = {"status": "running", "folds": [], "errors": []}

    d = pd.read_parquet(args.source)
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
    features = v2_numeric_feature_columns(d)
    folds = purged_walk_forward_folds(
        d,
        n_splits=args.n_splits,
        min_train_fraction=args.min_train_fraction,
        validation_fraction=args.validation_fraction,
        purge_bars=args.purge_bars,
        embargo_bars=args.embargo_bars,
    )
    report.update({
        "rows": int(len(d)), "features": len(features), "n_splits": len(folds),
        "purge_bars": args.purge_bars, "embargo_bars": args.embargo_bars,
        "classifier_iterations": args.classifier_iterations, "ranker_iterations": args.ranker_iterations,
    })
    _write_json(report_path, report)

    fold_reports = []
    for fold in folds:
        fold_dir = outdir / f"fold_{fold.fold}"
        pred_path = fold_dir / "predictions.parquet"
        metrics_path = fold_dir / "metrics.json"
        if args.resume and pred_path.exists() and metrics_path.exists():
            fold_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            fold_reports.append({"fold": fold.fold, "metrics": fold_metrics, "resumed": True})
            print(json.dumps({"resume": "completed_fold", "fold": fold.fold}), flush=True)
            continue

        train, val, test = fold.frames(d)
        print(json.dumps({
            "fold": fold.fold,
            "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
            "train_max": str(train["timestamp"].max()),
            "validation_min": str(val["timestamp"].min()), "validation_max": str(val["timestamp"].max()),
            "test_min": str(test["timestamp"].min()), "test_max": str(test["timestamp"].max()),
        }), flush=True)
        tabular = MultiHorizonTabularBranch(kind="catboost", features=features)
        ranker = CatBoostRankerBranch(kind="catboost", features=features)

        try:
            for spec in DEFAULT_HORIZONS:
                path = fold_dir / "classifiers" / f"{spec.name}.cbm"
                tabular.models[spec.target_col] = _fit_or_load_classifier(
                    train, features, spec, path,
                    args.classifier_iterations, args.classifier_depth,
                    args.learning_rate, args.threads, args.resume,
                )
                report["folds"].append({"fold": fold.fold, "stage": "classifier", "horizon": spec.name, "status": "ok"})
                _write_json(report_path, report)

            for spec in DEFAULT_HORIZONS:
                path = fold_dir / "rankers" / f"{spec.name}.cbm"
                ranker.models[spec.target_col] = _fit_or_load_ranker(
                    train, features, spec, path,
                    args.ranker_iterations, args.ranker_depth,
                    args.learning_rate, args.threads, args.resume,
                )
                ranker.model_kinds[spec.target_col] = "catboost_yetirank"
                report["folds"].append({"fold": fold.fold, "stage": "ranker", "horizon": spec.name, "status": "ok"})
                _write_json(report_path, report)

            val_scores = _base_scores(tabular, ranker, val, features)
            stacker = CalibratedStacker().fit(val_scores, val, DEFAULT_HORIZONS)
            joblib.dump(stacker, fold_dir / "stacker.joblib", compress=3)
            del val_scores
            gc.collect()

            test_scores = _base_scores(tabular, ranker, test, features)
            final = stacker.predict(test_scores, DEFAULT_HORIZONS)
            pred = _make_prediction_frame(test, test_scores, final, fold.fold)
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            pred.to_parquet(pred_path, index=False)
            fold_metrics = evaluate_v2_predictions(test.reset_index(drop=True), pred.reset_index(drop=True), DEFAULT_HORIZONS, top_k=args.top_k)
            _write_json(metrics_path, fold_metrics)
            fold_reports.append({"fold": fold.fold, "metrics": fold_metrics})
            report["folds"].append({"fold": fold.fold, "stage": "complete", "status": "ok"})
            _write_json(report_path, report)
            print(json.dumps({"fold": fold.fold, "metrics": fold_metrics}, indent=2), flush=True)

            ensemble = V2Ensemble(model_kind="catboost")
            ensemble.tabular = tabular
            ensemble.ranker = ranker
            ensemble.social = SocialBranch(kind="catboost", features=[])
            ensemble.stacker = stacker
            ensemble.save(fold_dir / "model.joblib")

            del train, val, test, tabular, ranker, stacker, test_scores, final, pred, ensemble
            gc.collect()
        except Exception as exc:
            report["errors"].append({"fold": fold.fold, "error": repr(exc), "traceback": traceback.format_exc()})
            report["status"] = "partial"
            _write_json(report_path, report)
            print(traceback.format_exc(), flush=True)
            # Exit so Actions' `if: always()` artifact step can persist every
            # horizon checkpoint completed before the failure/timeout.
            sys.exit(2)

    pred_files = [outdir / f"fold_{f.fold}" / "predictions.parquet" for f in folds]
    if not all(p.exists() for p in pred_files):
        report["status"] = "partial"
        _write_json(report_path, report)
        sys.exit(3)

    oos = pd.concat([pd.read_parquet(p) for p in pred_files], ignore_index=True)
    label_cols = ["timestamp", "symbol"] + [c for s in DEFAULT_HORIZONS for c in (s.target_col, s.mfe_col, s.mae_col)]
    aggregate = evaluate_v2_predictions(oos[label_cols], oos, DEFAULT_HORIZONS, top_k=args.top_k)
    final_report = {
        "purge_bars": args.purge_bars,
        "embargo_bars": args.embargo_bars,
        "n_splits": len(folds),
        "architecture": {"tabular": True, "catboost_ranker": True, "social": False, "bocd": bool("bocd_cp_prob" in d and d["bocd_cp_prob"].notna().any())},
        "folds": fold_reports,
        "aggregate_oos": aggregate,
    }
    oos.to_parquet(outdir / "oos_predictions.parquet", index=False)
    _write_json(outdir / "oos_metrics.json", final_report)
    report["status"] = "complete"
    report["aggregate_oos"] = aggregate
    _write_json(report_path, report)
    print(json.dumps(final_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
