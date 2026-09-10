#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, ndcg_score

from crypto_pump.v2.config import DEFAULT_HORIZONS
from crypto_pump.v2.features import v2_numeric_feature_columns


def main() -> None:
    ap = argparse.ArgumentParser(description="Resource-aware full-dataset CatBoost baseline with strict temporal purge")
    ap.add_argument("source")
    ap.add_argument("output_dir")
    ap.add_argument("--iterations", type=int, default=150)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.06)
    ap.add_argument("--test-fraction", type=float, default=0.25)
    ap.add_argument("--embargo-bars", type=int, default=12)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    from catboost import CatBoostClassifier

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(args.source)
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)

    features = v2_numeric_feature_columns(d)
    times = np.array(sorted(d["timestamp"].unique()))
    purge = max(s.horizon_bars for s in DEFAULT_HORIZONS)
    split = int(len(times) * (1.0 - args.test_fraction))
    train_end = split - purge - args.embargo_bars
    if train_end < 100:
        raise RuntimeError("not enough history for requested purge/embargo")
    train_times = set(times[:train_end])
    test_times = set(times[split:])
    train_mask_time = d["timestamp"].isin(train_times).to_numpy()
    test_mask_time = d["timestamp"].isin(test_times).to_numpy()

    report: dict[str, object] = {
        "rows": int(len(d)),
        "segmented_symbols": int(d["symbol"].nunique()),
        "unique_timestamps": int(len(times)),
        "feature_count": int(len(features)),
        "iterations": args.iterations,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "purge_bars": purge,
        "embargo_bars": args.embargo_bars,
        "train_timestamp_max": str(times[train_end - 1]),
        "test_timestamp_min": str(times[split]),
        "targets": {},
    }

    # Materialize one float32 matrix at a time to keep peak RAM bounded on
    # GitHub-hosted runners. Feature names remain stable for saved CatBoost models.
    for spec in DEFAULT_HORIZONS:
        valid = d[spec.target_col].notna().to_numpy()
        tr = train_mask_time & valid
        te = test_mask_time & valid
        y_train = d.loc[tr, spec.target_col].astype(np.int8).to_numpy()
        y_test = d.loc[te, spec.target_col].astype(np.int8).to_numpy()
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            report["targets"][spec.name] = {
                "error": "target has fewer than two classes in train or test",
                "train_rows": int(tr.sum()), "test_rows": int(te.sum()),
                "train_positives": int(y_train.sum()), "test_positives": int(y_test.sum()),
            }
            continue

        x_train = d.loc[tr, features].astype(np.float32)
        x_test = d.loc[te, features].astype(np.float32)
        model = CatBoostClassifier(
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.learning_rate,
            loss_function="Logloss",
            eval_metric="PRAUC",
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=25,
            allow_writing_files=False,
            thread_count=-1,
            border_count=128,
            task_type="CPU",
        )
        model.fit(x_train, y_train)
        p = model.predict_proba(x_test)[:, 1]
        base_rate = float(y_test.mean())
        metrics = {
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "train_positives": int(y_train.sum()),
            "test_positives": int(y_test.sum()),
            "test_positive_rate": base_rate,
            "pr_auc": float(average_precision_score(y_test, p)),
            "roc_auc": float(roc_auc_score(y_test, p)),
            "brier": float(brier_score_loss(y_test, p)),
            "log_loss": float(log_loss(y_test, p, labels=[0, 1])),
        }

        eval_df = d.loc[te, ["timestamp", "symbol", spec.target_col, spec.mfe_col]].copy()
        eval_df["p"] = p
        top = (
            eval_df.sort_values(["timestamp", "p"], ascending=[True, False])
            .groupby("timestamp", sort=False)
            .head(args.top_k)
        )
        precision = float(top[spec.target_col].astype(float).mean()) if len(top) else float("nan")
        metrics[f"precision_at_{args.top_k}"] = precision
        metrics[f"lift_at_{args.top_k}"] = float(precision / base_rate) if base_rate > 0 and np.isfinite(precision) else float("nan")

        ndcgs: list[float] = []
        for _, g in eval_df.groupby("timestamp", sort=False):
            if len(g) < 2:
                continue
            rel = np.maximum(pd.to_numeric(g[spec.mfe_col], errors="coerce").to_numpy(float), 0.0)
            score = g["p"].to_numpy(float)
            if np.isfinite(rel).all() and np.isfinite(score).all() and rel.max() > 0:
                ndcgs.append(float(ndcg_score([rel], [score], k=min(args.top_k, len(g)))))
        if ndcgs:
            metrics[f"ndcg_at_{args.top_k}"] = float(np.mean(ndcgs))

        model_path = outdir / f"catboost_baseline_{spec.name}.cbm"
        model.save_model(model_path)
        report["targets"][spec.name] = metrics
        print(json.dumps({spec.name: metrics}, indent=2), flush=True)

        del x_train, x_test, model, p, eval_df, top, y_train, y_test
        gc.collect()

    (outdir / "baseline_oos_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
