#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import tempfile
from pathlib import Path

import duckdb
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, ndcg_score

from crypto_pump.v2.config import DEFAULT_HORIZONS


def q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def fill_train_memmaps(con, source_sql: str, features: list[str], target: str, train_max: str, workdir: Path):
    where = f"timestamp <= TIMESTAMPTZ '{train_max}' AND {q(target)} IS NOT NULL"
    n = int(con.execute(f"SELECT COUNT(*) FROM {source_sql} WHERE {where}").fetchone()[0])
    x_path = workdir / f"{target}.x.f32"
    y_path = workdir / f"{target}.y.i8"
    x = np.memmap(x_path, mode="w+", dtype=np.float32, shape=(n, len(features)))
    y = np.memmap(y_path, mode="w+", dtype=np.int8, shape=(n,))
    select = ", ".join([f"CAST({q(c)} AS FLOAT) AS {q(c)}" for c in features] + [f"CAST({q(target)} AS TINYINT) AS __y"])
    reader = con.execute(
        f"SELECT {select} FROM {source_sql} WHERE {where} ORDER BY timestamp, symbol"
    ).fetch_record_batch(50_000)
    pos = 0
    for batch in reader:
        m = batch.num_rows
        for j in range(len(features)):
            x[pos:pos+m, j] = batch.column(j).to_numpy(zero_copy_only=False)
        y[pos:pos+m] = batch.column(len(features)).to_numpy(zero_copy_only=False)
        pos += m
    if pos != n:
        raise RuntimeError(f"train row fill mismatch {pos} != {n}")
    x.flush(); y.flush()
    return x, y


def predict_test_batched(con, source_sql: str, features: list[str], spec, test_min: str, model: CatBoostClassifier):
    where = f"timestamp >= TIMESTAMPTZ '{test_min}' AND {q(spec.target_col)} IS NOT NULL"
    n = int(con.execute(f"SELECT COUNT(*) FROM {source_sql} WHERE {where}").fetchone()[0])
    y = np.empty(n, dtype=np.int8)
    p = np.empty(n, dtype=np.float64)
    mfe = np.empty(n, dtype=np.float32)
    ts = np.empty(n, dtype="datetime64[ns]")
    select = ", ".join(
        ["timestamp", f"CAST({q(spec.target_col)} AS TINYINT) AS __y", f"CAST({q(spec.mfe_col)} AS FLOAT) AS __mfe"]
        + [f"CAST({q(c)} AS FLOAT) AS {q(c)}" for c in features]
    )
    reader = con.execute(
        f"SELECT {select} FROM {source_sql} WHERE {where} ORDER BY timestamp, symbol"
    ).fetch_record_batch(50_000)
    pos = 0
    for batch in reader:
        m = batch.num_rows
        xb = np.empty((m, len(features)), dtype=np.float32)
        for j in range(len(features)):
            xb[:, j] = batch.column(3+j).to_numpy(zero_copy_only=False)
        y[pos:pos+m] = batch.column(1).to_numpy(zero_copy_only=False)
        mfe[pos:pos+m] = batch.column(2).to_numpy(zero_copy_only=False)
        ts[pos:pos+m] = batch.column(0).to_numpy(zero_copy_only=False).astype("datetime64[ns]")
        p[pos:pos+m] = model.predict_proba(xb)[:, 1]
        pos += m
    if pos != n:
        raise RuntimeError(f"test row fill mismatch {pos} != {n}")
    return ts, y, mfe, p


def ranking_metrics(ts: np.ndarray, y: np.ndarray, mfe: np.ndarray, p: np.ndarray, top_k: int):
    if not len(y):
        return float("nan"), None
    boundaries = np.r_[0, np.flatnonzero(ts[1:] != ts[:-1]) + 1, len(ts)]
    top_hits: list[float] = []
    ndcgs: list[float] = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if b <= a:
            continue
        order = np.argsort(-p[a:b], kind="stable")[:top_k]
        top_hits.extend(y[a:b][order].astype(float).tolist())
        if b - a >= 2:
            rel = np.maximum(mfe[a:b].astype(float), 0.0)
            score = p[a:b].astype(float)
            if np.isfinite(rel).all() and np.isfinite(score).all() and rel.max() > 0:
                ndcgs.append(float(ndcg_score([rel], [score], k=min(top_k, b-a))))
    precision = float(np.mean(top_hits)) if top_hits else float("nan")
    return precision, (float(np.mean(ndcgs)) if ndcgs else None)


def main() -> None:
    ap = argparse.ArgumentParser(description="Memory-bounded exact historical 90d CatBoost baseline")
    ap.add_argument("source_glob")
    ap.add_argument("reference_model", help="Historical 30d CatBoost .cbm used only to recover exact feature names/order")
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--iterations", type=int, default=150)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.06)
    ap.add_argument("--test-fraction", type=float, default=0.25)
    ap.add_argument("--embargo-bars", type=int, default=12)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--memory-limit", default="5GB")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ref = CatBoostClassifier(); ref.load_model(args.reference_model)
    features = list(ref.feature_names_)
    if not features:
        raise RuntimeError("reference model does not expose feature names")

    con = duckdb.connect(str(args.output_dir / "training.duckdb"))
    con.execute("PRAGMA threads=4")
    con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
    con.execute(f"PRAGMA temp_directory='{str(args.output_dir / 'ducktmp')}'")
    glob_sql = args.source_glob.replace("'", "''")
    source_sql = f"read_parquet('{glob_sql}', union_by_name=true)"

    times = [r[0] for r in con.execute(f"SELECT DISTINCT timestamp FROM {source_sql} ORDER BY timestamp").fetchall()]
    purge = max(s.horizon_bars for s in DEFAULT_HORIZONS)
    split = int(len(times) * (1.0 - args.test_fraction))
    train_end = split - purge - args.embargo_bars
    if train_end < 100:
        raise RuntimeError("not enough history for requested purge/embargo")
    train_max = str(times[train_end - 1])
    test_min = str(times[split])
    rows = int(con.execute(f"SELECT COUNT(*) FROM {source_sql}").fetchone()[0])
    symbols = int(con.execute(f"SELECT COUNT(DISTINCT symbol) FROM {source_sql}").fetchone()[0])

    report = {
        "rows": rows,
        "segmented_symbols": symbols,
        "unique_timestamps": len(times),
        "feature_count": len(features),
        "features": features,
        "iterations": args.iterations,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "purge_bars": purge,
        "embargo_bars": args.embargo_bars,
        "train_timestamp_max": train_max,
        "test_timestamp_min": test_min,
        "targets": {},
    }

    with tempfile.TemporaryDirectory(dir=args.output_dir) as tmp:
        workdir = Path(tmp)
        for spec in DEFAULT_HORIZONS:
            x_train, y_train = fill_train_memmaps(con, source_sql, features, spec.target_col, train_max, workdir)
            train_pos = int(np.asarray(y_train).sum())
            if len(np.unique(np.asarray(y_train))) < 2:
                report["targets"][spec.name] = {"error": "target has fewer than two classes in train", "train_rows": len(y_train), "train_positives": train_pos}
                continue

            train_pool = Pool(x_train, label=y_train, feature_names=features)
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
            model.fit(train_pool)
            model_path = args.output_dir / f"catboost_baseline_{spec.name}.cbm"
            model.save_model(model_path)

            ts, y_test, mfe, pred = predict_test_batched(con, source_sql, features, spec, test_min, model)
            base_rate = float(y_test.mean())
            precision, ndcg = ranking_metrics(ts, y_test, mfe, pred, args.top_k)
            metrics = {
                "train_rows": int(len(y_train)),
                "test_rows": int(len(y_test)),
                "train_positives": train_pos,
                "test_positives": int(y_test.sum()),
                "test_positive_rate": base_rate,
                "pr_auc": float(average_precision_score(y_test, pred)),
                "roc_auc": float(roc_auc_score(y_test, pred)),
                "brier": float(brier_score_loss(y_test, pred)),
                "log_loss": float(log_loss(y_test, pred, labels=[0, 1])),
                f"precision_at_{args.top_k}": precision,
                f"lift_at_{args.top_k}": float(precision / base_rate) if base_rate > 0 and np.isfinite(precision) else float("nan"),
            }
            if ndcg is not None:
                metrics[f"ndcg_at_{args.top_k}"] = ndcg
            report["targets"][spec.name] = metrics
            print(json.dumps({spec.name: metrics}, indent=2), flush=True)

            del train_pool, model, x_train, y_train, ts, y_test, mfe, pred
            gc.collect()

    con.close()
    (args.output_dir / "baseline_oos_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
