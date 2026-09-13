from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from crypto_pump.v2.features import V2FeatureBuilder
from crypto_pump.v3.model import DEFAULT_TRADING_SPECS, V3TradingEnsemble, add_v3_training_labels
from crypto_pump.v3.validation import walk_forward_v3


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)


def require_execution_liquidity(frame: pd.DataFrame, *, min_coverage: float = 0.95) -> dict[str, float]:
    if "liquidity_usd" not in frame.columns:
        raise ValueError("V3 execution OOS requires point-in-time liquidity_usd; enrich the historical panel before training")
    liq = pd.to_numeric(frame["liquidity_usd"], errors="coerce")
    valid = liq.notna() & (liq > 0)
    coverage = float(valid.mean()) if len(frame) else 0.0
    if coverage < min_coverage:
        raise ValueError(f"V3 liquidity_usd coverage {coverage:.3%} is below required {min_coverage:.0%}")
    values = liq[valid]
    return {
        "coverage": coverage,
        "min_usd": float(values.min()),
        "median_usd": float(values.median()),
        "p95_usd": float(values.quantile(0.95)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train V3 directly on TP-before-SL first-passage outcomes")
    p.add_argument("source", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--model-kind", default="auto")
    p.add_argument("--n-splits", type=int, default=3)
    p.add_argument("--probability-threshold", type=float, default=0.70)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--notional-usd", type=float, default=50.0)
    p.add_argument("--already-featured", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = read_table(args.source)
    raw_liquidity = require_execution_liquidity(raw)
    featured = raw.copy() if args.already_featured else V2FeatureBuilder(benchmark_symbol="__NO_BENCHMARK__").transform(raw)
    if "liquidity_usd" not in featured.columns and "liquidity_usd" in raw.columns and len(featured) == len(raw):
        # V2 feature builders should preserve source columns; keep an explicit safe fallback
        # only when row cardinality is unchanged.
        featured["liquidity_usd"] = pd.to_numeric(raw["liquidity_usd"], errors="coerce").to_numpy()
    featured_liquidity = require_execution_liquidity(featured)
    labeled = add_v3_training_labels(featured, entry_delay_bars=1)
    labeled_liquidity = require_execution_liquidity(labeled)
    labeled.to_parquet(args.output_dir / "v3_labeled_features.parquet", index=False)

    _, _, report = walk_forward_v3(
        labeled,
        args.output_dir / "oos",
        specs=DEFAULT_TRADING_SPECS,
        n_splits=args.n_splits,
        model_kind=args.model_kind,
        probability_threshold=args.probability_threshold,
        top_k=args.top_k,
        notional_usd=args.notional_usd,
    )

    times = sorted(pd.to_datetime(labeled["timestamp"], utc=True).unique())
    purge = max(s.horizon_bars for s in DEFAULT_TRADING_SPECS) + 1
    validation_size = max(24, int(len(times) * 0.15))
    validation_start = len(times) - validation_size
    train_end = validation_start - purge
    if train_end <= 0:
        raise ValueError("not enough history for final V3 fit after 24h purge")
    ts = pd.to_datetime(labeled["timestamp"], utc=True)
    train = labeled[ts.isin(times[:train_end])].copy()
    validation = labeled[ts.isin(times[validation_start:])].copy()
    final = V3TradingEnsemble(model_kind=args.model_kind).fit(train, validation)
    final_path = args.output_dir / "v3_trading_model.joblib"
    final.save(final_path)

    reference_cols = ["timestamp", "symbol", *final.features]
    reference = labeled[reference_cols].tail(min(len(labeled), 100_000)).copy()
    reference.to_parquet(args.output_dir / "v3_reference_features.parquet", index=False)
    summary = {
        "model": str(final_path),
        "reference_features": str(args.output_dir / "v3_reference_features.parquet"),
        "feature_count": len(final.features),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "liquidity_evidence": {
            "raw": raw_liquidity,
            "featured": featured_liquidity,
            "labeled": labeled_liquidity,
        },
        "oos_report": report,
    }
    (args.output_dir / "v3_train_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
