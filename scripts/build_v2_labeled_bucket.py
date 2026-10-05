#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from crypto_pump.v2.features import V2FeatureBuilder
from crypto_pump.v2.labels import add_multi_horizon_labels


def discover_symbols(path: Path) -> list[str]:
    symbols: set[str] = set()
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=["symbol"], batch_size=250_000):
        symbols.update(x for x in batch.column(0).to_pylist() if x is not None)
    return sorted(symbols)


def main() -> None:
    ap = argparse.ArgumentParser(description="Exact historical V2 features+labels for one symbol bucket")
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--bucket-index", type=int, required=True)
    ap.add_argument("--bucket-count", type=int, required=True)
    ap.add_argument("--benchmark-symbol", default="LATOKEN:BTCUSDT")
    args = ap.parse_args()

    if not 0 <= args.bucket_index < args.bucket_count:
        raise ValueError("bucket-index must be in [0,bucket-count)")

    symbols = discover_symbols(args.source)
    chosen = symbols[args.bucket_index :: args.bucket_count]
    if not chosen:
        raise RuntimeError("empty symbol bucket")

    wanted = list(chosen)
    if args.benchmark_symbol in symbols and args.benchmark_symbol not in wanted:
        wanted.append(args.benchmark_symbol)

    dataset = ds.dataset(str(args.source), format="parquet")
    table = dataset.to_table(filter=ds.field("symbol").isin(wanted))
    frame = table.to_pandas(split_blocks=True, self_destruct=True)

    transformed = V2FeatureBuilder(benchmark_symbol=args.benchmark_symbol).transform(frame)
    transformed = transformed[transformed["symbol"].isin(chosen)].copy()
    labeled = add_multi_horizon_labels(transformed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(args.output, index=False, compression="zstd")
    print({
        "bucket_index": args.bucket_index,
        "bucket_count": args.bucket_count,
        "symbols": len(chosen),
        "rows": len(labeled),
        "timestamp_min": str(labeled["timestamp"].min()),
        "timestamp_max": str(labeled["timestamp"].max()),
        "output": str(args.output),
    })


if __name__ == "__main__":
    main()
