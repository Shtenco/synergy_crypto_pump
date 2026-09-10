#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}


def _read_panels(root: Path) -> tuple[list[pd.DataFrame], list[str], list[str]]:
    frames: list[pd.DataFrame] = []
    used: list[str] = []
    skipped: list[str] = []
    for path in sorted(root.rglob("panel.parquet")):
        try:
            df = pd.read_parquet(path)
            missing = REQUIRED.difference(df.columns)
            if df.empty or missing:
                skipped.append(f"{path}: empty/missing={sorted(missing)}")
                continue
            df = df.copy()
            df["_shard_path"] = str(path)
            frames.append(df)
            used.append(str(path))
        except Exception as exc:
            skipped.append(f"{path}: {type(exc).__name__}: {exc}")
    return frames, used, skipped


def _segment_gaps(df: pd.DataFrame, expected_minutes: int = 5) -> pd.DataFrame:
    """Split symbol history at missing time gaps so labels never jump across outages."""
    pieces: list[pd.DataFrame] = []
    expected = pd.Timedelta(minutes=expected_minutes)
    for symbol, part in df.groupby("symbol", sort=False):
        p = part.sort_values("timestamp", kind="mergesort").copy()
        p["base_symbol"] = symbol
        # A duplicated timestamp is handled before this function. Anything >5m
        # starts a new independent series; labels/features cannot cross that gap.
        segment = p["timestamp"].diff().gt(expected).fillna(False).cumsum().astype(int)
        p["series_segment"] = segment
        p["symbol"] = [symbol if s == 0 else f"{symbol}#SEG{s:03d}" for s in segment]
        pieces.append(p)
    return pd.concat(pieces, ignore_index=True, sort=False) if pieces else df.iloc[:0].copy()


def merge_shards(root: Path, *, min_bars: int = 600) -> tuple[pd.DataFrame, dict]:
    frames, used, skipped = _read_panels(root)
    if not frames:
        raise ValueError(f"no usable panel.parquet shards found under {root}; skipped={skipped[:20]}")

    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.floor("5min")
    raw = raw.dropna(subset=["timestamp", "symbol", "open", "high", "low", "close"])
    raw = raw.sort_values(["symbol", "timestamp", "_shard_path"], kind="mergesort")
    before_dedupe = len(raw)
    raw = raw.drop_duplicates(["symbol", "timestamp"], keep="last")

    segmented = _segment_gaps(raw)
    sizes = segmented.groupby("symbol", sort=False).size()
    keep = set(sizes[sizes >= min_bars].index)
    dropped = sizes[sizes < min_bars]
    out = segmented[segmented["symbol"].isin(keep)].copy()
    out = out.drop(columns=["_shard_path"], errors="ignore")
    out = out.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
    if out.empty:
        raise ValueError(
            f"no contiguous series survive min_bars={min_bars}; "
            f"largest={sizes.sort_values(ascending=False).head(20).to_dict()}"
        )

    src = out["source"].astype(str) if "source" in out else pd.Series("unknown", index=out.index)
    report = {
        "input_shards": len(used),
        "used_shards": used,
        "skipped_shards": skipped,
        "rows_before_dedupe": int(before_dedupe),
        "rows_after_dedupe": int(len(raw)),
        "rows_final": int(len(out)),
        "base_symbols": int(out["base_symbol"].nunique()),
        "contiguous_series": int(out["symbol"].nunique()),
        "dropped_short_series": int(len(dropped)),
        "min_bars": int(min_bars),
        "timestamp_min": str(out["timestamp"].min()),
        "timestamp_max": str(out["timestamp"].max()),
        "source_rows": {str(k): int(v) for k, v in src.value_counts().to_dict().items()},
        "source_symbols": {
            str(k): int(g["base_symbol"].nunique())
            for k, g in out.assign(_source=src).groupby("_source")
        },
        "largest_series": {
            str(k): int(v)
            for k, v in out.groupby("symbol").size().sort_values(ascending=False).head(25).to_dict().items()
        },
    }
    return out, report


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge LATOKEN/Uniswap shard artifacts without crossing missing-data gaps")
    ap.add_argument("root")
    ap.add_argument("--out", default="data/real_v2/panel.parquet")
    ap.add_argument("--report", default="artifacts/real_v2/data_report.json")
    ap.add_argument("--min-bars", type=int, default=600)
    args = ap.parse_args()

    panel, report = merge_shards(Path(args.root), min_bars=args.min_bars)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out, index=False)
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
