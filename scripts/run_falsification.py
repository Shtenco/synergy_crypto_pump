#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from crypto_pump.v2.falsification import run_falsification


def main() -> None:
    ap = argparse.ArgumentParser(description="Falsify suspiciously strong crypto-pump holdout scores")
    ap.add_argument("predictions")
    ap.add_argument("output")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--pre-pump-threshold", type=float, default=0.05)
    ap.add_argument("--placebo-shift-bars", type=int, default=2016)
    args = ap.parse_args()

    d = pd.read_parquet(args.predictions)
    report = run_falsification(
        d,
        top_k=args.top_k,
        pre_pump_threshold=args.pre_pump_threshold,
        placebo_shift_bars=args.placebo_shift_bars,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
