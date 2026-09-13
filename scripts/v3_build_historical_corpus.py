from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from crypto_pump.v3.corpus import build_clean_hf_panel, download_hf_corpus


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and build the cleaned PumpFun historical V3 panel")
    p.add_argument("--data-dir", type=Path, default=Path("data/v3/hf_pumpfun_corpus"))
    p.add_argument("--output", type=Path, default=Path("data/v3/pumpfun_historical_5m.parquet"))
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--memory-limit", default="8GB")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.no_download:
        download_hf_corpus(args.data_dir)
    report = build_clean_hf_panel(
        args.data_dir,
        args.output,
        start=args.start,
        end=args.end,
        threads=args.threads,
        memory_limit=args.memory_limit,
    )
    print(json.dumps(asdict(report), indent=2, default=str))


if __name__ == "__main__":
    main()
