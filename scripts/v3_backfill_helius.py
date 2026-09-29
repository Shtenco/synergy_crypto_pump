from __future__ import annotations

import argparse
import json
from pathlib import Path

from crypto_pump.v3.historical import HeliusHistoricalClient, utc_epoch


def main() -> None:
    p = argparse.ArgumentParser(description="Read-only chronological Helius historical backfill")
    p.add_argument("address", help="Pump/PumpSwap program, wallet, mint or other Solana address")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--signatures-only", action="store_true")
    p.add_argument("--max-pages", type=int)
    args = p.parse_args()

    client = HeliusHistoricalClient()
    rows = list(client.iter_transactions(
        args.address,
        start_time=utc_epoch(args.start) if args.start else None,
        end_time=utc_epoch(args.end) if args.end else None,
        full=not args.signatures_only,
        max_pages=args.max_pages,
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    print(json.dumps({"address": args.address, "rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
