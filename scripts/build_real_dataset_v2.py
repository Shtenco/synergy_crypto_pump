#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from crypto_pump.data_sources import collect_latoken_panel, collect_uniswap_v3_panel, merge_and_regularize_panels
from crypto_pump.data_sources.common import write_table


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect real LATOKEN + Uniswap data and build a point-in-time 5m panel")
    ap.add_argument("--out", default="data/real_v2/panel.parquet")
    ap.add_argument("--report", default="artifacts/real_v2/data_report.json")
    ap.add_argument("--latoken-days", type=int, default=14)
    ap.add_argument("--latoken-max-pairs", type=int, default=120)
    ap.add_argument("--uniswap-blocks", type=int, default=80000)
    ap.add_argument("--uniswap-max-pools", type=int, default=100)
    ap.add_argument("--min-bars", type=int, default=600)
    ap.add_argument("--skip-latoken", action="store_true")
    ap.add_argument("--skip-uniswap", action="store_true")
    ap.add_argument("--strict-both", action="store_true")
    args = ap.parse_args()

    panels: list[pd.DataFrame] = []
    reports: list[dict] = []
    if not args.skip_latoken:
        try:
            lat, rep = collect_latoken_panel(days=args.latoken_days, max_pairs=args.latoken_max_pairs)
            reports.append(rep)
            if len(lat):
                panels.append(lat)
        except Exception as exc:
            reports.append({"source": "latoken", "error": str(exc)})
    if not args.skip_uniswap:
        rpc = os.getenv("ETH_RPC_URL", "https://eth.llamarpc.com")
        try:
            uni, rep = collect_uniswap_v3_panel(rpc_url=rpc, lookback_blocks=args.uniswap_blocks, max_pools=args.uniswap_max_pools, workdir="data/real_v2/cryo")
            reports.append(rep)
            if len(uni):
                panels.append(uni)
        except Exception as exc:
            reports.append({"source": "uniswap_cryo", "error": str(exc)})

    success_sources = {r.get("source") for r in reports if not r.get("error") and ((r.get("rows", 0) or r.get("bar_rows", 0)) > 0)}
    if args.strict_both and not {"latoken", "uniswap_cryo"}.issubset(success_sources):
        raise SystemExit(f"strict-both failed; successful={sorted(success_sources)} reports={reports}")

    panel = merge_and_regularize_panels(panels, min_bars=args.min_bars)
    write_table(panel, args.out)
    report = {
        "sources": reports,
        "panel_rows": len(panel),
        "panel_symbols": int(panel["symbol"].nunique()),
        "timestamp_min": str(panel["timestamp"].min()),
        "timestamp_max": str(panel["timestamp"].max()),
        "latoken_symbols": int(panel[panel["source"].eq("latoken")]["symbol"].nunique()) if "source" in panel else 0,
        "uniswap_symbols": int(panel[panel["source"].eq("uniswap")]["symbol"].nunique()) if "source" in panel else 0,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
