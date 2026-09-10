#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from crypto_pump.data_sources import (
    collect_latoken_panel,
    collect_uniswap_v3_panel,
    merge_and_regularize_panels,
)
from crypto_pump.data_sources.common import write_table


def _save_report(path: str, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str), flush=True)


def _rpc_candidates() -> list[tuple[str, str]]:
    """Return ordered Ethereum RPC candidates without leaking secret URLs to reports."""
    candidates: list[tuple[str, str]] = []
    primary = (os.getenv("ETH_RPC_URL") or "").strip()
    if primary:
        candidates.append(("env_primary", primary))

    custom = (os.getenv("ETH_RPC_FALLBACKS") or "").strip()
    if custom:
        for i, url in enumerate(custom.split(",")):
            url = url.strip()
            if url:
                candidates.append((f"env_fallback_{i + 1}", url))

    # Keyless/public fallbacks documented by their providers. GitHub-hosted
    # runner egress is provider-dependent, so keep several independent networks.
    candidates.extend([
        ("drpc_public", "https://eth.drpc.org/"),
        ("envio_hyperrpc", "https://eth.rpc.hypersync.xyz"),
        ("cloudflare", "https://cloudflare-eth.com/v1/mainnet"),
        ("publicnode", "https://ethereum-rpc.publicnode.com"),
        ("1rpc", "https://public.1rpc.io/eth"),
        ("llamarpc", "https://eth.llamarpc.com"),
    ])

    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for label, url in candidates:
        if url in seen:
            continue
        seen.add(url)
        out.append((label, url))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect real LATOKEN + Uniswap data and build a point-in-time 5m panel")
    ap.add_argument("--out", default="data/real_v2/panel.parquet")
    ap.add_argument("--report", default="artifacts/real_v2/data_report.json")

    ap.add_argument("--latoken-days", type=int, default=14)
    ap.add_argument("--latoken-end-offset-days", type=int, default=0)
    ap.add_argument("--latoken-pair-offset", type=int, default=0)
    ap.add_argument("--latoken-max-pairs", type=int, default=120, help="0 = all pairs from offset")
    ap.add_argument("--latoken-workers", type=int, default=10)
    ap.add_argument("--latoken-chunk-hours", type=int, default=72)

    ap.add_argument("--uniswap-blocks", type=int, default=80000)
    ap.add_argument("--uniswap-end-block-offset", type=int, default=0)
    ap.add_argument("--uniswap-pool-lookback-blocks", type=int, default=0)
    ap.add_argument("--uniswap-max-pools", type=int, default=100, help="0 = all discovered quote pools")

    ap.add_argument("--min-bars", type=int, default=600)
    ap.add_argument("--skip-latoken", action="store_true")
    ap.add_argument("--skip-uniswap", action="store_true")
    ap.add_argument("--strict-both", action="store_true")
    args = ap.parse_args()

    panels: list[pd.DataFrame] = []
    reports: list[dict] = []

    if not args.skip_latoken:
        try:
            lat, rep = collect_latoken_panel(
                days=args.latoken_days,
                end_offset_days=args.latoken_end_offset_days,
                pair_offset=args.latoken_pair_offset,
                max_pairs=args.latoken_max_pairs,
                workers=args.latoken_workers,
                chunk_hours=args.latoken_chunk_hours,
            )
            reports.append(rep)
            if len(lat):
                panels.append(lat)
        except Exception as exc:
            reports.append({"source": "latoken", "error": repr(exc)})

    if not args.skip_uniswap:
        attempts: list[dict[str, str]] = []
        success_rep: dict | None = None
        success_panel: pd.DataFrame | None = None
        for idx, (rpc_label, rpc) in enumerate(_rpc_candidates()):
            try:
                uni, rep = collect_uniswap_v3_panel(
                    rpc_url=rpc,
                    lookback_blocks=args.uniswap_blocks,
                    end_block_offset=args.uniswap_end_block_offset,
                    pool_lookback_blocks=(args.uniswap_pool_lookback_blocks or None),
                    max_pools=args.uniswap_max_pools,
                    workdir=f"data/real_v2/cryo_rpc_{idx}",
                )
                if len(uni) and not rep.get("error"):
                    success_rep = dict(rep)
                    success_rep["rpc_label"] = rpc_label
                    success_panel = uni
                    attempts.append({"rpc_label": rpc_label, "status": "success"})
                    break
                attempts.append({
                    "rpc_label": rpc_label,
                    "status": "empty",
                    "error": str(rep.get("error") or "no market bars")[:800],
                })
            except Exception as exc:
                attempts.append({
                    "rpc_label": rpc_label,
                    "status": "error",
                    "error": repr(exc)[:800],
                })

        if success_rep is not None and success_panel is not None:
            success_rep["rpc_attempts"] = attempts
            reports.append(success_rep)
            panels.append(success_panel)
        else:
            reports.append({
                "source": "uniswap_cryo",
                "error": "all Ethereum RPC candidates failed or returned no bars",
                "rpc_attempts": attempts,
            })

    pre_report = {
        "sources": reports,
        "requested": {
            "latoken_days": args.latoken_days,
            "latoken_end_offset_days": args.latoken_end_offset_days,
            "latoken_pair_offset": args.latoken_pair_offset,
            "latoken_max_pairs": args.latoken_max_pairs,
            "uniswap_blocks": args.uniswap_blocks,
            "uniswap_end_block_offset": args.uniswap_end_block_offset,
            "uniswap_pool_lookback_blocks": args.uniswap_pool_lookback_blocks,
            "uniswap_max_pools": args.uniswap_max_pools,
            "min_bars": args.min_bars,
        },
        "panel_rows": 0,
        "panel_symbols": 0,
    }
    _save_report(args.report, pre_report)

    success_sources = {
        r.get("source") for r in reports
        if not r.get("error") and ((r.get("rows", 0) or r.get("bar_rows", 0)) > 0)
    }
    if args.strict_both and not {"latoken", "uniswap_cryo"}.issubset(success_sources):
        raise SystemExit(f"strict-both failed; successful={sorted(success_sources)}")

    try:
        panel = merge_and_regularize_panels(panels, min_bars=args.min_bars)
    except Exception as exc:
        pre_report["merge_error"] = repr(exc)
        _save_report(args.report, pre_report)
        raise

    write_table(panel, args.out)
    report = {
        **pre_report,
        "panel_rows": len(panel),
        "panel_symbols": int(panel["symbol"].nunique()),
        "timestamp_min": str(panel["timestamp"].min()),
        "timestamp_max": str(panel["timestamp"].max()),
        "latoken_symbols": int(panel[panel["source"].eq("latoken")]["symbol"].nunique()) if "source" in panel else 0,
        "uniswap_symbols": int(panel[panel["source"].eq("uniswap")]["symbol"].nunique()) if "source" in panel else 0,
    }
    _save_report(args.report, report)


if __name__ == "__main__":
    main()
