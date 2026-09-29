#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from crypto_pump.v3.combined_oos import combine_oos_predictions, combine_oos_trades, combined_oos_report


def read_table(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_parquet(p) if p.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(p)


def read_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Pool LATOKEN legacy OOS and Pump.fun V3 OOS into one fail-closed COMBINED_OOS report")
    ap.add_argument("--latoken-trades", required=True)
    ap.add_argument("--pumpfun-trades", required=True)
    ap.add_argument("--latoken-predictions")
    ap.add_argument("--pumpfun-predictions")
    ap.add_argument("--latoken-metrics")
    ap.add_argument("--pumpfun-metrics")
    ap.add_argument("--latoken-execution-stress", help="JSON with at least costs_x2_net_pnl_usd")
    ap.add_argument("--pumpfun-execution-stress", help="JSON with at least costs_x2_net_pnl_usd")
    ap.add_argument("--evaluation-equity-usd", type=float, required=True, help="Equity denominator used by the max-DD promotion gate")
    ap.add_argument(
        "--pumpfun-horizon",
        default=None,
        help="Predeclared single Pump.fun trading horizon. Required when the OOS ledger contains multiple horizons; horizons are never pooled into one PnL.",
    )
    ap.add_argument("--latoken-target", default="legacy_future_pump")
    ap.add_argument("--pumpfun-target", default="tp_before_sl")
    ap.add_argument("--target-col", default="y_true")
    ap.add_argument("--score-col", default="probability")
    ap.add_argument("--out", default="artifacts/combined_oos")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    latoken_trades = read_table(args.latoken_trades)
    pumpfun_trades = read_table(args.pumpfun_trades)
    if "horizon" in pumpfun_trades.columns:
        horizons = sorted(str(x) for x in pumpfun_trades["horizon"].dropna().unique())
        if len(horizons) > 1 and not args.pumpfun_horizon:
            raise SystemExit(
                "Pump.fun OOS contains multiple horizons; refuse to pool them into one profitability verdict. "
                "Pass --pumpfun-horizon with a horizon predeclared before reading its PnL. "
                f"available={horizons}"
            )
        if args.pumpfun_horizon:
            if args.pumpfun_horizon not in horizons:
                raise SystemExit(f"Unknown --pumpfun-horizon={args.pumpfun_horizon}; available={horizons}")
            pumpfun_trades = pumpfun_trades[pumpfun_trades["horizon"].astype(str).eq(args.pumpfun_horizon)].copy()
    trade_sources = {"latoken_legacy": latoken_trades, "pumpfun_33m": pumpfun_trades}
    trades = combine_oos_trades(trade_sources)
    trades.to_parquet(out / "combined_oos_trades.parquet", index=False)

    prediction_sources = {"latoken_legacy": read_table(args.latoken_predictions), "pumpfun_33m": read_table(args.pumpfun_predictions)}
    predictions = combine_oos_predictions(prediction_sources, target_definitions={"latoken_legacy": args.latoken_target, "pumpfun_33m": args.pumpfun_target})
    if not predictions.empty:
        predictions.to_parquet(out / "combined_oos_predictions.parquet", index=False)

    source_metric_snapshots = {"latoken_legacy": read_json(args.latoken_metrics), "pumpfun_33m": read_json(args.pumpfun_metrics)}
    source_metric_snapshots = {k: v for k, v in source_metric_snapshots.items() if v}
    execution_stress = {"latoken_legacy": read_json(args.latoken_execution_stress), "pumpfun_33m": read_json(args.pumpfun_execution_stress)}
    execution_stress = {k: v for k, v in execution_stress.items() if v}

    report = combined_oos_report(
        trades,
        predictions=predictions if not predictions.empty else None,
        source_metric_snapshots=source_metric_snapshots,
        execution_stress=execution_stress,
        evaluation_equity_usd=args.evaluation_equity_usd,
        dataset_metadata={
            "latoken_legacy": {"dataset": "LATOKEN legacy LARGE OOS", "known_size": "~2.45M 5m rows in LARGE data run", "target": args.latoken_target},
            "pumpfun_33m": {"dataset": "Pump.fun launch-to-graduation corpus", "known_size": "33.58M raw trades", "target": args.pumpfun_target},
        },
        target_col=args.target_col,
        score_col=args.score_col,
    )
    (out / "combined_oos_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
