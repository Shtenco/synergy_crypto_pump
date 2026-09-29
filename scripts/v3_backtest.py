#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from crypto_pump.v3.execution import ExecutionBacktester, ExecutionCostModel
from crypto_pump.v3.metrics import capacity_curve, financial_report, rerun_cost_stress, stress_test_trades


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(p)


def main() -> None:
    ap = argparse.ArgumentParser(description="Execution-aware V3 backtest")
    ap.add_argument("source")
    ap.add_argument("--out", default="artifacts/v3")
    ap.add_argument("--signal-col", default="signal")
    ap.add_argument("--notional-col", default="notional_usd")
    ap.add_argument("--liquidity-col", default="liquidity_usd")
    ap.add_argument("--exit-bar-col", default="exit_bar")
    args = ap.parse_args()

    d = read_table(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    costs = ExecutionCostModel()
    trades = ExecutionBacktester(costs).backtest(
        d,
        signal_col=args.signal_col,
        notional_col=args.notional_col,
        liquidity_col=args.liquidity_col,
        exit_bar_col=args.exit_bar_col,
    )
    trades.to_parquet(out / "trades.parquet", index=False)

    report = {
        "financial": financial_report(trades),
        "winner_robustness": stress_test_trades(trades),
        "cost_latency_stress": rerun_cost_stress(
            d,
            base_costs=costs,
            signal_col=args.signal_col,
            notional_col=args.notional_col,
            liquidity_col=args.liquidity_col,
            exit_bar_col=args.exit_bar_col,
        ),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    cap = capacity_curve(
        d,
        [20, 50, 100, 250, 500, 1000, 2000, 5000],
        costs=costs,
        signal_col=args.signal_col,
        liquidity_col=args.liquidity_col,
        exit_bar_col=args.exit_bar_col,
    )
    cap.to_csv(out / "capacity_curve.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
