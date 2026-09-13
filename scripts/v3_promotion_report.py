from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

from crypto_pump.v3.metrics import financial_report, stress_test_trades
from crypto_pump.v3.promotion import can_promote_to_canary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate whether V3 is allowed to move from shadow to tiny live canary")
    p.add_argument("--store", default="data/v3/live.sqlite")
    p.add_argument("--oos-report", type=Path, required=True)
    p.add_argument("--horizon", default="tp20_sl08_1h")
    p.add_argument("--equity-usd", type=float, default=10_000.0)
    p.add_argument("--ci-green", action="store_true")
    p.add_argument("--risk-violations", type=int, default=0)
    return p.parse_args()


def load_closed_positions(path: str) -> pd.DataFrame:
    with sqlite3.connect(path) as con:
        df = pd.read_sql_query(
            "SELECT * FROM shadow_positions WHERE status='CLOSED' ORDER BY closed_at ASC", con
        )
    if not df.empty:
        df["executable"] = True
    return df


def costs_x2_from_oos(report: dict, horizon: str) -> float:
    total = 0.0
    found = False
    for fold in report.get("folds", []):
        spec = fold.get("specs", {}).get(horizon, {})
        metric = (
            spec.get("execution_stress", {})
            .get("costs_x2", {})
            .get("metrics", {})
            .get("net_pnl_usd")
        )
        if metric is not None:
            total += float(metric)
            found = True
    return total if found else float("-inf")


def main() -> None:
    args = parse_args()
    closed = load_closed_positions(args.store)
    metrics = financial_report(closed)
    stress = stress_test_trades(closed)
    if closed.empty:
        shadow_days = 0
    else:
        opened = pd.to_datetime(closed["opened_at"], utc=True)
        closed_at = pd.to_datetime(closed["closed_at"], utc=True)
        shadow_days = max(1, int((closed_at.max().normalize() - opened.min().normalize()).days) + 1)

    oos = json.loads(args.oos_report.read_text(encoding="utf-8"))
    costs_x2 = costs_x2_from_oos(oos, args.horizon)
    top1 = stress.get("remove_top_1pct", {}).get("net_pnl_usd", 0.0)
    allowed, reasons = can_promote_to_canary(
        shadow_trades=int(metrics.get("trades", 0)),
        shadow_days=shadow_days,
        net_expectancy_usd=float(metrics.get("expectancy_usd", 0.0) or 0.0),
        profit_factor=metrics.get("profit_factor"),
        max_drawdown_usd=float(metrics.get("max_drawdown_usd", 0.0) or 0.0),
        equity_usd=args.equity_usd,
        costs_x2_net_pnl_usd=costs_x2,
        remove_top1pct_net_pnl_usd=float(top1 or 0.0),
        ci_green=args.ci_green,
        risk_violations=args.risk_violations,
    )
    result = {
        "promote_to_canary": allowed,
        "blocking_reasons": reasons,
        "shadow_days": shadow_days,
        "shadow_metrics": metrics,
        "shadow_winner_stress": stress,
        "oos_costs_x2_net_pnl_usd": costs_x2,
        "ci_green": args.ci_green,
    }
    print(json.dumps(result, indent=2, default=str))
    raise SystemExit(0 if allowed else 2)


if __name__ == "__main__":
    main()
