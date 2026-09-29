from __future__ import annotations

import math

import pandas as pd

from .execution import ExecutionBacktester, ExecutionCostModel


def financial_report(trades: pd.DataFrame) -> dict[str, float | int | None]:
    if trades.empty:
        return {"trades": 0}
    x = trades[trades.get("executable", True)].copy()
    if x.empty or "net_pnl_usd" not in x:
        return {"trades": 0}

    # PnL path must be chronological. Without this, pooled folds/horizons can
    # manufacture an incorrect drawdown simply from dataframe append order.
    time_col = next(
        (c for c in ("exit_timestamp", "entry_timestamp", "signal_timestamp", "timestamp") if c in x.columns),
        None,
    )
    if time_col is not None:
        x[time_col] = pd.to_datetime(x[time_col], utc=True, errors="coerce")
        x = x.sort_values(time_col, kind="stable", na_position="last").reset_index(drop=True)

    pnl = pd.to_numeric(x["net_pnl_usd"], errors="coerce").fillna(0.0)
    ret = pd.to_numeric(x.get("net_return", 0.0), errors="coerce").fillna(0.0)
    equity = pnl.cumsum()
    # Include the pre-trade equity/PnL baseline at 0. A first losing trade must
    # create drawdown instead of becoming its own (negative) high-water mark.
    peak = equity.cummax().clip(lower=0.0)
    drawdown = equity - peak
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    std = float(ret.std(ddof=1)) if len(ret) > 1 else 0.0

    return {
        "trades": int(len(x)),
        "net_pnl_usd": float(pnl.sum()),
        "expectancy_usd": float(pnl.mean()),
        "median_pnl_usd": float(pnl.median()),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else None,
        "max_drawdown_usd": float(drawdown.min()) if len(drawdown) else 0.0,
        "mean_net_return": float(ret.mean()),
        "sharpe_per_trade": float(ret.mean() / std * math.sqrt(len(ret))) if std > 0 else None,
    }


def _trim_best(trades: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0 or trades.empty:
        return trades.copy()
    return trades.sort_values("net_pnl_usd", ascending=False).iloc[n:].copy()


def stress_test_trades(trades: pd.DataFrame) -> dict[str, dict]:
    if trades.empty:
        return {"base": financial_report(trades)}
    executable = trades[trades.get("executable", True)].copy()
    n_top1pct = max(1, int(math.ceil(len(executable) * 0.01))) if len(executable) else 0
    return {
        "base": financial_report(executable),
        "remove_best_1": financial_report(_trim_best(executable, 1)),
        "remove_best_3": financial_report(_trim_best(executable, 3)),
        "remove_top_1pct": financial_report(_trim_best(executable, n_top1pct)),
    }


def capacity_curve(
    market_frame: pd.DataFrame,
    notionals: list[float],
    *,
    costs: ExecutionCostModel | None = None,
    signal_col: str = "signal",
    liquidity_col: str = "liquidity_usd",
    exit_bar_col: str = "exit_bar",
) -> pd.DataFrame:
    rows: list[dict] = []
    for notional in notionals:
        d = market_frame.copy()
        d["notional_usd"] = float(notional)
        bt = ExecutionBacktester(costs).backtest(
            d,
            signal_col=signal_col,
            notional_col="notional_usd",
            liquidity_col=liquidity_col,
            exit_bar_col=exit_bar_col,
        )
        rows.append({"notional_usd": float(notional), **financial_report(bt)})
    return pd.DataFrame(rows)


def rerun_cost_stress(
    market_frame: pd.DataFrame,
    *,
    base_costs: ExecutionCostModel | None = None,
    scenarios: tuple[tuple[str, float, float], ...] = (
        ("base", 1.0, 1.0),
        ("costs_x2", 2.0, 1.0),
        ("latency_x2", 1.0, 2.0),
        ("costs_x2_latency_x2", 2.0, 2.0),
    ),
    signal_col: str = "signal",
    notional_col: str = "notional_usd",
    liquidity_col: str = "liquidity_usd",
    exit_bar_col: str = "exit_bar",
) -> dict[str, dict]:
    c = base_costs or ExecutionCostModel()
    out: dict[str, dict] = {}
    for name, cm, lm in scenarios:
        stressed = c.scaled(cost_mult=cm, latency_mult=lm)
        trades = ExecutionBacktester(stressed).backtest(
            market_frame,
            signal_col=signal_col,
            notional_col=notional_col,
            liquidity_col=liquidity_col,
            exit_bar_col=exit_bar_col,
        )
        out[name] = {
            "cost_model": {
                "dex_fee_bps": stressed.dex_fee_bps,
                "base_slippage_bps": stressed.base_slippage_bps,
                "mev_haircut_bps": stressed.mev_haircut_bps,
                "latency_bars": stressed.latency_bars,
            },
            "metrics": financial_report(trades),
        }
    return out
