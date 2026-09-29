from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class PortfolioPolicy:
    """Deterministic capital/exposure limits for historical OOS execution."""

    initial_equity_usd: float = 5_000.0
    max_total_exposure_fraction: float = 0.10
    max_position_fraction_equity: float = 0.02
    max_concurrent_positions: int = 10
    max_daily_loss_fraction: float = 0.02
    one_open_position_per_symbol: bool = True


def apply_portfolio_limits(
    trades: pd.DataFrame,
    policy: PortfolioPolicy | None = None,
) -> pd.DataFrame:
    """Apply live-like portfolio constraints to an already priced candidate ledger.

    Candidates are considered in executable entry-time order. Capital is released
    only when a position exits. At equal entry times, lower rank / higher
    probability receives priority when those columns are available.
    """
    if trades.empty:
        return trades.copy()

    p = policy or PortfolioPolicy()
    if p.initial_equity_usd <= 0:
        raise ValueError("initial_equity_usd must be > 0")
    required = {"entry_timestamp", "exit_timestamp", "symbol", "notional_usd"}
    missing = required.difference(trades.columns)
    if missing:
        raise ValueError(f"portfolio ledger missing columns: {sorted(missing)}")

    out = trades.copy()
    out["entry_timestamp"] = pd.to_datetime(out["entry_timestamp"], utc=True, errors="coerce")
    out["exit_timestamp"] = pd.to_datetime(out["exit_timestamp"], utc=True, errors="coerce")
    out["_original_order"] = range(len(out))

    sort_cols = ["entry_timestamp"]
    ascending = [True]
    if "rank" in out.columns:
        sort_cols.append("rank")
        ascending.append(True)
    if "probability" in out.columns:
        sort_cols.append("probability")
        ascending.append(False)
    sort_cols.append("_original_order")
    ascending.append(True)
    out = out.sort_values(sort_cols, ascending=ascending, kind="stable").reset_index(drop=True)

    out["portfolio_approved"] = False
    out["portfolio_reject_reason"] = pd.Series([None] * len(out), dtype="object")

    equity = float(p.initial_equity_usd)
    open_positions: list[dict] = []
    daily_realized: dict[object, float] = {}

    def release_until(ts: pd.Timestamp) -> None:
        nonlocal equity, open_positions
        still_open: list[dict] = []
        for pos in open_positions:
            if pos["exit_timestamp"] <= ts:
                pnl = float(pos["net_pnl_usd"])
                equity += pnl
                day = pos["exit_timestamp"].date()
                daily_realized[day] = daily_realized.get(day, 0.0) + pnl
            else:
                still_open.append(pos)
        open_positions = still_open

    for idx, row in out.iterrows():
        base_executable = bool(row.get("executable", True))
        if not base_executable:
            out.at[idx, "portfolio_reject_reason"] = row.get("reject_reason") or "execution_rejected"
            continue

        entry_ts = row["entry_timestamp"]
        exit_ts = row["exit_timestamp"]
        if pd.isna(entry_ts) or pd.isna(exit_ts) or exit_ts <= entry_ts:
            out.at[idx, "executable"] = False
            out.at[idx, "reject_reason"] = "invalid_trade_timestamps"
            out.at[idx, "portfolio_reject_reason"] = "invalid_trade_timestamps"
            continue

        release_until(entry_ts)
        notional = float(row["notional_usd"])
        exposure = sum(float(x["notional_usd"]) for x in open_positions)
        symbol = str(row["symbol"])
        current_day_loss = daily_realized.get(entry_ts.date(), 0.0)

        reason: str | None = None
        if current_day_loss <= -max(equity, 0.0) * p.max_daily_loss_fraction:
            reason = "portfolio_daily_loss_limit"
        elif p.one_open_position_per_symbol and any(str(x["symbol"]) == symbol for x in open_positions):
            reason = "portfolio_symbol_already_open"
        elif len(open_positions) >= p.max_concurrent_positions:
            reason = "portfolio_max_concurrent_positions"
        elif notional > max(equity, 0.0) * p.max_position_fraction_equity:
            reason = "portfolio_position_fraction"
        elif exposure + notional > max(equity, 0.0) * p.max_total_exposure_fraction:
            reason = "portfolio_total_exposure"

        if reason is not None:
            out.at[idx, "executable"] = False
            out.at[idx, "reject_reason"] = reason
            out.at[idx, "portfolio_reject_reason"] = reason
            continue

        out.at[idx, "portfolio_approved"] = True
        open_positions.append({
            "exit_timestamp": exit_ts,
            "symbol": symbol,
            "notional_usd": notional,
            "net_pnl_usd": float(row.get("net_pnl_usd", 0.0) or 0.0),
        })

    out = out.sort_values("_original_order", kind="stable").drop(columns=["_original_order"]).reset_index(drop=True)
    return out
