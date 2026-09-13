from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ExecutionCostModel:
    dex_fee_bps: float = 30.0
    base_slippage_bps: float = 35.0
    gas_usd: float = 0.05
    mev_haircut_bps: float = 15.0
    failure_rate: float = 0.02
    failed_tx_cost_usd: float = 0.01
    latency_bars: int = 1
    impact_coefficient: float = 0.50
    min_liquidity_usd: float = 5_000.0
    max_impact_bps: float = 300.0

    def scaled(self, cost_mult: float = 1.0, latency_mult: float = 1.0) -> "ExecutionCostModel":
        if cost_mult <= 0 or latency_mult <= 0:
            raise ValueError("stress multipliers must be > 0")
        return replace(
            self,
            dex_fee_bps=self.dex_fee_bps * cost_mult,
            base_slippage_bps=self.base_slippage_bps * cost_mult,
            gas_usd=self.gas_usd * cost_mult,
            mev_haircut_bps=self.mev_haircut_bps * cost_mult,
            failed_tx_cost_usd=self.failed_tx_cost_usd * cost_mult,
            latency_bars=max(0, int(round(self.latency_bars * latency_mult))),
        )


@dataclass(frozen=True, slots=True)
class FillEstimate:
    executable: bool
    reason: str
    impact_bps: float
    entry_price: float | None
    exit_price: float | None
    net_return: float | None
    net_pnl_usd: float


class ExecutionBacktester:
    """Long-only event backtester with explicit transaction frictions.

    `failure_rate` is charged as expected cost rather than randomly dropping
    trades, making OOS comparisons deterministic and reproducible.
    """

    def __init__(self, costs: ExecutionCostModel | None = None):
        self.costs = costs or ExecutionCostModel()

    def _impact_bps(self, notional_usd: float, liquidity_usd: float) -> float:
        if liquidity_usd <= 0:
            return float("inf")
        return 10_000.0 * self.costs.impact_coefficient * notional_usd / liquidity_usd

    def estimate_trade(
        self,
        entry_mid: float,
        exit_mid: float,
        notional_usd: float,
        liquidity_usd: float,
    ) -> FillEstimate:
        if notional_usd <= 0:
            return FillEstimate(False, "non_positive_notional", 0.0, None, None, None, 0.0)
        if not np.isfinite(liquidity_usd) or liquidity_usd < self.costs.min_liquidity_usd:
            return FillEstimate(False, "liquidity_below_min", float("inf"), None, None, None, 0.0)
        if entry_mid <= 0 or exit_mid <= 0:
            return FillEstimate(False, "invalid_price", 0.0, None, None, None, 0.0)

        impact_bps = self._impact_bps(notional_usd, liquidity_usd)
        if impact_bps > self.costs.max_impact_bps:
            return FillEstimate(False, "impact_above_max", impact_bps, None, None, None, 0.0)

        entry_bps = self.costs.base_slippage_bps + impact_bps + self.costs.mev_haircut_bps
        exit_bps = self.costs.base_slippage_bps + impact_bps + self.costs.mev_haircut_bps
        entry_price = entry_mid * (1.0 + entry_bps / 10_000.0)
        exit_price = exit_mid * (1.0 - exit_bps / 10_000.0)

        gross_return = exit_price / entry_price - 1.0
        two_way_fee = 2.0 * self.costs.dex_fee_bps / 10_000.0
        gas_return = 2.0 * self.costs.gas_usd / notional_usd
        expected_failure_return = self.costs.failure_rate * self.costs.failed_tx_cost_usd / notional_usd
        net_return = gross_return - two_way_fee - gas_return - expected_failure_return
        net_pnl = notional_usd * net_return
        return FillEstimate(True, "ok", impact_bps, entry_price, exit_price, net_return, net_pnl)

    def backtest(
        self,
        frame: pd.DataFrame,
        *,
        signal_col: str = "signal",
        notional_col: str = "notional_usd",
        liquidity_col: str = "liquidity_usd",
        exit_bar_col: str = "exit_bar",
    ) -> pd.DataFrame:
        required = {"timestamp", "symbol", "close", signal_col, notional_col, exit_bar_col}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")

        d = frame.copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        d = d.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
        rows: list[dict] = []

        for _, g in d.groupby("symbol", sort=False):
            g = g.reset_index(drop=True)
            for i, r in g.iterrows():
                if not bool(r[signal_col]):
                    continue
                entry_i = i + self.costs.latency_bars
                if pd.isna(r[exit_bar_col]):
                    continue
                hold = int(r[exit_bar_col])
                exit_i = entry_i + max(1, hold)
                if entry_i >= len(g) or exit_i >= len(g):
                    continue
                liq = float(r[liquidity_col]) if liquidity_col in g.columns and pd.notna(r.get(liquidity_col)) else float("nan")
                est = self.estimate_trade(
                    float(g.loc[entry_i, "close"]),
                    float(g.loc[exit_i, "close"]),
                    float(r[notional_col]),
                    liq,
                )
                rows.append({
                    "signal_timestamp": r["timestamp"],
                    "entry_timestamp": g.loc[entry_i, "timestamp"],
                    "exit_timestamp": g.loc[exit_i, "timestamp"],
                    "symbol": r["symbol"],
                    "notional_usd": float(r[notional_col]),
                    "liquidity_usd": liq,
                    "executable": est.executable,
                    "reject_reason": None if est.executable else est.reason,
                    "impact_bps": est.impact_bps,
                    "entry_price": est.entry_price,
                    "exit_price": est.exit_price,
                    "net_return": est.net_return,
                    "net_pnl_usd": est.net_pnl_usd,
                })

        return pd.DataFrame(rows)
