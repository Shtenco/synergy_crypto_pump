from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RiskLimits:
    max_position_usd: float = 100.0
    max_position_fraction_equity: float = 0.005
    max_daily_loss_fraction: float = 0.02
    max_total_exposure_fraction: float = 0.10
    max_price_impact_bps: float = 150.0
    max_estimated_slippage_bps: float = 200.0
    min_liquidity_usd: float = 10_000.0
    max_consecutive_failures: int = 3


@dataclass(frozen=True, slots=True)
class RiskDecision:
    approved: bool
    notional_usd: float
    reason: str


class RiskManager:
    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()
        self.daily_realized_pnl_usd = 0.0
        self.open_exposure_usd = 0.0
        self.consecutive_failures = 0
        self.kill_switch = False

    def reset_day(self) -> None:
        self.daily_realized_pnl_usd = 0.0
        self.consecutive_failures = 0

    def record_fill_result(self, *, pnl_usd: float = 0.0, failed: bool = False) -> None:
        self.daily_realized_pnl_usd += float(pnl_usd)
        self.consecutive_failures = self.consecutive_failures + 1 if failed else 0
        if self.consecutive_failures >= self.limits.max_consecutive_failures:
            self.kill_switch = True

    def set_open_exposure(self, usd: float) -> None:
        self.open_exposure_usd = max(0.0, float(usd))

    def evaluate(
        self,
        *,
        equity_usd: float,
        requested_notional_usd: float,
        liquidity_usd: float,
        estimated_impact_bps: float,
        estimated_slippage_bps: float,
        data_fresh: bool = True,
        model_healthy: bool = True,
    ) -> RiskDecision:
        if self.kill_switch:
            return RiskDecision(False, 0.0, "kill_switch")
        if equity_usd <= 0:
            return RiskDecision(False, 0.0, "invalid_equity")
        if not data_fresh:
            return RiskDecision(False, 0.0, "stale_data")
        if not model_healthy:
            return RiskDecision(False, 0.0, "model_unhealthy")
        if self.daily_realized_pnl_usd <= -equity_usd * self.limits.max_daily_loss_fraction:
            self.kill_switch = True
            return RiskDecision(False, 0.0, "daily_loss_limit")
        if liquidity_usd < self.limits.min_liquidity_usd:
            return RiskDecision(False, 0.0, "liquidity_below_min")
        if estimated_impact_bps > self.limits.max_price_impact_bps:
            return RiskDecision(False, 0.0, "impact_above_max")
        if estimated_slippage_bps > self.limits.max_estimated_slippage_bps:
            return RiskDecision(False, 0.0, "slippage_above_max")
        if self.open_exposure_usd >= equity_usd * self.limits.max_total_exposure_fraction:
            return RiskDecision(False, 0.0, "portfolio_exposure_limit")

        cap = min(
            self.limits.max_position_usd,
            equity_usd * self.limits.max_position_fraction_equity,
            max(0.0, equity_usd * self.limits.max_total_exposure_fraction - self.open_exposure_usd),
        )
        approved = min(float(requested_notional_usd), cap)
        if approved <= 0:
            return RiskDecision(False, 0.0, "zero_sized")
        return RiskDecision(True, approved, "approved")
