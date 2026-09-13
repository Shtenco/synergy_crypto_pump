from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromotionThresholds:
    min_shadow_trades: int = 300
    min_shadow_days: int = 30
    min_net_expectancy_usd: float = 0.0
    min_profit_factor: float = 1.05
    max_drawdown_fraction_equity: float = 0.05
    max_shadow_live_drift_bps: float = 75.0
    require_costs_x2_nonnegative: bool = True
    require_top1pct_robustness: bool = True


def can_promote_to_canary(
    *,
    shadow_trades: int,
    shadow_days: int,
    net_expectancy_usd: float,
    profit_factor: float | None,
    max_drawdown_usd: float,
    equity_usd: float,
    shadow_live_drift_bps: float,
    costs_x2_net_pnl_usd: float,
    remove_top1pct_net_pnl_usd: float,
    ci_green: bool,
    risk_violations: int,
    thresholds: PromotionThresholds | None = None,
) -> tuple[bool, list[str]]:
    t = thresholds or PromotionThresholds()
    reasons: list[str] = []
    if not ci_green:
        reasons.append("ci_not_green")
    if risk_violations:
        reasons.append("risk_violations")
    if shadow_trades < t.min_shadow_trades:
        reasons.append("insufficient_shadow_trades")
    if shadow_days < t.min_shadow_days:
        reasons.append("insufficient_shadow_days")
    if net_expectancy_usd <= t.min_net_expectancy_usd:
        reasons.append("nonpositive_expectancy")
    if profit_factor is None or profit_factor < t.min_profit_factor:
        reasons.append("profit_factor_below_min")
    if equity_usd <= 0 or abs(min(0.0, max_drawdown_usd)) / equity_usd > t.max_drawdown_fraction_equity:
        reasons.append("drawdown_above_max")
    if shadow_live_drift_bps > t.max_shadow_live_drift_bps:
        reasons.append("shadow_live_drift_above_max")
    if t.require_costs_x2_nonnegative and costs_x2_net_pnl_usd < 0:
        reasons.append("fails_costs_x2")
    if t.require_top1pct_robustness and remove_top1pct_net_pnl_usd < 0:
        reasons.append("depends_on_top1pct_winners")
    return not reasons, reasons
