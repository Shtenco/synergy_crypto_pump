from pathlib import Path

from crypto_pump.v3.execution import ExecutionBacktester, ExecutionCostModel
from crypto_pump.v3.risk import RiskLimits, RiskManager
from crypto_pump.v3.shadow import ShadowExecutor


def test_execution_costs_reduce_return_and_impact_rejects():
    costs = ExecutionCostModel(
        dex_fee_bps=30,
        base_slippage_bps=20,
        mev_haircut_bps=10,
        gas_usd=0,
        impact_coefficient=0.5,
        max_impact_bps=200,
        min_liquidity_usd=1000,
    )
    bt = ExecutionBacktester(costs)
    good = bt.estimate_trade(100, 110, 100, 100_000)
    assert good.executable
    assert 0 < good.net_return < 0.10

    bad = bt.estimate_trade(100, 110, 10_000, 10_000)
    assert not bad.executable
    assert bad.reason == "impact_above_max"


def test_risk_daily_loss_and_shadow_never_broadcasts(tmp_path: Path):
    risk = RiskManager(RiskLimits(
        max_position_usd=50,
        max_position_fraction_equity=0.01,
        max_daily_loss_fraction=0.02,
        max_total_exposure_fraction=0.10,
        min_liquidity_usd=1000,
    ))
    journal = tmp_path / "shadow.jsonl"
    shadow = ShadowExecutor(journal, risk=risk)
    order = shadow.submit(
        timestamp="2026-01-01T00:00:00Z",
        symbol="A",
        reference_price=1.0,
        requested_notional_usd=100,
        equity_usd=10_000,
        liquidity_usd=100_000,
        estimated_impact_bps=5,
        estimated_slippage_bps=10,
        probability=0.8,
    )
    assert order.approved_notional_usd == 50
    assert journal.exists()

    risk.record_fill_result(pnl_usd=-250)
    decision = risk.evaluate(
        equity_usd=10_000,
        requested_notional_usd=10,
        liquidity_usd=100_000,
        estimated_impact_bps=5,
        estimated_slippage_bps=10,
    )
    assert not decision.approved
    assert decision.reason == "daily_loss_limit"
