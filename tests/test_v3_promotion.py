from crypto_pump.v3.promotion import can_promote_to_canary, can_scale_live


def test_canary_gate_does_not_require_live_drift_before_live_exists():
    allowed, reasons = can_promote_to_canary(
        shadow_trades=350,
        shadow_days=35,
        net_expectancy_usd=0.50,
        profit_factor=1.30,
        max_drawdown_usd=-100.0,
        equity_usd=10_000.0,
        costs_x2_net_pnl_usd=10.0,
        remove_top1pct_net_pnl_usd=5.0,
        ci_green=True,
        risk_violations=0,
    )
    assert allowed
    assert reasons == []


def test_canary_gate_blocks_unproven_shadow_edge():
    allowed, reasons = can_promote_to_canary(
        shadow_trades=50,
        shadow_days=4,
        net_expectancy_usd=-0.10,
        profit_factor=0.90,
        max_drawdown_usd=-1000.0,
        equity_usd=10_000.0,
        costs_x2_net_pnl_usd=-20.0,
        remove_top1pct_net_pnl_usd=-10.0,
        ci_green=False,
        risk_violations=1,
    )
    assert not allowed
    assert "ci_not_green" in reasons
    assert "insufficient_shadow_trades" in reasons
    assert "fails_costs_x2" in reasons


def test_scale_gate_requires_canary_parity_and_positive_live_ev():
    allowed, reasons = can_scale_live(
        canary_trades=40,
        canary_days=10,
        live_net_expectancy_usd=0.25,
        shadow_live_drift_bps=25.0,
        risk_violations=0,
        kill_switch_events=0,
    )
    assert allowed
    assert reasons == []
