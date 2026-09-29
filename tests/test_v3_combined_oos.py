import pandas as pd

from crypto_pump.v3.combined_oos import (
    combine_oos_predictions,
    combine_oos_trades,
    combined_oos_report,
    pooled_ml_report,
)


def _passing_ledgers():
    latoken = pd.DataFrame({
        "timestamp": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"],
        "net_pnl_usd": [10.0, -2.0, 4.0],
        "net_return": [0.10, -0.02, 0.04],
        "executable": [True, True, True],
    })
    pumpfun = pd.DataFrame({
        "timestamp": ["2026-06-01T00:00:00Z", "2026-06-02T00:00:00Z", "2026-06-03T00:00:00Z"],
        "net_pnl_usd": [8.0, -1.0, 3.0],
        "net_return": [0.08, -0.01, 0.03],
        "executable": [True, True, True],
    })
    return latoken, pumpfun


def test_combined_oos_financials_are_recomputed_from_pooled_ledger():
    latoken = pd.DataFrame({
        "timestamp": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
        "net_pnl_usd": [10.0, -5.0],
        "net_return": [0.10, -0.05],
        "executable": [True, True],
    })
    pumpfun = pd.DataFrame({
        "timestamp": ["2026-06-01T00:00:00Z", "2026-06-02T00:00:00Z"],
        "net_pnl_usd": [3.0, 2.0],
        "net_return": [0.03, 0.02],
        "executable": [True, True],
    })
    combined = combine_oos_trades({"latoken_legacy": latoken, "pumpfun_33m": pumpfun})
    report = combined_oos_report(combined)
    assert report["report_name"] == "COMBINED_OOS"
    assert report["financial"]["trades"] == 4
    assert report["financial"]["net_pnl_usd"] == 10.0
    assert report["financial"]["expectancy_usd"] == 2.5
    assert report["financial"]["profit_factor"] == 3.0
    assert set(report["sources"]) == {"latoken_legacy", "pumpfun_33m"}
    assert report["verdict"]["historical_gate_passed"] is False
    assert report["verdict"]["canary_eligible"] is False


def test_combined_oos_requires_execution_stress_for_every_source():
    latoken, pumpfun = _passing_ledgers()
    combined = combine_oos_trades({"latoken_legacy": latoken, "pumpfun_33m": pumpfun})
    report = combined_oos_report(
        combined,
        evaluation_equity_usd=5000.0,
        execution_stress={"pumpfun_33m": {"costs_x2_net_pnl_usd": 5.0}},
    )
    assert report["verdict"]["historical_gate_passed"] is False
    assert report["verdict"]["execution_stress_complete"] is False
    assert any("execution_stress_incomplete:latoken_legacy" == x for x in report["verdict"]["shadow_reasons"])


def test_passing_historical_gate_recommends_shadow_but_never_canary():
    latoken, pumpfun = _passing_ledgers()
    combined = combine_oos_trades({"latoken_legacy": latoken, "pumpfun_33m": pumpfun})
    report = combined_oos_report(
        combined,
        evaluation_equity_usd=5000.0,
        execution_stress={
            "latoken_legacy": {"costs_x2_net_pnl_usd": 6.0},
            "pumpfun_33m": {"costs_x2_net_pnl_usd": 5.0},
        },
    )
    assert report["verdict"]["historical_gate_passed"] is True
    assert report["verdict"]["recommended_stage"] == "SHADOW"
    assert report["verdict"]["canary_eligible"] is False
    assert "requires_300_shadow_trades" in report["verdict"]["canary_reasons"]


def test_combined_predictions_refuse_to_pool_different_target_definitions():
    legacy = pd.DataFrame({"y_true": [1, 0], "probability": [0.9, 0.1]})
    pump = pd.DataFrame({"y_true": [1, 0], "probability": [0.8, 0.2]})
    combined = combine_oos_predictions(
        {"latoken_legacy": legacy, "pumpfun_33m": pump},
        target_definitions={"latoken_legacy": "legacy_future_pump", "pumpfun_33m": "tp_before_sl"},
    )
    report = pooled_ml_report(combined)
    assert report["comparable"] is False
    assert report["pr_auc"] is None


def test_pooled_ml_report_when_targets_match():
    a = pd.DataFrame({"y_true": [1, 0], "probability": [0.9, 0.1]})
    b = pd.DataFrame({"y_true": [1, 0], "probability": [0.8, 0.2]})
    combined = combine_oos_predictions({"a": a, "b": b}, target_definitions={"a": "tp_before_sl", "b": "tp_before_sl"})
    report = pooled_ml_report(combined, precision_ks=(2,))
    assert report["comparable"] is True
    assert report["pr_auc"] == 1.0
    assert report["precision_at_k"]["2"] == 1.0
