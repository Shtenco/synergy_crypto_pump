from __future__ import annotations

import pandas as pd

from crypto_pump.v3.live import SOL_MINT, ShadowSignalEngine, ShadowStrategyConfig
from crypto_pump.v3.monitor import HealthThresholds, health_report, population_stability_index
from crypto_pump.v3.panel import normalize_pumpportal_trades, trades_to_panel
from crypto_pump.v3.store import V3Store


class FakeQuoteClient:
    def quote(self, *, input_mint, output_mint, amount_atomic, slippage_bps=100):
        if input_mint == SOL_MINT:
            return {
                "outAmount": "1000000",
                "priceImpactPct": "0.001",
                "routePlan": [{"percent": 100}],
            }
        return {
            "outAmount": str(int(1.25 * 50 / 100 * 1_000_000_000)),
            "priceImpactPct": "0.001",
            "routePlan": [{"percent": 100}],
        }


def test_store_is_idempotent_and_builds_panel(tmp_path):
    store = V3Store(tmp_path / "live.sqlite")
    base = pd.Timestamp("2026-09-13T10:00:00Z")
    event1 = {
        "txType": "buy",
        "mint": "MINT1",
        "signature": "sig1",
        "tokenAmount": 1000,
        "solAmount": 1.0,
        "marketCapSol": 50.0,
        "vSolInBondingCurve": 25.0,
        "vTokensInBondingCurve": 25000.0,
        "traderPublicKey": "wallet-a",
    }
    event2 = {
        "txType": "sell",
        "mint": "MINT1",
        "signature": "sig2",
        "tokenAmount": 500,
        "solAmount": 0.6,
        "marketCapSol": 60.0,
        "vSolInBondingCurve": 26.0,
        "vTokensInBondingCurve": 24500.0,
        "traderPublicKey": "wallet-b",
    }
    assert store.append_event(event1, received_at=base)
    assert not store.append_event(event1, received_at=base)
    assert store.append_event(event2, received_at=base + pd.Timedelta(minutes=1))

    trades = normalize_pumpportal_trades(store.events())
    assert len(trades) == 2
    panel = trades_to_panel(trades, sol_usd=150.0)
    assert len(panel) == 1
    assert panel.loc[0, "trades"] == 2
    assert panel.loc[0, "buy_trade_count"] == 1
    assert panel.loc[0, "sell_trade_count"] == 1
    assert panel.loc[0, "liquidity_usd"] == 26.0 * 150.0 * 2.0


def test_health_gate_detects_schema_staleness_and_drift():
    ref = pd.DataFrame({
        "timestamp": pd.date_range("2026-09-13T00:00:00Z", periods=100, freq="5min"),
        "x": list(range(100)),
    })
    live = pd.DataFrame({
        "timestamp": pd.date_range("2026-09-13T08:00:00Z", periods=60, freq="5min"),
        "x": list(range(1000, 1060)),
    })
    psi = population_stability_index(ref["x"], live["x"])
    assert psi > 0.3
    report = health_report(
        live,
        expected_features=["x", "missing_feature"],
        reference_frame=ref,
        now=pd.Timestamp("2026-09-13T12:00:00Z"),
        thresholds=HealthThresholds(max_data_age_seconds=60, max_feature_psi=0.3),
    )
    assert not report.healthy
    assert "feature_parity" in report.reasons
    assert "stale_data" in report.reasons
    assert "feature_drift" in report.reasons


def test_ranking_opens_and_marks_shadow_position(tmp_path):
    store = V3Store(tmp_path / "shadow.sqlite")
    engine = ShadowSignalEngine(
        store=store,
        quote_client=FakeQuoteClient(),
        config=ShadowStrategyConfig(
            probability_column="pump_probability__20pct_1h",
            min_probability=0.70,
            top_k=1,
            requested_notional_usd=50.0,
            tp_return=0.10,
            sl_return=-0.08,
            max_hold_minutes=60,
        ),
    )
    ranking = pd.DataFrame({
        "timestamp": [pd.Timestamp.now(tz="UTC")],
        "symbol": ["MINT1"],
        "liquidity_usd": [100_000.0],
        "pump_probability__20pct_1h": [0.91],
    })
    opened = engine.submit_ranking(ranking, equity_usd=10_000.0, sol_usd=100.0)
    assert len(opened) == 1
    assert len(store.open_positions()) == 1

    actions = engine.mark_open_positions(sol_usd=100.0)
    assert any(a["action"] == "CLOSE" and a["reason"] == "take_profit" for a in actions)
    assert store.open_positions().empty
