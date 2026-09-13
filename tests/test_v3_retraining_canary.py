from __future__ import annotations

import json

import pandas as pd
import pytest

from crypto_pump.v3.canary import CanaryIntentService, CanaryLimits
from crypto_pump.v3.retraining import RetrainingPolicy, decide_retraining


def test_retraining_requires_history_and_breadth():
    ts = pd.date_range("2026-01-01", periods=100, freq="5min", tz="UTC")
    frame = pd.DataFrame({"timestamp": ts, "symbol": ["A"] * len(ts)})
    d = decide_retraining(
        frame,
        policy=RetrainingPolicy(min_new_bars=1, min_history_days=30, min_symbols=1000),
    )
    assert not d.should_retrain
    assert "insufficient_history_days" in d.reasons
    assert "insufficient_symbol_breadth" in d.reasons


def test_retraining_triggers_after_enough_new_data():
    ts = pd.date_range("2026-01-01", periods=40, freq="1D", tz="UTC")
    frame = pd.DataFrame({
        "timestamp": list(ts) * 2,
        "symbol": ["A"] * len(ts) + ["B"] * len(ts),
    })
    d = decide_retraining(
        frame,
        previous_manifest={"rows": 10, "trained_at": "2026-01-01T00:00:00Z"},
        now=pd.Timestamp("2026-02-10", tz="UTC"),
        policy=RetrainingPolicy(min_new_bars=20, max_model_age_days=7, min_history_days=30, min_symbols=2),
    )
    assert d.should_retrain
    assert "enough_new_bars" in d.reasons


def test_canary_cannot_create_intent_before_promotion(tmp_path):
    svc = CanaryIntentService(tmp_path / "queue.jsonl", secret="x" * 32)
    with pytest.raises(PermissionError, match="shadow_to_canary_gate_not_passed"):
        svc.create_intent(
            mint="mint",
            notional_usd=10,
            max_slippage_bps=100,
            promotion_allowed=False,
            current_open_positions=0,
            realized_today_usd=0,
        )


def test_canary_hard_caps_and_hmac(tmp_path):
    svc = CanaryIntentService(
        tmp_path / "queue.jsonl",
        limits=CanaryLimits(max_trade_usd=20, max_daily_notional_usd=25, max_open_positions=2),
        secret="y" * 32,
    )
    intent = svc.create_intent(
        mint="mint",
        notional_usd=200,
        max_slippage_bps=100,
        promotion_allowed=True,
        current_open_positions=0,
        realized_today_usd=0,
    )
    assert intent.notional_usd == 20
    row = json.loads((tmp_path / "queue.jsonl").read_text().strip())
    assert svc.verify(row)
    with pytest.raises(PermissionError, match="canary_daily_notional_limit"):
        svc.create_intent(
            mint="mint2",
            notional_usd=10,
            max_slippage_bps=100,
            promotion_allowed=True,
            current_open_positions=0,
            realized_today_usd=0,
        )
