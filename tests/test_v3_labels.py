import pandas as pd

from crypto_pump.v3.labels import TripleBarrierSpec, add_triple_barrier_labels


def test_tp_must_be_hit_before_sl():
    spec = TripleBarrierSpec("x", 3, 0.20, 0.10)
    d = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=6, freq="5min", tz="UTC"),
        "symbol": ["A"] * 6,
        "close": [100, 100, 89, 121, 120, 120],
        "high":  [100, 101, 90, 122, 121, 121],
        "low":   [100,  99, 88, 118, 119, 119],
    })
    out = add_triple_barrier_labels(d, specs=[spec], entry_delay_bars=1)
    assert out.loc[0, spec.target_col] == 0
    assert out.loc[0, spec.reason_col] == "stop"


def test_same_bar_tp_and_sl_is_conservative_stop():
    spec = TripleBarrierSpec("x", 2, 0.10, 0.05)
    d = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC"),
        "symbol": ["A"] * 5,
        "close": [100, 100, 100, 100, 100],
        "high":  [100, 100, 111, 100, 100],
        "low":   [100, 100,  94, 100, 100],
    })
    out = add_triple_barrier_labels(d, specs=[spec], entry_delay_bars=1)
    assert out.loc[0, spec.target_col] == 0
    assert out.loc[0, spec.reason_col] == "both_stop_first"
