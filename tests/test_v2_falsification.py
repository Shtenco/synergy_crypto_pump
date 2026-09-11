import pandas as pd

from crypto_pump.v2.config import DEFAULT_HORIZONS
from crypto_pump.v2.falsification import run_falsification


def test_falsification_reports_placebo_decluster_and_time_slices():
    spec = DEFAULT_HORIZONS[0]
    times = pd.date_range("2026-01-01", periods=20, freq="5min", tz="UTC")
    rows = []
    for i, ts in enumerate(times):
        for j, symbol in enumerate(("A", "B", "C")):
            y = int(symbol == "A" and 8 <= i <= 10)
            score = 0.95 if y else (0.4 if symbol == "A" else 0.05 + 0.01 * j)
            rows.append({
                "timestamp": ts,
                "symbol": symbol,
                spec.target_col: y,
                f"p__{spec.name}": score,
                "ret_1": 0.01,
                "ret_3": 0.02,
                "current_candle_return": 0.005,
            })
    d = pd.DataFrame(rows)
    report = run_falsification(
        d,
        specs=[spec],
        top_k=1,
        pre_pump_threshold=0.05,
        placebo_shift_bars=2,
    )
    r = report["targets"][spec.name]
    assert r["raw"]["pr_auc"] > r["raw"]["positive_rate"]
    assert len(r["time_slices"]) == 4
    assert r["decluster"]["event_clusters"] == 1
    assert r["decluster"]["unique_event_clusters_hit"] == 1
    assert "within_symbol_random_permutation" in r["placebos"]
    assert r["pre_pump_filter"]["available"] is True


def test_pre_pump_filter_removes_already_moving_rows():
    spec = DEFAULT_HORIZONS[0]
    d = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC"),
        "symbol": ["A"] * 4,
        spec.target_col: [0, 1, 1, 0],
        f"p__{spec.name}": [0.1, 0.9, 0.8, 0.2],
        "ret_1": [0.0, 0.01, 0.10, 0.0],
    })
    report = run_falsification(d, specs=[spec], top_k=1, pre_pump_threshold=0.05, placebo_shift_bars=1)
    filt = report["targets"][spec.name]["pre_pump_filter"]
    assert filt["rows_after"] == 3
