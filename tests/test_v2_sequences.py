import numpy as np
import pandas as pd

from crypto_pump.v2.temporal import build_causal_sequences


def test_sequences_end_at_current_row_only():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=6, freq="5min", tz="UTC"),
        "symbol": ["X"] * 6,
        "f": np.arange(6, dtype=float),
    })
    x, rows = build_causal_sequences(df, ["f"], seq_len=3)
    assert rows.tolist() == [2, 3, 4, 5]
    assert x[0, :, 0].tolist() == [0.0, 1.0, 2.0]
    assert x[-1, :, 0].tolist() == [3.0, 4.0, 5.0]
