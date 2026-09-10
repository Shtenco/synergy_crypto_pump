import numpy as np
import pandas as pd

from crypto_pump.v2.config import HorizonSpec
from crypto_pump.v2.labels import add_multi_horizon_labels


def test_multihorizon_labels_are_strict_and_right_censored():
    n = 12
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"),
        "symbol": ["X"] * n,
        "open": np.ones(n),
        "high": np.array([1, 1, 1.25, 1, 1, 1, 1, 1, 1, 1, 1, 1], float),
        "low": np.ones(n) * 0.95,
        "close": np.ones(n),
        "volume": np.ones(n),
    })
    spec = HorizonSpec("x", 3, 0.20)
    out = add_multi_horizon_labels(df, [spec])
    assert int(out.loc[0, spec.target_col]) == 1
    assert out.loc[0, spec.hit_col] == 2
    assert pd.isna(out.loc[n - 3, spec.target_col])
    assert pd.isna(out.loc[n - 1, spec.target_col])
