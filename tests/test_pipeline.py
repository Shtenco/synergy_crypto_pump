import numpy as np
import pandas as pd

from crypto_pump.features import FeatureBuilder
from crypto_pump.labels import add_forward_labels


def sample_frame(n=30):
    rows = []
    for symbol, offset in [("AAA", 0.0), ("BBB", 0.02)]:
        for i in range(n):
            rows.append({
                "timestamp": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=i),
                "symbol": symbol,
                "open": 1 + offset + i * 0.001,
                "high": 1.01 + offset + i * 0.001,
                "low": 0.99 + offset + i * 0.001,
                "close": 1 + offset + i * 0.001,
                "volume": 100 + i,
                "buy_volume": 60 + i,
                "sell_volume": 40,
                "social_mentions": 10 + i,
            })
    return pd.DataFrame(rows)


def test_features_do_not_change_when_future_is_modified():
    df = sample_frame()
    cutoff = pd.Timestamp("2026-01-01 00:15:00", tz="UTC")
    base = FeatureBuilder().transform(df)
    changed = df.copy()
    changed.loc[changed["timestamp"] > cutoff, ["close", "volume"]] *= 1000
    altered = FeatureBuilder().transform(changed)
    cols = [c for c in base.columns if c not in df.columns]
    left = base[base["timestamp"] <= cutoff][cols].reset_index(drop=True)
    right = altered[altered["timestamp"] <= cutoff][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False)


def test_labels_are_strictly_forward():
    df = sample_frame(n=8)
    out = add_forward_labels(df, horizon=2, threshold=0.001)
    aaa = out[out.symbol == "AAA"].reset_index(drop=True)
    expected = aaa.loc[1:2, "close"].max() / aaa.loc[0, "close"] - 1
    assert np.isclose(aaa.loc[0, "future_max_return"], expected)
    assert pd.isna(aaa.loc[len(aaa) - 1, "target"])


def test_cross_sectional_rank_is_bounded():
    out = FeatureBuilder().transform(sample_frame())
    x = out["xrank_ret_1"].dropna()
    assert ((x > 0) & (x <= 1)).all()
