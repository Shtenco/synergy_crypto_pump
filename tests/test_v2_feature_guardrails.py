import numpy as np
import pandas as pd

from crypto_pump.v2.features import v2_numeric_feature_columns


def test_v2_numeric_features_exclude_pipeline_metadata_and_ids():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC"),
        "symbol": ["A"] * 4,
        "open": [1.0] * 4,
        "high": [1.1] * 4,
        "low": [0.9] * 4,
        "close": [1.0] * 4,
        "volume": [10.0] * 4,
        "series_segment": [0, 0, 1, 1],
        "row_id": [1, 2, 3, 4],
        "pair_id": [10, 10, 10, 10],
        "block_number": [100, 101, 102, 103],
        "custom_entity_id": [7, 7, 7, 7],
        "transaction_index": [0, 1, 0, 2],
        "signal": [0.1, 0.2, 0.3, 0.4],
    })
    features = v2_numeric_feature_columns(frame)
    assert "signal" in features
    assert "series_segment" not in features
    assert "row_id" not in features
    assert "pair_id" not in features
    assert "block_number" not in features
    assert "custom_entity_id" not in features
    assert "transaction_index" not in features


def test_v2_numeric_features_drop_all_nan_columns():
    frame = pd.DataFrame({
        "open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0],
        "close": [1.0, 1.0], "volume": [1.0, 1.0],
        "valid_signal": [0.0, 1.0], "empty_signal": [np.nan, np.nan],
    })
    features = v2_numeric_feature_columns(frame)
    assert "valid_signal" in features
    assert "empty_signal" not in features
