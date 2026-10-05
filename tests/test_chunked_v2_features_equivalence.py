from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_pump.v2.features import V2FeatureBuilder


def _fixture() -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=180, freq="5min", tz="UTC")
    rows = []
    for k, symbol in enumerate(["LATOKEN:BTCUSDT", "LATOKEN:AAAUSDT", "LATOKEN:BBBUSDT", "LATOKEN:CCCUSDT"]):
        base = 100.0 + 10 * k
        for i, t in enumerate(ts):
            close = base * (1 + 0.0007 * i + 0.01 * np.sin((i + 3 * k) / 11))
            open_ = close * (1 - 0.001 * np.cos(i / 7))
            high = max(open_, close) * 1.004
            low = min(open_, close) * 0.996
            vol = 1000 + 20 * k + 3 * i + 15 * np.sin(i / 5)
            rows.append({
                "timestamp": t,
                "symbol": symbol,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": vol,
                "quote_volume": vol * close,
            })
    return pd.DataFrame(rows)


def test_chunked_symbol_batches_match_monolithic_v2_features():
    from crypto_pump.v2.chunked_features import transform_v2_in_symbol_chunks

    frame = _fixture()
    mono = V2FeatureBuilder(benchmark_symbol="LATOKEN:BTCUSDT").transform(frame)
    chunked = transform_v2_in_symbol_chunks(
        frame,
        benchmark_symbol="LATOKEN:BTCUSDT",
        symbols_per_chunk=1,
    )

    key = ["symbol", "timestamp"]
    mono = mono.sort_values(key).reset_index(drop=True)
    chunked = chunked.sort_values(key).reset_index(drop=True)
    assert mono[key].equals(chunked[key])
    assert list(mono.columns) == list(chunked.columns)

    numeric = mono.select_dtypes(include=[np.number]).columns
    for col in numeric:
        np.testing.assert_allclose(
            mono[col].to_numpy(float),
            chunked[col].to_numpy(float),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
            err_msg=col,
        )
