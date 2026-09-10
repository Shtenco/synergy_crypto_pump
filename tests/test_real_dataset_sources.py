import pandas as pd
from crypto_pump.data_sources.merge import merge_and_regularize_panels
from crypto_pump.data_sources.uniswap_cryo import _decode_pools


def test_poolcreated_raw_decode():
    token0 = "0x" + "11" * 20
    token1 = "0x" + "22" * 20
    pool = "0x" + "33" * 20
    topic = lambda a: "0x" + "00" * 12 + a[2:]
    data = "0x" + ("00" * 32) + ("00" * 12) + pool[2:]
    raw = pd.DataFrame([{"block_number": 1, "topic1": topic(token0), "topic2": topic(token1), "topic3": "0x000bb8", "data": data, "address": "0xfactory"}])
    d = _decode_pools(raw)
    assert d.iloc[0].token0 == token0
    assert d.iloc[0].token1 == token1
    assert d.iloc[0].pool == pool
    assert d.iloc[0].fee == 3000


def test_regularizer_makes_true_5m_horizon():
    p = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T00:10Z"]),
        "symbol": ["X", "X"], "open": [1, 2], "high": [1, 2], "low": [1, 2], "close": [1, 2], "volume": [3, 4], "source": ["x", "x"]
    })
    out = merge_and_regularize_panels([p], min_bars=3)
    assert len(out) == 3
    assert out.iloc[1].close == 1
    assert out.iloc[1].volume == 0
