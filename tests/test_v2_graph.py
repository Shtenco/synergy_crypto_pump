import pandas as pd

from crypto_pump.v2.graph import build_wallet_token_snapshot


def test_graph_snapshot_excludes_future_edges():
    tx = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:01:00Z",
            "2026-01-01T00:02:00Z",
        ]),
        "wallet": ["w1", "w2", "future_wallet"],
        "symbol": ["AAA", "AAA", "ZZZ"],
        "amount": [1.0, 2.0, 100.0],
        "direction": ["buy", "buy", "buy"],
    })
    snap = build_wallet_token_snapshot(tx, "2026-01-01T00:01:00Z", lookback=None)
    assert "AAA" in snap.token_nodes
    assert "ZZZ" not in snap.token_nodes
    assert snap.edge_index.shape[1] == 4
