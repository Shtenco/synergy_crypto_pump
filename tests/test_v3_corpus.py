from __future__ import annotations

import pandas as pd

from crypto_pump.v3.corpus import SYSTEM_PROGRAM_WALLET, build_clean_hf_panel


def test_historical_corpus_filters_and_builds_panel(tmp_path):
    root = tmp_path / "corpus"
    (root / "trades").mkdir(parents=True)
    tokens = pd.DataFrame({
        "mint": ["GOOD", "BAD"],
        "top10_pct_suspect": [False, True],
    })
    tokens.to_parquet(root / "tokens.parquet", index=False)
    trades = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-06-10T00:01:00Z",
            "2026-06-10T00:02:00Z",
            "2026-06-10T00:03:00Z",
            "2026-06-10T00:04:00Z",
        ]),
        "mint": ["GOOD", "GOOD", "GOOD", "BAD"],
        "price_sol": [0.001, 0.002, 0.0015, 0.005],
        "sol_amount": [1.0, 1.0, 1.0, 1.0],
        "token_amount": [1000.0, 500.0, 1000.0, 200.0],
        "is_buy": [True, False, True, True],
        "user_wallet": ["wallet-a", "wallet-b", SYSTEM_PROGRAM_WALLET, "wallet-c"],
    })
    trades.to_parquet(root / "trades" / "part-0.parquet", index=False)

    output = tmp_path / "panel.parquet"
    report = build_clean_hf_panel(root, output, threads=1, memory_limit="1GB")
    panel = pd.read_parquet(output)

    # GOOD has two valid trades. System-program row and suspect BAD token are removed.
    assert report.input_trades == 4
    assert report.clean_trades == 2
    assert report.rejected_trades == 2
    assert report.symbols == 1
    assert set(panel["symbol"]) == {"GOOD"}
    assert panel.iloc[0]["trades"] == 2
    assert panel.iloc[0]["unique_buyers"] == 1
    assert report.sha256
    assert output.with_suffix(".parquet.manifest.json").exists()
