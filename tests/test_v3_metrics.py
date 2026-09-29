import pandas as pd

from crypto_pump.v3.metrics import financial_report, stress_test_trades


def test_financial_report_and_winner_stress():
    trades = pd.DataFrame({
        "executable": [True] * 5,
        "net_pnl_usd": [1.0, 2.0, -1.0, 1.0, 20.0],
        "net_return": [0.01, 0.02, -0.01, 0.01, 0.20],
    })
    rep = financial_report(trades)
    assert rep["trades"] == 5
    assert rep["net_pnl_usd"] == 23.0
    stress = stress_test_trades(trades)
    assert stress["remove_best_1"]["net_pnl_usd"] == 3.0
