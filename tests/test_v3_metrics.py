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



def test_financial_report_counts_first_loss_as_drawdown_and_sorts_time():
    trades = pd.DataFrame({
        "exit_timestamp": [
            "2026-01-03T00:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z",
        ],
        "executable": [True, True, True],
        "net_pnl_usd": [2.0, -5.0, 1.0],
        "net_return": [0.02, -0.05, 0.01],
    })
    rep = financial_report(trades)
    assert rep["net_pnl_usd"] == -2.0
    assert rep["max_drawdown_usd"] == -5.0
