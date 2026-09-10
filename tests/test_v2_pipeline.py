import numpy as np
import pandas as pd

from crypto_pump.v2.config import HorizonSpec
from crypto_pump.v2.features import V2FeatureBuilder
from crypto_pump.v2.labels import add_multi_horizon_labels
from crypto_pump.v2.pipeline import V2Ensemble


def test_v2_core_ensemble_smoke():
    rng = np.random.default_rng(7)
    times = pd.date_range("2026-01-01", periods=100, freq="5min", tz="UTC")
    rows = []
    for j, sym in enumerate(["BTCUSDT", "AAAUSDT", "BBBUSDT"]):
        p = 100.0 if j == 0 else 1.0
        for i, ts in enumerate(times):
            p *= np.exp(rng.normal(0, 0.006) + (0.025 if sym == "AAAUSDT" and i in (45, 70) else 0))
            rows.append({
                "timestamp": ts, "symbol": sym, "open": p * .999,
                "high": p * 1.01, "low": p * .99, "close": p,
                "volume": 100 + 200 * rng.random(),
                "buy_volume": 60 + 80 * rng.random(),
                "sell_volume": 50 + 80 * rng.random(),
            })
    spec = HorizonSpec("smoke", 3, 0.01)
    frame = V2FeatureBuilder(z_window=24).transform(pd.DataFrame(rows))
    frame = add_multi_horizon_labels(frame, [spec])
    unique = sorted(frame["timestamp"].unique())
    train = frame[frame["timestamp"].isin(unique[:60])]
    val = frame[frame["timestamp"].isin(unique[64:80])]
    test = frame[frame["timestamp"].isin(unique[84:96])]
    model = V2Ensemble(specs=(spec,), model_kind="hist").fit(train, val)
    pred = model.predict(test)
    col = "pump_probability__smoke"
    assert col in pred
    assert pred[col].between(0, 1).all()
