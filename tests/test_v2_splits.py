import pandas as pd

from crypto_pump.v2.splits import purged_walk_forward_folds


def test_purged_walk_forward_has_required_gaps():
    times = pd.date_range("2026-01-01", periods=300, freq="5min", tz="UTC")
    df = pd.DataFrame({"timestamp": times, "symbol": "X"})
    folds = purged_walk_forward_folds(
        df, n_splits=3, min_train_fraction=0.35,
        validation_fraction=0.08, purge_bars=12, embargo_bars=4
    )
    pos = {t: i for i, t in enumerate(times)}
    for f in folds:
        train_end = pos[pd.Timestamp(f.train_times[-1])]
        val_start = pos[pd.Timestamp(f.validation_times[0])]
        val_end = pos[pd.Timestamp(f.validation_times[-1])]
        test_start = pos[pd.Timestamp(f.test_times[0])]
        assert val_start - train_end - 1 >= 12
        assert test_start - val_end - 1 >= 16
