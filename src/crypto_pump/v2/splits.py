from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    fold: int
    train_times: np.ndarray
    validation_times: np.ndarray
    test_times: np.ndarray

    def frames(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        ts = pd.to_datetime(frame["timestamp"], utc=True)
        return (
            frame[ts.isin(self.train_times)].copy(),
            frame[ts.isin(self.validation_times)].copy(),
            frame[ts.isin(self.test_times)].copy(),
        )


def purged_walk_forward_folds(
    frame: pd.DataFrame,
    n_splits: int = 3,
    min_train_fraction: float = 0.45,
    validation_fraction: float = 0.10,
    purge_bars: int = 0,
    embargo_bars: int = 0,
) -> list[WalkForwardFold]:
    """Expanding-window OOS folds with label purge and post-validation embargo.

    `purge_bars` should be >= the longest label horizon. This prevents a train
    label from using any price observation inside validation, and a validation
    label from using any observation inside test.
    """
    times = np.array(sorted(pd.to_datetime(frame["timestamp"], utc=True).unique()))
    n = len(times)
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if n < 20:
        raise ValueError("not enough unique timestamps for walk-forward evaluation")

    min_train = max(5, int(n * min_train_fraction))
    val_size = max(2, int(n * validation_fraction))
    required_before_test = min_train + purge_bars + val_size + purge_bars + embargo_bars
    remaining = n - required_before_test
    if remaining < n_splits:
        raise ValueError(
            f"not enough history for purge={purge_bars}, embargo={embargo_bars}, "
            f"{n_splits} folds: have {n} unique timestamps"
        )
    test_size = max(1, remaining // n_splits)
    folds: list[WalkForwardFold] = []

    for fold in range(n_splits):
        test_start = required_before_test + fold * test_size
        test_end = n if fold == n_splits - 1 else min(n, test_start + test_size)
        val_end = test_start - purge_bars - embargo_bars
        val_start = val_end - val_size
        train_end = val_start - purge_bars
        if train_end < min_train:
            raise RuntimeError("internal split construction violated min_train")
        folds.append(
            WalkForwardFold(
                fold=fold,
                train_times=times[:train_end],
                validation_times=times[val_start:val_end],
                test_times=times[test_start:test_end],
            )
        )
    return folds
