from __future__ import annotations

import numpy as np
import pandas as pd


def _forward_extrema(close: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    fmax = np.full(n, np.nan, dtype=float)
    fmin = np.full(n, np.nan, dtype=float)
    fend = np.full(n, np.nan, dtype=float)
    for i in range(n):
        j = min(n, i + horizon + 1)
        future = close[i + 1 : j]
        if future.size:
            fmax[i] = np.nanmax(future)
            fmin[i] = np.nanmin(future)
            fend[i] = future[-1]
    return fmax, fmin, fend


def add_forward_labels(frame: pd.DataFrame, horizon: int = 12, threshold: float = 0.20) -> pd.DataFrame:
    """Create labels using only t+1..t+horizon. These columns must never enter features."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    out = frame.sort_values(["symbol", "timestamp"], kind="mergesort").copy()
    parts = []
    for _, part in out.groupby("symbol", sort=False):
        p = part.copy()
        close = p["close"].to_numpy(dtype=float)
        fmax, fmin, fend = _forward_extrema(close, horizon)
        p["future_max_return"] = fmax / close - 1.0
        p["future_min_return"] = fmin / close - 1.0
        p["future_end_return"] = fend / close - 1.0
        p["target"] = (p["future_max_return"] >= threshold).astype("Int64")
        p.loc[p["future_max_return"].isna(), "target"] = pd.NA
        parts.append(p)
    return pd.concat(parts, ignore_index=True)
