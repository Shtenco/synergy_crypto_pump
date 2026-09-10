from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .config import DEFAULT_HORIZONS, HorizonSpec


def _forward_path_stats(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    horizon: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Strict fixed-horizon labels. Tail rows without a complete horizon are censored."""
    n = len(close)
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    end = np.full(n, np.nan)
    hit = np.full(n, np.nan)
    for i in range(n):
        stop = i + horizon + 1
        if stop > n:
            continue
        base = close[i]
        if not np.isfinite(base) or base <= 0:
            continue
        fh = high[i + 1 : stop]
        fl = low[i + 1 : stop]
        fc = close[i + 1 : stop]
        if len(fc) != horizon:
            continue
        mfe[i] = np.nanmax(fh) / base - 1.0
        mae[i] = np.nanmin(fl) / base - 1.0
        end[i] = fc[-1] / base - 1.0
        touched = np.flatnonzero((fh / base - 1.0) >= threshold)
        if touched.size:
            hit[i] = float(touched[0] + 1)
    return mfe, mae, end, hit


def add_multi_horizon_labels(
    frame: pd.DataFrame,
    specs: Sequence[HorizonSpec] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    required = {"timestamp", "symbol", "close", "high", "low"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
    parts: list[pd.DataFrame] = []
    for _, p in out.groupby("symbol", sort=False):
        p = p.copy()
        close = p["close"].to_numpy(float)
        high = p["high"].to_numpy(float)
        low = p["low"].to_numpy(float)
        for spec in specs:
            mfe, mae, end, hit = _forward_path_stats(
                close, high, low, spec.horizon_bars, spec.return_threshold
            )
            p[spec.mfe_col] = mfe
            p[spec.mae_col] = mae
            p[spec.end_col] = end
            p[spec.hit_col] = pd.array(hit, dtype="Float64")
            target = pd.Series(
                np.where(np.isnan(mfe), np.nan, (mfe >= spec.return_threshold).astype(float)),
                index=p.index,
                dtype="Float64",
            )
            p[spec.target_col] = target.astype("Int64")
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def v2_label_columns(specs: Sequence[HorizonSpec] = DEFAULT_HORIZONS) -> set[str]:
    cols: set[str] = set()
    for s in specs:
        cols.update({s.target_col, s.mfe_col, s.mae_col, s.end_col, s.hit_col})
    return cols
