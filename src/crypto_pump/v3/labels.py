from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class TripleBarrierSpec:
    name: str
    horizon_bars: int
    take_profit: float
    stop_loss: float

    @property
    def target_col(self) -> str:
        return f"tb_target__{self.name}"

    @property
    def exit_bar_col(self) -> str:
        return f"tb_exit_bar__{self.name}"

    @property
    def exit_return_col(self) -> str:
        return f"tb_exit_return__{self.name}"

    @property
    def reason_col(self) -> str:
        return f"tb_exit_reason__{self.name}"


DEFAULT_TRIPLE_BARRIERS: tuple[TripleBarrierSpec, ...] = (
    TripleBarrierSpec("tp10_sl05_15m", 3, 0.10, 0.05),
    TripleBarrierSpec("tp20_sl08_1h", 12, 0.20, 0.08),
    TripleBarrierSpec("tp50_sl15_6h", 72, 0.50, 0.15),
    TripleBarrierSpec("tp100_sl25_24h", 288, 1.00, 0.25),
)


def _one_path(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    i: int,
    spec: TripleBarrierSpec,
    entry_delay_bars: int,
) -> tuple[float, float, float, str | None]:
    entry_idx = i + entry_delay_bars
    if entry_idx >= len(close):
        return np.nan, np.nan, np.nan, None

    stop = entry_idx + spec.horizon_bars
    if stop >= len(close):
        return np.nan, np.nan, np.nan, None

    entry = close[entry_idx]
    if not np.isfinite(entry) or entry <= 0:
        return np.nan, np.nan, np.nan, None

    tp_px = entry * (1.0 + spec.take_profit)
    sl_px = entry * (1.0 - spec.stop_loss)

    for j in range(entry_idx + 1, stop + 1):
        h = high[j]
        l = low[j]
        hit_tp = np.isfinite(h) and h >= tp_px
        hit_sl = np.isfinite(l) and l <= sl_px
        if hit_tp and hit_sl:
            return 0.0, float(j - entry_idx), -spec.stop_loss, "both_stop_first"
        if hit_sl:
            return 0.0, float(j - entry_idx), -spec.stop_loss, "stop"
        if hit_tp:
            return 1.0, float(j - entry_idx), spec.take_profit, "take_profit"

    end_px = close[stop]
    if not np.isfinite(end_px):
        return np.nan, np.nan, np.nan, None
    ret = end_px / entry - 1.0
    return 0.0, float(spec.horizon_bars), float(ret), "timeout"


def add_triple_barrier_labels(
    frame: pd.DataFrame,
    specs: Iterable[TripleBarrierSpec] = DEFAULT_TRIPLE_BARRIERS,
    entry_delay_bars: int = 1,
) -> pd.DataFrame:
    """Create first-passage labels where TP must be touched before SL.

    The signal is assumed known at the close of bar t. By default entry occurs
    one complete bar later, avoiding an impossible fill at signal close.
    Tail rows without the full horizon are censored.
    """
    if entry_delay_bars < 0:
        raise ValueError("entry_delay_bars must be >= 0")
    required = {"timestamp", "symbol", "close", "high", "low"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
    parts: list[pd.DataFrame] = []

    for _, part in out.groupby("symbol", sort=False):
        p = part.copy()
        close = pd.to_numeric(p["close"], errors="coerce").to_numpy(float)
        high = pd.to_numeric(p["high"], errors="coerce").to_numpy(float)
        low = pd.to_numeric(p["low"], errors="coerce").to_numpy(float)

        for spec in specs:
            target = np.full(len(p), np.nan)
            exit_bar = np.full(len(p), np.nan)
            exit_ret = np.full(len(p), np.nan)
            reasons: list[str | None] = [None] * len(p)
            for i in range(len(p)):
                t, b, r, reason = _one_path(close, high, low, i, spec, entry_delay_bars)
                target[i] = t
                exit_bar[i] = b
                exit_ret[i] = r
                reasons[i] = reason
            p[spec.target_col] = pd.array(target, dtype="Int64")
            p[spec.exit_bar_col] = pd.array(exit_bar, dtype="Float64")
            p[spec.exit_return_col] = exit_ret
            p[spec.reason_col] = pd.Series(reasons, dtype="string")
        parts.append(p)

    return pd.concat(parts, ignore_index=True)
