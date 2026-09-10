from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.astype(float) / b.astype(float).replace(0.0, np.nan)


def _past_zscore(s: pd.Series, window: int) -> pd.Series:
    # rolling window includes current observation only; never future rows
    mean = s.rolling(window, min_periods=max(3, window // 4)).mean()
    std = s.rolling(window, min_periods=max(3, window // 4)).std(ddof=0)
    return (s - mean) / std.replace(0.0, np.nan)


@dataclass(slots=True)
class FeatureBuilder:
    windows: tuple[int, ...] = (3, 6, 12, 24, 48)
    z_window: int = 48

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = REQUIRED.difference(frame.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")

        out = frame.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        out = out.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
        g = out.groupby("symbol", group_keys=False, sort=False)

        out["ret_1"] = g["close"].pct_change(fill_method=None)
        log_close = np.log(out["close"].clip(lower=1e-18))
        out["log_ret_1"] = log_close.groupby(out["symbol"]).diff()
        out["range_frac"] = _safe_div(out["high"] - out["low"], out["close"])
        out["body_frac"] = _safe_div((out["close"] - out["open"]).abs(), out["open"])
        out["volume_log1p"] = np.log1p(out["volume"].clip(lower=0))

        for w in self.windows:
            grp = out.groupby("symbol", group_keys=False, sort=False)
            out[f"ret_{w}"] = grp["close"].pct_change(w, fill_method=None)
            out[f"volatility_{w}"] = grp["log_ret_1"].transform(lambda s: s.rolling(w, min_periods=max(2, w // 3)).std(ddof=0))
            out[f"volume_mean_{w}"] = grp["volume"].transform(lambda s: s.rolling(w, min_periods=max(2, w // 3)).mean())
            out[f"volume_ratio_{w}"] = _safe_div(out["volume"], out[f"volume_mean_{w}"])
            rolling_high = grp["high"].transform(lambda s: s.rolling(w, min_periods=max(2, w // 3)).max())
            rolling_low = grp["low"].transform(lambda s: s.rolling(w, min_periods=max(2, w // 3)).min())
            out[f"compression_{w}"] = _safe_div(rolling_high - rolling_low, out["close"])
            out[f"distance_high_{w}"] = _safe_div(out["close"], rolling_high) - 1.0
            out[f"distance_low_{w}"] = _safe_div(out["close"], rolling_low) - 1.0

        grp = out.groupby("symbol", group_keys=False, sort=False)
        out["volume_z"] = grp["volume_log1p"].transform(lambda s: _past_zscore(s, self.z_window))
        out["volume_accel"] = grp["volume_log1p"].diff().diff()
        out["return_accel"] = grp["ret_1"].diff()

        if {"buy_volume", "sell_volume"}.issubset(out.columns):
            denom = out["buy_volume"].fillna(0) + out["sell_volume"].fillna(0)
            out["flow_imbalance"] = _safe_div(out["buy_volume"].fillna(0) - out["sell_volume"].fillna(0), denom)
        if {"bid_depth", "ask_depth"}.issubset(out.columns):
            denom = out["bid_depth"].fillna(0) + out["ask_depth"].fillna(0)
            out["depth_imbalance"] = _safe_div(out["bid_depth"].fillna(0) - out["ask_depth"].fillna(0), denom)

        derivative_cols = [
            "trades", "unique_buyers", "unique_sellers", "holders", "liquidity",
            "social_mentions", "social_users"
        ]
        for col in derivative_cols:
            if col in out.columns:
                grp = out.groupby("symbol", group_keys=False, sort=False)
                base = np.log1p(out[col].clip(lower=0)) if (out[col].dropna() >= 0).all() else out[col]
                out[f"{col}_velocity"] = base.groupby(out["symbol"]).diff()
                out[f"{col}_accel"] = base.groupby(out["symbol"]).diff().diff()

        # Cross-sectional context: rank each coin versus the universe at exactly the same timestamp.
        for col in ["ret_1", "volume_z", "volume_accel"]:
            out[f"xrank_{col}"] = out.groupby("timestamp")[col].rank(pct=True)

        return out.replace([np.inf, -np.inf], np.nan)


def numeric_feature_columns(frame: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    blocked = {"target", "future_max_return", "future_min_return", "future_end_return", *exclude}
    return [c for c in frame.select_dtypes(include=[np.number]).columns if c not in blocked]
