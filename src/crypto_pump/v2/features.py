from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ..features import FeatureBuilder as V1FeatureBuilder
from .bocd import bocd_change_probability
from .config import DEFAULT_HORIZONS
from .labels import v2_label_columns


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.astype(float) / b.astype(float).replace(0.0, np.nan)


def _past_zscore(s: pd.Series, window: int) -> pd.Series:
    minp = max(3, window // 4)
    mean = s.rolling(window, min_periods=minp).mean()
    std = s.rolling(window, min_periods=minp).std(ddof=0)
    return (s - mean) / std.replace(0.0, np.nan)


def _rsi(s: pd.Series, window: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).rolling(window, min_periods=window).mean()
    dn = (-d.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = up / dn.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


class V2FeatureBuilder:
    """Clean-room feature distillation from the 15 pinned research sources.

    Every rolling/derivative feature is causal: its value at t only uses rows <= t.
    """

    def __init__(
        self,
        windows: tuple[int, ...] = (3, 6, 12, 24, 48, 96),
        z_window: int = 96,
        benchmark_symbol: str = "BTCUSDT",
        bocd_hazard_bars: int = 40,
        bocd_max_run_length: int = 128,
    ):
        self.windows = windows
        self.z_window = z_window
        self.benchmark_symbol = benchmark_symbol
        self.bocd_hazard_bars = bocd_hazard_bars
        self.bocd_max_run_length = bocd_max_run_length

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = V1FeatureBuilder(windows=self.windows, z_window=self.z_window).transform(frame)
        out = out.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
        g = out.groupby("symbol", group_keys=False, sort=False)

        out["quote_volume"] = (
            out["quote_volume"].astype(float)
            if "quote_volume" in out
            else out["volume"].astype(float) * out["close"].astype(float)
        )
        out["quote_volume_log1p"] = np.log1p(out["quote_volume"].clip(lower=0))
        out["quote_abs_zscore"] = g["quote_volume_log1p"].transform(
            lambda s: _past_zscore(s, self.z_window)
        )
        out["current_candle_return"] = _safe_div(out["close"], out["open"]) - 1.0
        out["rsi_14"] = g["close"].transform(_rsi)

        if "taker_buy_quote" in out.columns:
            out["buy_pressure"] = _safe_div(out["taker_buy_quote"], out["quote_volume"])
        elif "buy_volume" in out.columns:
            out["buy_pressure"] = _safe_div(out["buy_volume"], out["volume"])

        if {"buy_trade_count", "trades"}.issubset(out.columns):
            out["share_of_long_trades"] = _safe_div(out["buy_trade_count"], out["trades"])

        if {"quote_slippage_signed", "quote_slippage_abs"}.issubset(out.columns):
            out["slippage_imbalance"] = _safe_div(
                out["quote_slippage_signed"], out["quote_slippage_abs"]
            )
        elif {"buy_slippage", "sell_slippage"}.issubset(out.columns):
            denom = out["buy_slippage"].abs() + out["sell_slippage"].abs()
            out["slippage_imbalance"] = _safe_div(
                out["buy_slippage"] - out["sell_slippage"], denom
            )

        duration_col = next(
            (c for c in ("activity_duration_sec", "active_duration_secs", "age_seconds") if c in out),
            None,
        )
        if duration_col and "trades" in out:
            out["tx_per_sec"] = _safe_div(out["trades"], out[duration_col].clip(lower=1))
        elif duration_col and "num_trades" in out:
            out["tx_per_sec"] = _safe_div(out["num_trades"], out[duration_col].clip(lower=1))

        decay_sources = [
            c for c in ("quote_volume", "trades", "unique_buyers", "liquidity")
            if c in out.columns
        ]
        for col in decay_sources:
            group = out.groupby("symbol", group_keys=False, sort=False)[col]
            fast = group.transform(lambda s: s.ewm(span=6, adjust=False, min_periods=2).mean())
            slow = group.transform(lambda s: s.ewm(span=24, adjust=False, min_periods=4).mean())
            out[f"{col}_ewm_fast_slow"] = _safe_div(fast, slow) - 1.0

        if "creator_num_tokens" in out:
            out["creator_num_tokens_log1p"] = np.log1p(out["creator_num_tokens"].clip(lower=0))
        if "current_vsol" in out:
            out["current_vsol_log1p"] = np.log1p(out["current_vsol"].clip(lower=0))
        if {"current_vsol", "initial_vsol"}.issubset(out.columns):
            out["vsol_growth"] = _safe_div(out["current_vsol"], out["initial_vsol"]) - 1.0

        conc = [c for c in ("top_buyer_pct", "top3_buyer_pct", "top5_share", "wallet_gini") if c in out]
        if conc:
            out["wallet_concentration_mean"] = out[conc].astype(float).mean(axis=1)
        bot_cols = [
            c for c in (
                "repeat_buyer_rate", "dust_buy_rate", "unknown_buyer_pct",
                "low_history_pct", "sniper_pct"
            ) if c in out
        ]
        if bot_cols:
            out["wallet_bot_pressure"] = out[bot_cols].astype(float).mean(axis=1)
        if {"trades", "unique_buyers"}.issubset(out.columns):
            out["buys_per_buyer_proxy"] = _safe_div(out["trades"], out["unique_buyers"])

        for col in ("social_mentions", "social_users"):
            if col in out:
                out[f"{col}_log1p"] = np.log1p(out[col].clip(lower=0))
                out[f"{col}_z"] = out.groupby("symbol", sort=False)[f"{col}_log1p"].transform(
                    lambda s: _past_zscore(s, self.z_window)
                )
        if {"social_users", "social_mentions"}.issubset(out.columns):
            out["social_unique_user_ratio"] = _safe_div(out["social_users"], out["social_mentions"])
        if "sentiment" in out:
            out["sentiment_delta"] = g["sentiment"].diff()
            out["sentiment_z"] = g["sentiment"].transform(lambda s: _past_zscore(s, self.z_window))
        coordination_parts = [
            c for c in ("coordination_score", "cross_platform_breadth", "new_account_rate")
            if c in out
        ]
        if coordination_parts:
            zparts = []
            for col in coordination_parts:
                zc = f"_{col}_xrank"
                out[zc] = out.groupby("timestamp")[col].rank(pct=True)
                zparts.append(zc)
            out["social_coordination_composite"] = out[zparts].mean(axis=1)

        bench = out.loc[out["symbol"] == self.benchmark_symbol, ["timestamp", "ret_1"]].drop_duplicates("timestamp")
        if len(bench):
            bench = bench.rename(columns={"ret_1": "benchmark_ret_1"})
            out = out.merge(bench, on="timestamp", how="left", sort=False)
            out["market_residual_1"] = out["ret_1"] - out["benchmark_ret_1"]
        else:
            out["market_residual_1"] = out["ret_1"]

        out["market_residual_z"] = out.groupby("symbol", sort=False)["market_residual_1"].transform(
            lambda s: _past_zscore(s, self.z_window)
        )
        cp_parts = []
        for _, p in out.groupby("symbol", sort=False):
            cp = bocd_change_probability(
                p["market_residual_z"].to_numpy(float),
                hazard_bars=self.bocd_hazard_bars,
                max_run_length=self.bocd_max_run_length,
            )
            cp_parts.append(pd.Series(cp, index=p.index))
        out["bocd_cp_prob"] = pd.concat(cp_parts).sort_index() if cp_parts else np.nan

        out["pre_pump_activity_score"] = (
            out["quote_abs_zscore"].fillna(0)
            + out["volume_z"].fillna(0)
            - 2.0 * out["current_candle_return"].abs().clip(0, 0.25)
        )

        rank_cols = [
            c for c in (
                "quote_abs_zscore", "flow_imbalance", "slippage_imbalance", "tx_per_sec",
                "wallet_concentration_mean", "social_mentions_z", "social_users_z",
                "bocd_cp_prob", "pre_pump_activity_score"
            ) if c in out
        ]
        for col in rank_cols:
            out[f"xrank_{col}"] = out.groupby("timestamp")[col].rank(pct=True)

        helper_cols = [c for c in out if c.startswith("_") and c.endswith("_xrank")]
        if helper_cols:
            out = out.drop(columns=helper_cols)
        return out.replace([np.inf, -np.inf], np.nan)


def v2_numeric_feature_columns(frame: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    blocked = v2_label_columns(DEFAULT_HORIZONS) | {
        "target", "future_max_return", "future_min_return", "future_end_return",
        "open", "high", "low", "close", "volume", *exclude
    }
    return [
        c for c in frame.select_dtypes(include=[np.number]).columns
        if c not in blocked
        and not c.startswith(("target__", "mfe__", "mae__", "end_return__", "hit_bar__"))
        and frame[c].notna().any()
    ]


def social_feature_columns(frame: pd.DataFrame) -> list[str]:
    hints = ("social", "sentiment", "coordination", "cross_platform", "new_account")
    return [
        c for c in v2_numeric_feature_columns(frame)
        if any(h in c.lower() for h in hints)
    ]
