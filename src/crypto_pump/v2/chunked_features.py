from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .features import V2FeatureBuilder


_V1_XRANK_SOURCES = ("ret_1", "volume_z", "volume_accel")
_V2_XRANK_SOURCES = (
    "quote_abs_zscore",
    "flow_imbalance",
    "slippage_imbalance",
    "tx_per_sec",
    "wallet_concentration_mean",
    "social_mentions_z",
    "social_users_z",
    "bocd_cp_prob",
    "pre_pump_activity_score",
)
_COORDINATION_SOURCES = (
    "coordination_score",
    "cross_platform_breadth",
    "new_account_rate",
)


def _recompute_global_cross_sectional_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Restore every timestamp-global feature after symbol-local chunking.

    The historical V1/V2 builders compute rolling/EWM/BOCD features per symbol,
    but ranks across the whole universe at each timestamp. Symbol chunks make the
    former memory-safe; this function restores the latter exactly after concat.
    """
    out = frame.copy()

    coordination = [c for c in _COORDINATION_SOURCES if c in out.columns]
    if coordination:
        rank_parts = [out.groupby("timestamp")[c].rank(pct=True) for c in coordination]
        out["social_coordination_composite"] = pd.concat(rank_parts, axis=1).mean(axis=1)

    for col in (*_V1_XRANK_SOURCES, *_V2_XRANK_SOURCES):
        if col in out.columns:
            out[f"xrank_{col}"] = out.groupby("timestamp")[col].rank(pct=True)

    return out.replace([np.inf, -np.inf], np.nan)


def transform_v2_in_symbol_chunks(
    frame: pd.DataFrame,
    *,
    benchmark_symbol: str = "BTCUSDT",
    symbols_per_chunk: int = 32,
    windows: tuple[int, ...] = (3, 6, 12, 24, 48, 96),
    z_window: int = 96,
    bocd_hazard_bars: int = 40,
    bocd_max_run_length: int = 128,
) -> pd.DataFrame:
    """Compute historical V2 features with bounded memory and exact semantics.

    Each non-benchmark symbol belongs to exactly one batch. The benchmark is
    injected into every batch solely so benchmark_ret_1/market_residual match the
    monolithic historical builder. Benchmark output is retained only once.
    Timestamp-global ranks are deliberately recomputed after all batches are
    concatenated, restoring the full-universe historical values.
    """
    if symbols_per_chunk < 1:
        raise ValueError("symbols_per_chunk must be >= 1")
    if "symbol" not in frame.columns:
        raise ValueError("missing required column: symbol")

    symbols = list(pd.unique(frame["symbol"]))
    has_benchmark = benchmark_symbol in set(symbols)
    non_benchmark = [s for s in symbols if s != benchmark_symbol]

    builder_kwargs = dict(
        windows=windows,
        z_window=z_window,
        benchmark_symbol=benchmark_symbol,
        bocd_hazard_bars=bocd_hazard_bars,
        bocd_max_run_length=bocd_max_run_length,
    )
    parts: list[pd.DataFrame] = []

    if has_benchmark:
        bench_input = frame.loc[frame["symbol"].eq(benchmark_symbol)].copy()
        parts.append(V2FeatureBuilder(**builder_kwargs).transform(bench_input))

    for start in range(0, len(non_benchmark), symbols_per_chunk):
        batch_symbols = non_benchmark[start:start + symbols_per_chunk]
        wanted = set(batch_symbols)
        if has_benchmark:
            wanted.add(benchmark_symbol)
        chunk_input = frame.loc[frame["symbol"].isin(wanted)].copy()
        transformed = V2FeatureBuilder(**builder_kwargs).transform(chunk_input)
        transformed = transformed.loc[transformed["symbol"].isin(batch_symbols)].copy()
        parts.append(transformed)

    if not parts:
        return V2FeatureBuilder(**builder_kwargs).transform(frame)

    out = pd.concat(parts, ignore_index=True, sort=False)
    out = _recompute_global_cross_sectional_features(out)
    return out.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
