from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class HealthThresholds:
    max_data_age_seconds: float = 180.0
    max_missing_fraction: float = 0.25
    max_feature_psi: float = 0.30
    min_live_rows: int = 50


@dataclass(frozen=True, slots=True)
class HealthReport:
    healthy: bool
    data_age_seconds: float
    missing_fraction: float
    max_psi: float
    missing_features: tuple[str, ...]
    reasons: tuple[str, ...]


def feature_parity(frame: pd.DataFrame, expected: Iterable[str]) -> tuple[str, ...]:
    expected_set = {str(c) for c in expected}
    return tuple(sorted(expected_set.difference(frame.columns)))


def population_stability_index(
    reference: pd.Series,
    live: pd.Series,
    *,
    bins: int = 10,
) -> float:
    r = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(float)
    l = pd.to_numeric(live, errors="coerce").dropna().to_numpy(float)
    if len(r) < 20 or len(l) < 20:
        return float("nan")
    edges = np.unique(np.quantile(r, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r_count, _ = np.histogram(r, bins=edges)
    l_count, _ = np.histogram(l, bins=edges)
    eps = 1e-6
    rp = np.clip(r_count / max(1, r_count.sum()), eps, None)
    lp = np.clip(l_count / max(1, l_count.sum()), eps, None)
    return float(np.sum((lp - rp) * np.log(lp / rp)))


def health_report(
    live_frame: pd.DataFrame,
    *,
    expected_features: Iterable[str],
    reference_frame: pd.DataFrame | None = None,
    timestamp_col: str = "timestamp",
    now: pd.Timestamp | None = None,
    thresholds: HealthThresholds | None = None,
) -> HealthReport:
    th = thresholds or HealthThresholds()
    reasons: list[str] = []
    missing = feature_parity(live_frame, expected_features)
    if missing:
        reasons.append("feature_parity")

    if live_frame.empty or timestamp_col not in live_frame:
        age = float("inf")
    else:
        now_ts = now or pd.Timestamp.now(tz="UTC")
        latest = pd.to_datetime(live_frame[timestamp_col], utc=True, errors="coerce").max()
        age = float("inf") if pd.isna(latest) else max(0.0, (now_ts - latest).total_seconds())
    if age > th.max_data_age_seconds:
        reasons.append("stale_data")

    present = [c for c in expected_features if c in live_frame.columns]
    if present and len(live_frame):
        missing_fraction = float(live_frame[present].isna().mean().mean())
    else:
        missing_fraction = 1.0
    if missing_fraction > th.max_missing_fraction:
        reasons.append("missingness")

    max_psi = 0.0
    if reference_frame is not None and len(live_frame) >= th.min_live_rows:
        psis: list[float] = []
        for col in present:
            if col not in reference_frame.columns:
                continue
            if not pd.api.types.is_numeric_dtype(live_frame[col]):
                continue
            psi = population_stability_index(reference_frame[col], live_frame[col])
            if np.isfinite(psi):
                psis.append(psi)
        max_psi = max(psis) if psis else 0.0
        if max_psi > th.max_feature_psi:
            reasons.append("feature_drift")

    return HealthReport(
        healthy=not reasons,
        data_age_seconds=age,
        missing_fraction=missing_fraction,
        max_psi=max_psi,
        missing_features=missing,
        reasons=tuple(reasons),
    )
