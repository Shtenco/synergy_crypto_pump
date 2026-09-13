from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class RetrainingPolicy:
    min_new_bars: int = 20_000
    max_model_age_days: int = 7
    max_reference_psi: float = 0.20
    min_history_days: int = 30
    min_symbols: int = 1_000


@dataclass(frozen=True, slots=True)
class RetrainingDecision:
    should_retrain: bool
    reasons: tuple[str, ...]
    history_days: float
    symbols: int
    rows: int
    new_rows: int
    model_age_days: float | None


def decide_retraining(
    frame: pd.DataFrame,
    *,
    previous_manifest: dict | None = None,
    latest_health: dict | None = None,
    now: pd.Timestamp | None = None,
    policy: RetrainingPolicy | None = None,
) -> RetrainingDecision:
    p = policy or RetrainingPolicy()
    if frame.empty:
        return RetrainingDecision(False, ("empty_dataset",), 0.0, 0, 0, 0, None)
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    history_days = max(0.0, (ts.max() - ts.min()).total_seconds() / 86400.0)
    symbols = int(frame["symbol"].nunique())
    rows = int(len(frame))
    prev_rows = int((previous_manifest or {}).get("rows", 0) or 0)
    new_rows = max(0, rows - prev_rows)
    now = now or pd.Timestamp.now(tz="UTC")
    model_age = None
    trained_at = (previous_manifest or {}).get("trained_at")
    if trained_at:
        age = now - pd.Timestamp(trained_at)
        model_age = max(0.0, age.total_seconds() / 86400.0)

    reasons: list[str] = []
    if history_days < p.min_history_days:
        reasons.append("insufficient_history_days")
    if symbols < p.min_symbols:
        reasons.append("insufficient_symbol_breadth")

    trigger = False
    if previous_manifest is None:
        trigger = True
        reasons.append("no_previous_model")
    if new_rows >= p.min_new_bars:
        trigger = True
        reasons.append("enough_new_bars")
    if model_age is not None and model_age >= p.max_model_age_days:
        trigger = True
        reasons.append("model_age_limit")
    max_psi = (latest_health or {}).get("max_psi")
    if max_psi is not None and float(max_psi) >= p.max_reference_psi:
        trigger = True
        reasons.append("feature_drift")

    readiness_blocked = history_days < p.min_history_days or symbols < p.min_symbols
    return RetrainingDecision(
        should_retrain=bool(trigger and not readiness_blocked),
        reasons=tuple(reasons),
        history_days=history_days,
        symbols=symbols,
        rows=rows,
        new_rows=new_rows,
        model_age_days=model_age,
    )


def write_model_manifest(
    path: str | Path,
    *,
    dataset_path: str | Path,
    rows: int,
    symbols: int,
    first_timestamp,
    last_timestamp,
    model_path: str | Path,
    git_sha: str | None = None,
    extra: dict | None = None,
) -> dict:
    payload = {
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "dataset_path": str(dataset_path),
        "rows": int(rows),
        "symbols": int(symbols),
        "first_timestamp": str(first_timestamp),
        "last_timestamp": str(last_timestamp),
        "model_path": str(model_path),
        "git_sha": git_sha,
        **(extra or {}),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return payload
