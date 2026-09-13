from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .metrics import financial_report, stress_test_trades


def normalize_oos_trades(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize one OOS trade ledger before pooling it with other OOS ledgers.

    The function never changes PnL/return values. It only adds provenance and a
    common executable flag/timestamp so old and new OOS ledgers can be audited
    after concatenation.
    """
    out = frame.copy()
    out["oos_source"] = str(source)
    if "executable" not in out.columns:
        out["executable"] = True

    if "timestamp" not in out.columns:
        for candidate in ("entry_timestamp", "signal_timestamp", "entry_time", "signal_time"):
            if candidate in out.columns:
                out["timestamp"] = out[candidate]
                break

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    return out


def combine_oos_trades(sources: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Pool multiple historical OOS trade ledgers into one auditable ledger."""
    frames = [normalize_oos_trades(frame, source) for source, frame in sources.items() if not frame.empty]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "timestamp" in combined.columns:
        combined = combined.sort_values(["timestamp", "oos_source"], kind="stable", na_position="last")
    combined = combined.reset_index(drop=True)
    combined["combined_oos_seq"] = range(len(combined))
    return combined


def normalize_oos_predictions(
    frame: pd.DataFrame,
    source: str,
    *,
    target_definition: str | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    out["oos_source"] = str(source)
    if target_definition is not None:
        out["target_definition"] = str(target_definition)
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    return out


def combine_oos_predictions(
    sources: Mapping[str, pd.DataFrame],
    *,
    target_definitions: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    definitions = target_definitions or {}
    for source, frame in sources.items():
        if frame.empty:
            continue
        frames.append(
            normalize_oos_predictions(
                frame,
                source,
                target_definition=definitions.get(source),
            )
        )
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "timestamp" in combined.columns:
        combined = combined.sort_values(["timestamp", "oos_source"], kind="stable", na_position="last")
    return combined.reset_index(drop=True)


def _binary_arrays(
    frame: pd.DataFrame,
    target_col: str,
    score_col: str,
) -> tuple[pd.Series, pd.Series]:
    y = pd.to_numeric(frame[target_col], errors="coerce")
    score = pd.to_numeric(frame[score_col], errors="coerce")
    valid = y.notna() & score.notna()
    y = (y[valid] > 0).astype(int)
    score = score[valid].astype(float)
    return y, score


def average_precision(y: pd.Series, score: pd.Series) -> float | None:
    """Binary average precision, used as the pooled PR-AUC/AP statistic."""
    if len(y) == 0:
        return None
    order = score.sort_values(ascending=False, kind="stable").index
    ranked_y = y.loc[order].reset_index(drop=True)
    positives = int(ranked_y.sum())
    if positives == 0:
        return None
    cumulative = ranked_y.cumsum()
    ranks = pd.Series(range(1, len(ranked_y) + 1), index=ranked_y.index, dtype=float)
    precision = cumulative / ranks
    return float(precision[ranked_y == 1].sum() / positives)


def precision_at_k(y: pd.Series, score: pd.Series, k: int) -> float | None:
    if len(y) == 0 or k <= 0:
        return None
    k = min(int(k), len(y))
    order = score.sort_values(ascending=False, kind="stable").index[:k]
    return float(y.loc[order].mean()) if k else None


def pooled_ml_report(
    predictions: pd.DataFrame,
    *,
    target_col: str = "y_true",
    score_col: str = "probability",
    precision_ks: tuple[int, ...] = (10, 50, 100, 500),
    require_common_target: bool = True,
) -> dict[str, Any]:
    if predictions.empty:
        return {"samples": 0, "pr_auc": None, "precision_at_k": {}}
    missing = [c for c in (target_col, score_col) if c not in predictions.columns]
    if missing:
        return {
            "samples": int(len(predictions)),
            "pr_auc": None,
            "precision_at_k": {},
            "comparable": False,
            "reason": f"missing columns: {', '.join(missing)}",
        }

    target_defs: list[str] = []
    if "target_definition" in predictions.columns:
        target_defs = sorted(str(x) for x in predictions["target_definition"].dropna().unique())
    if require_common_target and len(target_defs) > 1:
        return {
            "samples": int(len(predictions)),
            "pr_auc": None,
            "precision_at_k": {},
            "comparable": False,
            "target_definitions": target_defs,
            "reason": "PR-AUC/Precision@K cannot be pooled across different target definitions",
        }

    y, score = _binary_arrays(predictions, target_col, score_col)
    return {
        "samples": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else None,
        "pr_auc": average_precision(y, score),
        "precision_at_k": {str(k): precision_at_k(y, score, k) for k in precision_ks},
        "comparable": True,
        "target_definitions": target_defs,
    }


def combined_oos_report(
    trades: pd.DataFrame,
    *,
    predictions: pd.DataFrame | None = None,
    source_metric_snapshots: Mapping[str, Mapping[str, Any]] | None = None,
    dataset_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    target_col: str = "y_true",
    score_col: str = "probability",
) -> dict[str, Any]:
    """Create one canonical OOS report from all historical OOS sources.

    Financial statistics are always recomputed from the pooled chronological
    trade ledger; they are never averaged from summary metrics. Source blocks
    remain only for provenance/audit.
    """
    sources = sorted(str(x) for x in trades.get("oos_source", pd.Series(dtype=str)).dropna().unique())
    by_source: dict[str, Any] = {}
    if "oos_source" in trades.columns:
        for source, part in trades.groupby("oos_source", sort=True):
            by_source[str(source)] = financial_report(part)

    report: dict[str, Any] = {
        "report_name": "COMBINED_OOS",
        "sources": sources,
        "financial": financial_report(trades),
        "winner_robustness": stress_test_trades(trades),
        "by_source": by_source,
        "dataset_metadata": dict(dataset_metadata or {}),
        "source_metric_snapshots": {
            str(k): dict(v) for k, v in (source_metric_snapshots or {}).items()
        },
    }
    if predictions is not None:
        report["ml"] = pooled_ml_report(
            predictions,
            target_col=target_col,
            score_col=score_col,
        )
    return report
