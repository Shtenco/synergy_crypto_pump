from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import DEFAULT_HORIZONS, HorizonSpec


def _safe_pr_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, p))


def _safe_roc_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def _basic_metrics(df: pd.DataFrame, target_col: str, score_col: str, top_k: int) -> dict[str, float | int | None]:
    use = df[target_col].notna() & df[score_col].notna()
    x = df.loc[use, ["timestamp", "symbol", target_col, score_col]].copy()
    if x.empty:
        return {"rows": 0, "positive_rate": None, "pr_auc": None, "roc_auc": None, f"precision_at_{top_k}": None}
    y = x[target_col].astype(np.int8).to_numpy()
    p = x[score_col].astype(float).to_numpy()
    top = (
        x.sort_values(["timestamp", score_col], ascending=[True, False], kind="mergesort")
        .groupby("timestamp", sort=False)
        .head(top_k)
    )
    precision = float(top[target_col].astype(float).mean()) if len(top) else None
    base = float(y.mean())
    return {
        "rows": int(len(x)),
        "positive_rate": base,
        "pr_auc": _safe_pr_auc(y, p),
        "roc_auc": _safe_roc_auc(y, p),
        f"precision_at_{top_k}": precision,
        f"lift_at_{top_k}": (float(precision / base) if precision is not None and base > 0 else None),
    }


def _time_slices(df: pd.DataFrame, target_col: str, score_col: str, top_k: int, n_slices: int = 4) -> list[dict]:
    times = np.array(sorted(pd.to_datetime(df["timestamp"], utc=True).dropna().unique()))
    if len(times) == 0:
        return []
    chunks = [c for c in np.array_split(times, min(n_slices, len(times))) if len(c)]
    out: list[dict] = []
    ts = pd.to_datetime(df["timestamp"], utc=True)
    for i, chunk in enumerate(chunks):
        m = ts.isin(set(chunk))
        metrics = _basic_metrics(df.loc[m], target_col, score_col, top_k)
        metrics.update({"slice": i, "start": str(chunk[0]), "end": str(chunk[-1])})
        out.append(metrics)
    return out


def _symbol_concentration(df: pd.DataFrame, target_col: str, score_col: str, top_k: int) -> dict:
    use = df[target_col].notna() & df[score_col].notna()
    x = df.loc[use, ["timestamp", "symbol", target_col, score_col]].copy()
    if x.empty:
        return {}
    top = (
        x.sort_values(["timestamp", score_col], ascending=[True, False], kind="mergesort")
        .groupby("timestamp", sort=False)
        .head(top_k)
    )
    tp = top[top[target_col].astype(bool)]
    tp_counts = tp.groupby("symbol").size().sort_values(ascending=False)
    tp_total = int(tp_counts.sum())
    positives_by_symbol = x.groupby("symbol")[target_col].sum().sort_values(ascending=False)
    result = {
        "top_true_positive_symbols": {str(k): int(v) for k, v in tp_counts.head(10).items()},
        "top1_share_of_topk_true_positives": float(tp_counts.head(1).sum() / tp_total) if tp_total else None,
        "top5_share_of_topk_true_positives": float(tp_counts.head(5).sum() / tp_total) if tp_total else None,
        "top10_share_of_topk_true_positives": float(tp_counts.head(10).sum() / tp_total) if tp_total else None,
        "leave_top_positive_symbols_out": {},
    }
    for n in (1, 3, 5, 10):
        excluded = set(positives_by_symbol.head(n).index)
        kept = x[~x["symbol"].isin(excluded)]
        result["leave_top_positive_symbols_out"][str(n)] = {
            "excluded": [str(s) for s in positives_by_symbol.head(n).index],
            **_basic_metrics(kept, target_col, score_col, top_k),
        }
    return result


def _decluster(df: pd.DataFrame, target_col: str, score_col: str, top_k: int) -> dict:
    use = df[target_col].notna() & df[score_col].notna()
    x = df.loc[use, ["timestamp", "symbol", target_col, score_col]].copy()
    if x.empty:
        return {}
    x = x.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
    pos = x[target_col].astype(bool)
    prev = pos.groupby(x["symbol"], sort=False).shift(1, fill_value=False)
    starts = pos & ~prev
    local_cluster = starts.groupby(x["symbol"], sort=False).cumsum()
    x["event_id"] = np.where(pos, x["symbol"].astype(str) + "#" + local_cluster.astype(str), None)
    events = int(pd.Series(x.loc[pos, "event_id"]).nunique())
    top = (
        x.sort_values(["timestamp", score_col], ascending=[True, False], kind="mergesort")
        .groupby("timestamp", sort=False)
        .head(top_k)
    )
    hit_ids = top.loc[top[target_col].astype(bool), "event_id"].dropna()
    unique_hits = int(hit_ids.nunique())
    alerts = int(len(top))
    return {
        "positive_rows": int(pos.sum()),
        "event_clusters": events,
        "topk_alert_rows": alerts,
        "topk_true_positive_rows": int(top[target_col].sum()),
        "unique_event_clusters_hit": unique_hits,
        "event_recall_at_topk": float(unique_hits / events) if events else None,
        "unique_events_per_1000_alerts": float(unique_hits * 1000.0 / alerts) if alerts else None,
        "mean_positive_rows_per_event": float(pos.sum() / events) if events else None,
    }


def _shuffle_by_groups(x: pd.DataFrame, target_col: str, group_col: str, rng: np.random.Generator) -> pd.Series:
    shuffled = x[target_col].copy()
    for _, idx in x.groupby(group_col, sort=False).groups.items():
        vals = shuffled.loc[idx].to_numpy(copy=True)
        rng.shuffle(vals)
        shuffled.loc[idx] = vals
    return shuffled


def _placebos(df: pd.DataFrame, target_col: str, score_col: str, top_k: int, shift_bars: int, seed: int = 42) -> dict:
    use = df[target_col].notna() & df[score_col].notna()
    x = df.loc[use, ["timestamp", "symbol", target_col, score_col]].copy()
    x = x.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
    if x.empty:
        return {}

    # 1) Global permutation: destroys symbol, time and cross-sectional structure.
    rng = np.random.default_rng(seed)
    global_x = x.copy()
    global_values = global_x[target_col].to_numpy(copy=True)
    rng.shuffle(global_values)
    global_x["_placebo"] = global_values

    # 2) Within-symbol permutation: preserves each symbol's unconditional pump
    # rate but destroys timing. If this remains strong, symbol memorization or
    # symbol-specific feature distributions are likely carrying the score.
    symbol_x = x.copy()
    symbol_x["_placebo"] = _shuffle_by_groups(x, target_col, "symbol", np.random.default_rng(seed + 1)).astype(float)

    # 3) Within-timestamp permutation: preserves exactly how many positives
    # occur at each timestamp but destroys which symbol pumps. This is a direct
    # null for cross-sectional ranking skill.
    time_x = x.copy()
    time_x["_placebo"] = _shuffle_by_groups(x, target_col, "timestamp", np.random.default_rng(seed + 2)).astype(float)

    # 4) Large temporal shift inside symbol: preserves target autocorrelation
    # shape/base rate while breaking alignment with the contemporaneous score.
    shift_x = x.copy()
    shift_x["_placebo"] = x.groupby("symbol", sort=False)[target_col].shift(shift_bars)

    return {
        "global_random_permutation": _basic_metrics(global_x, "_placebo", score_col, top_k),
        "within_symbol_random_permutation": _basic_metrics(symbol_x, "_placebo", score_col, top_k),
        "within_timestamp_random_permutation": _basic_metrics(time_x, "_placebo", score_col, top_k),
        f"within_symbol_shift_{shift_bars}_bars": _basic_metrics(shift_x, "_placebo", score_col, top_k),
        "interpretation": {
            "global": "should collapse to base rate if the metric implementation is sane",
            "within_symbol": "tests timing skill beyond persistent symbol identity/base-rate differences",
            "within_timestamp": "tests which-symbol ranking skill while preserving market-wide event intensity",
            "shift": "tests long-lag alignment inside each symbol",
        },
    }


def _pre_pump_filter(df: pd.DataFrame, target_col: str, score_col: str, top_k: int, threshold: float) -> dict:
    candidates = [c for c in ("current_candle_return", "ret_1", "ret_3", "ret_6", "ret_12") if c in df.columns]
    if not candidates:
        return {"available": False, "reason": "no recent-return columns found"}
    vals = df[candidates].apply(pd.to_numeric, errors="coerce")
    recent_max = vals.max(axis=1, skipna=True)
    keep = recent_max.lt(threshold) | recent_max.isna()
    return {
        "available": True,
        "threshold": float(threshold),
        "columns": candidates,
        "rows_before": int(len(df)),
        "rows_after": int(keep.sum()),
        "retained_fraction": float(keep.mean()) if len(keep) else None,
        "metrics": _basic_metrics(df.loc[keep], target_col, score_col, top_k),
    }


def run_falsification(
    frame: pd.DataFrame,
    specs: Sequence[HorizonSpec] = DEFAULT_HORIZONS,
    top_k: int = 10,
    pre_pump_threshold: float = 0.05,
    placebo_shift_bars: int = 2016,
) -> dict:
    """Stress-test saved OOS/holdout scores without retraining the predictor."""
    d = frame.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    report: dict[str, object] = {
        "rows": int(len(d)),
        "symbols": int(d["symbol"].nunique()) if "symbol" in d else 0,
        "timestamps": int(d["timestamp"].nunique()),
        "top_k": int(top_k),
        "pre_pump_threshold": float(pre_pump_threshold),
        "placebo_shift_bars": int(placebo_shift_bars),
        "targets": {},
    }
    for spec in specs:
        score_col = f"p__{spec.name}"
        if score_col not in d.columns or spec.target_col not in d.columns:
            continue
        report["targets"][spec.name] = {
            "raw": _basic_metrics(d, spec.target_col, score_col, top_k),
            "time_slices": _time_slices(d, spec.target_col, score_col, top_k),
            "symbol_concentration": _symbol_concentration(d, spec.target_col, score_col, top_k),
            "decluster": _decluster(d, spec.target_col, score_col, top_k),
            "placebos": _placebos(d, spec.target_col, score_col, top_k, placebo_shift_bars),
            "pre_pump_filter": _pre_pump_filter(d, spec.target_col, score_col, top_k, pre_pump_threshold),
        }
    return report
