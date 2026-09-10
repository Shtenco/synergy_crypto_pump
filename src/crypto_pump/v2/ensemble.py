from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import DEFAULT_HORIZONS, HorizonSpec


class _ConstantStacker:
    def __init__(self, p: float):
        self.p = float(p)

    def predict_proba(self, x):
        p = np.full(len(x), self.p)
        return np.column_stack([1 - p, p])


@dataclass
class CalibratedStacker:
    models: dict[str, object] = field(default_factory=dict)
    isotonic: dict[str, IsotonicRegression | None] = field(default_factory=dict)
    columns: dict[str, list[str]] = field(default_factory=dict)

    def fit(
        self,
        scores: pd.DataFrame,
        labels: pd.DataFrame,
        specs: Sequence[HorizonSpec] = DEFAULT_HORIZONS,
    ) -> "CalibratedStacker":
        for spec in specs:
            cols = [
                c for c in scores.columns
                if c.endswith(f"__{spec.target_col}") or c == "bocd_cp_prob"
            ]
            if not cols:
                raise ValueError(f"no branch scores for {spec.target_col}")

            valid_idx = labels.index[labels[spec.target_col].notna()].to_numpy()
            if "timestamp" in labels:
                sort_cols = ["timestamp", "symbol"] if "symbol" in labels else ["timestamp"]
                valid_idx = labels.loc[valid_idx].sort_values(sort_cols).index.to_numpy()
            self.columns[spec.target_col] = cols
            y_all = labels.loc[valid_idx, spec.target_col].astype(int).to_numpy()

            split = int(len(valid_idx) * 0.70)
            can_split = (
                split >= 10
                and len(valid_idx) - split >= 10
                and len(np.unique(y_all[:split])) > 1
            )
            fit_idx = valid_idx[:split] if can_split else valid_idx
            cal_idx = valid_idx[split:] if can_split else np.array([], dtype=valid_idx.dtype)
            x_fit = scores.loc[fit_idx, cols]
            y_fit = labels.loc[fit_idx, spec.target_col].astype(int).to_numpy()

            if len(np.unique(y_fit)) < 2:
                model = _ConstantStacker(float(np.mean(y_fit)))
            else:
                model = Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("logit", LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                    )),
                ])
                model.fit(x_fit, y_fit)
            self.models[spec.target_col] = model

            iso = None
            if len(cal_idx):
                y_cal = labels.loc[cal_idx, spec.target_col].astype(int).to_numpy()
                if (
                    len(np.unique(y_cal)) > 1
                    and y_cal.sum() >= 5
                    and (len(y_cal) - y_cal.sum()) >= 5
                ):
                    raw_cal = model.predict_proba(scores.loc[cal_idx, cols])[:, 1]
                    iso = IsotonicRegression(out_of_bounds="clip")
                    iso.fit(raw_cal, y_cal)
            self.isotonic[spec.target_col] = iso
        return self

    def predict(
        self,
        scores: pd.DataFrame,
        specs: Sequence[HorizonSpec] = DEFAULT_HORIZONS,
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=scores.index)
        for spec in specs:
            cols = self.columns[spec.target_col]
            raw = self.models[spec.target_col].predict_proba(scores[cols])[:, 1]
            iso = self.isotonic.get(spec.target_col)
            out[f"pump_probability__{spec.name}"] = iso.predict(raw) if iso is not None else raw
        return out
