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
        return np.column_stack([1-p, p])


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
            cols = [c for c in scores.columns if c.endswith(f"__{spec.target_col}") or c == "bocd_cp_prob"]
            if not cols:
                raise ValueError(f"no branch scores for {spec.target_col}")
            use = labels[spec.target_col].notna()
            x = scores.loc[use, cols]
            y = labels.loc[use, spec.target_col].astype(int).to_numpy()
            self.columns[spec.target_col] = cols
            if len(np.unique(y)) < 2:
                model = _ConstantStacker(float(np.mean(y)))
            else:
                model = Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("logit", LogisticRegression(
                        C=0.5, class_weight="balanced", max_iter=1000, random_state=42
                    )),
                ])
                model.fit(x, y)
            self.models[spec.target_col] = model
            raw = model.predict_proba(x)[:, 1]
            iso = None
            if len(np.unique(y)) > 1 and y.sum() >= 10 and (len(y) - y.sum()) >= 10:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(raw, y)
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
