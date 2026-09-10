from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .config import DEFAULT_HORIZONS, HorizonSpec
from .features import social_feature_columns, v2_numeric_feature_columns


class _ConstantModel:
    def __init__(self, value: float):
        self.value = float(value)

    def predict_proba(self, x):
        p = np.full(len(x), self.value)
        return np.column_stack([1.0 - p, p])

    def predict(self, x):
        return np.full(len(x), self.value)


def _fit_binary(x: pd.DataFrame, y: np.ndarray, kind: str, seed: int):
    if len(np.unique(y)) < 2:
        return _ConstantModel(float(np.mean(y)))
    if kind in {"auto", "catboost"}:
        try:
            from catboost import CatBoostClassifier
            m = CatBoostClassifier(
                iterations=500, depth=7, learning_rate=0.04,
                loss_function="Logloss", eval_metric="PRAUC",
                auto_class_weights="Balanced", random_seed=seed, verbose=False,
            )
            m.fit(x, y)
            return m
        except ImportError:
            if kind == "catboost":
                raise
    m = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=350, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=seed,
        )),
    ])
    positives = max(1, int(y.sum()))
    negatives = max(1, int((1 - y).sum()))
    weights = np.where(y == 1, negatives / positives, 1.0)
    m.fit(x, y, model__sample_weight=weights)
    return m


@dataclass
class MultiHorizonTabularBranch:
    kind: str = "auto"
    random_state: int = 42
    features: list[str] = field(default_factory=list)
    models: dict[str, object] = field(default_factory=dict)
    prefix: str = "tabular"

    def fit(
        self,
        frame: pd.DataFrame,
        specs: Sequence[HorizonSpec] = DEFAULT_HORIZONS,
        feature_subset: list[str] | None = None,
    ) -> "MultiHorizonTabularBranch":
        self.features = feature_subset or v2_numeric_feature_columns(frame)
        if not self.features:
            raise ValueError(f"{self.prefix}: no numeric features")
        for spec in specs:
            use = frame[spec.target_col].notna()
            y = frame.loc[use, spec.target_col].astype(int).to_numpy()
            self.models[spec.target_col] = _fit_binary(
                frame.loc[use, self.features], y, self.kind, self.random_state
            )
        return self

    def predict(
        self, frame: pd.DataFrame, specs: Sequence[HorizonSpec] = DEFAULT_HORIZONS
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        for spec in specs:
            m = self.models[spec.target_col]
            out[f"{self.prefix}__{spec.target_col}"] = m.predict_proba(frame[self.features])[:, 1]
        return out


@dataclass
class SocialBranch(MultiHorizonTabularBranch):
    prefix: str = "social"

    def fit(self, frame: pd.DataFrame, specs: Sequence[HorizonSpec] = DEFAULT_HORIZONS, feature_subset=None):
        cols = social_feature_columns(frame)
        if not cols:
            self.features = []
            self.models = {}
            return self
        return super().fit(frame, specs, cols)

    def predict(self, frame: pd.DataFrame, specs: Sequence[HorizonSpec] = DEFAULT_HORIZONS):
        if not self.features:
            return pd.DataFrame(index=frame.index)
        return super().predict(frame, specs)


@dataclass
class CatBoostRankerBranch:
    kind: str = "auto"
    random_state: int = 42
    features: list[str] = field(default_factory=list)
    models: dict[str, object] = field(default_factory=dict)
    model_kinds: dict[str, str] = field(default_factory=dict)

    def fit(
        self,
        frame: pd.DataFrame,
        specs: Sequence[HorizonSpec] = DEFAULT_HORIZONS,
        feature_subset: list[str] | None = None,
    ) -> "CatBoostRankerBranch":
        self.features = feature_subset or v2_numeric_feature_columns(frame)
        for spec in specs:
            use = frame[spec.mfe_col].notna()
            df = frame.loc[use].sort_values(["timestamp", "symbol"], kind="mergesort").copy()
            if len(df) == 0:
                raise ValueError(f"no rows for {spec.name}")
            relevance = df.groupby("timestamp")[spec.mfe_col].rank(pct=True, ascending=True).to_numpy(float)
            if self.kind in {"auto", "catboost"}:
                try:
                    from catboost import CatBoostRanker, Pool
                    group_id = pd.factorize(df["timestamp"], sort=False)[0]
                    model = CatBoostRanker(
                        loss_function="YetiRank:mode=NDCG",
                        eval_metric="NDCG:top=10",
                        iterations=450,
                        depth=5,
                        learning_rate=0.04,
                        border_count=255,
                        random_seed=self.random_state,
                        verbose=False,
                    )
                    model.fit(Pool(df[self.features], label=relevance, group_id=group_id))
                    self.models[spec.target_col] = model
                    self.model_kinds[spec.target_col] = "catboost_yetirank"
                    continue
                except ImportError:
                    if self.kind == "catboost":
                        raise
            reg = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingRegressor(
                    learning_rate=0.05, max_iter=350, max_leaf_nodes=31,
                    l2_regularization=1.0, random_state=self.random_state,
                )),
            ])
            reg.fit(df[self.features], relevance)
            self.models[spec.target_col] = reg
            self.model_kinds[spec.target_col] = "hist_gradient_rank_fallback"
        return self

    def predict(
        self, frame: pd.DataFrame, specs: Sequence[HorizonSpec] = DEFAULT_HORIZONS
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        for spec in specs:
            raw = np.asarray(self.models[spec.target_col].predict(frame[self.features]), dtype=float)
            tmp = pd.DataFrame({"timestamp": frame["timestamp"].to_numpy(), "raw": raw}, index=frame.index)
            out[f"ranker__{spec.target_col}"] = tmp.groupby("timestamp")["raw"].rank(pct=True)
        return out
