from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import numeric_feature_columns

META_EXCLUDE = {"open", "high", "low", "close", "volume"}


@dataclass
class PumpModel:
    pipeline: object
    features: list[str]
    kind: str

    @classmethod
    def fit(cls, frame: pd.DataFrame, kind: str = "auto", random_state: int = 42) -> "PumpModel":
        train = frame.loc[frame["target"].notna()].copy()
        y = train["target"].astype(int).to_numpy()
        features = numeric_feature_columns(train, exclude=META_EXCLUDE)
        if not features:
            raise ValueError("no numeric feature columns found")

        use_catboost = kind in {"auto", "catboost"}
        if use_catboost:
            try:
                from catboost import CatBoostClassifier
                model = CatBoostClassifier(
                    iterations=500,
                    depth=7,
                    learning_rate=0.05,
                    loss_function="Logloss",
                    eval_metric="PRAUC",
                    auto_class_weights="Balanced",
                    random_seed=random_state,
                    verbose=False,
                )
                model.fit(train[features], y)
                return cls(model, features, "catboost")
            except ImportError:
                if kind == "catboost":
                    raise

        estimator = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=350,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=random_state,
        )
        prep = ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features)])
        pipe = Pipeline([("prep", prep), ("model", estimator)])
        positives = max(1, int(y.sum()))
        negatives = max(1, int((1 - y).sum()))
        weights = np.where(y == 1, negatives / positives, 1.0)
        pipe.fit(train[features], y, model__sample_weight=weights)
        return cls(pipe, features, "hist_gradient_boosting")

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(frame[self.features])[:, 1]

    def rank(self, frame: pd.DataFrame, top_k: int | None = None) -> pd.DataFrame:
        out = frame.copy()
        out["pump_probability"] = self.predict_proba(out)
        out["rank"] = out.groupby("timestamp")["pump_probability"].rank(method="first", ascending=False).astype(int)
        if top_k is not None:
            out = out[out["rank"] <= top_k]
        return out.sort_values(["timestamp", "rank"])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "PumpModel":
        return joblib.load(path)


def chronological_split(frame: pd.DataFrame, train_fraction: float = 0.70, validation_fraction: float = 0.15):
    times = np.array(sorted(pd.to_datetime(frame["timestamp"], utc=True).unique()))
    n = len(times)
    a = max(1, int(n * train_fraction))
    b = max(a + 1, int(n * (train_fraction + validation_fraction)))
    train_t, val_t = set(times[:a]), set(times[a:b])
    train = frame[pd.to_datetime(frame["timestamp"], utc=True).isin(train_t)]
    val = frame[pd.to_datetime(frame["timestamp"], utc=True).isin(val_t)]
    test = frame[~pd.to_datetime(frame["timestamp"], utc=True).isin(train_t | val_t)]
    return train, val, test


def evaluate(model: PumpModel, frame: pd.DataFrame, top_k: int = 10) -> dict[str, float]:
    df = frame.loc[frame["target"].notna()].copy()
    y = df["target"].astype(int).to_numpy()
    p = model.predict_proba(df)
    metrics = {
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }
    if len(np.unique(y)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y, p))
    df["p"] = p
    top = df.sort_values(["timestamp", "p"], ascending=[True, False]).groupby("timestamp", sort=False).head(top_k)
    metrics[f"precision_at_{top_k}"] = float(top["target"].astype(float).mean()) if len(top) else float("nan")
    return metrics
