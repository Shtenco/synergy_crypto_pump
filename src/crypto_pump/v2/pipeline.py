from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    ndcg_score,
    roc_auc_score,
)

from .branches import CatBoostRankerBranch, MultiHorizonTabularBranch, SocialBranch
from .config import DEFAULT_HORIZONS, HorizonSpec
from .ensemble import CalibratedStacker
from .graph import WalletGNNBranch
from .splits import purged_walk_forward_folds
from .temporal import TemporalTransformerBranch


@dataclass
class V2Ensemble:
    specs: Sequence[HorizonSpec] = DEFAULT_HORIZONS
    model_kind: str = "auto"
    enable_temporal: bool = False
    enable_gnn: bool = False
    sequence_length: int = 48
    tabular: MultiHorizonTabularBranch | None = None
    ranker: CatBoostRankerBranch | None = None
    social: SocialBranch | None = None
    temporal: TemporalTransformerBranch | None = None
    gnn: WalletGNNBranch | None = None
    stacker: CalibratedStacker | None = None

    def _base_scores(self, frame: pd.DataFrame, transactions: pd.DataFrame | None = None) -> pd.DataFrame:
        pieces = [
            self.tabular.predict(frame, self.specs),
            self.ranker.predict(frame, self.specs),
        ]
        if self.social is not None and self.social.features:
            pieces.append(self.social.predict(frame, self.specs))
        if self.temporal is not None:
            pieces.append(self.temporal.predict(frame, self.specs))
        if self.gnn is not None:
            if transactions is None:
                raise ValueError("transactions are required for a fitted GNN branch")
            pieces.append(self.gnn.predict(frame, transactions, self.specs))
        scores = pd.concat(pieces, axis=1)
        if "bocd_cp_prob" in frame:
            scores["bocd_cp_prob"] = frame["bocd_cp_prob"].astype(float)
        return scores

    def fit(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        transactions: pd.DataFrame | None = None,
        temporal_epochs: int = 8,
        gnn_epochs: int = 4,
    ) -> "V2Ensemble":
        self.tabular = MultiHorizonTabularBranch(kind=self.model_kind).fit(train, self.specs)
        self.ranker = CatBoostRankerBranch(kind=self.model_kind).fit(train, self.specs)
        self.social = SocialBranch(kind=self.model_kind).fit(train, self.specs)
        if self.enable_temporal:
            self.temporal = TemporalTransformerBranch(
                seq_len=self.sequence_length, epochs=temporal_epochs
            ).fit(train, self.specs)
        if self.enable_gnn:
            if transactions is None:
                raise ValueError("enable_gnn=True requires a transactions table")
            self.gnn = WalletGNNBranch(epochs=gnn_epochs).fit(train, transactions, self.specs)

        val_scores = self._base_scores(validation, transactions)
        self.stacker = CalibratedStacker().fit(val_scores, validation, self.specs)
        return self

    def predict(
        self, frame: pd.DataFrame, transactions: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        if self.stacker is None:
            raise RuntimeError("V2Ensemble is not fitted")
        base = self._base_scores(frame, transactions)
        final = self.stacker.predict(base, self.specs)
        return pd.concat([base, final], axis=1)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @staticmethod
    def load(path: str | Path) -> "V2Ensemble":
        return joblib.load(path)


def evaluate_v2_predictions(
    labeled: pd.DataFrame,
    predictions: pd.DataFrame,
    specs: Sequence[HorizonSpec] = DEFAULT_HORIZONS,
    top_k: int = 10,
) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for spec in specs:
        prob_col = f"pump_probability__{spec.name}"
        use = labeled[spec.target_col].notna() & predictions[prob_col].notna()
        df = labeled.loc[use, ["timestamp", "symbol", spec.target_col, spec.mfe_col]].copy()
        if df.empty:
            report[spec.name] = {}
            continue
        p = predictions.loc[use, prob_col].to_numpy(float)
        y = df[spec.target_col].astype(int).to_numpy()
        metrics: dict[str, float] = {
            "rows": float(len(df)),
            "positive_rate": float(y.mean()),
            "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
        }
        if len(np.unique(y)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y, p))
        df["p"] = p
        top = (
            df.sort_values(["timestamp", "p"], ascending=[True, False])
            .groupby("timestamp", sort=False)
            .head(top_k)
        )
        metrics[f"precision_at_{top_k}"] = float(top[spec.target_col].astype(float).mean())

        ndcgs = []
        for _, g in df.groupby("timestamp", sort=False):
            if len(g) < 2:
                continue
            rel = np.maximum(g[spec.mfe_col].to_numpy(float), 0.0)
            score = g["p"].to_numpy(float)
            if np.isfinite(rel).all() and np.isfinite(score).all() and rel.max() > 0:
                ndcgs.append(float(ndcg_score([rel], [score], k=min(top_k, len(g)))))
        if ndcgs:
            metrics[f"ndcg_at_{top_k}"] = float(np.mean(ndcgs))
        report[spec.name] = metrics
    return report


def walk_forward_oos(
    frame: pd.DataFrame,
    output_dir: str | Path,
    specs: Sequence[HorizonSpec] = DEFAULT_HORIZONS,
    n_splits: int = 3,
    min_train_fraction: float = 0.45,
    validation_fraction: float = 0.10,
    purge_bars: int | None = None,
    embargo_bars: int = 12,
    model_kind: str = "auto",
    top_k: int = 10,
    enable_temporal: bool = False,
    enable_gnn: bool = False,
    transactions: pd.DataFrame | None = None,
    sequence_length: int = 48,
    temporal_epochs: int = 8,
    gnn_epochs: int = 4,
) -> tuple[pd.DataFrame, dict]:
    """True expanding-window OOS: every test prediction comes from an earlier-only fit."""
    if purge_bars is None:
        purge_bars = max(s.horizon_bars for s in specs)
    folds = purged_walk_forward_folds(
        frame,
        n_splits=n_splits,
        min_train_fraction=min_train_fraction,
        validation_fraction=validation_fraction,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )
    outputs: list[pd.DataFrame] = []
    fold_reports = []
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        train, val, test = fold.frames(frame)
        model = V2Ensemble(
            specs=specs,
            model_kind=model_kind,
            enable_temporal=enable_temporal,
            enable_gnn=enable_gnn,
            sequence_length=sequence_length,
        ).fit(
            train, val, transactions,
            temporal_epochs=temporal_epochs, gnn_epochs=gnn_epochs,
        )
        pred = model.predict(test, transactions)
        pred["fold"] = fold.fold
        pred["timestamp"] = test["timestamp"].to_numpy()
        pred["symbol"] = test["symbol"].to_numpy()
        for spec in specs:
            pred[spec.target_col] = test[spec.target_col].to_numpy()
            pred[spec.mfe_col] = test[spec.mfe_col].to_numpy()
            pred[spec.mae_col] = test[spec.mae_col].to_numpy()
        outputs.append(pred)
        fold_report = evaluate_v2_predictions(test, pred, specs, top_k=top_k)
        fold_reports.append({"fold": fold.fold, "metrics": fold_report})
        model.save(outdir / f"fold_{fold.fold}_model.joblib")

    oos = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    label_cols = ["timestamp", "symbol"] + [
        c for s in specs for c in (s.target_col, s.mfe_col, s.mae_col)
    ]
    aggregate = evaluate_v2_predictions(oos[label_cols], oos, specs, top_k=top_k) if len(oos) else {}
    report = {
        "purge_bars": purge_bars,
        "embargo_bars": embargo_bars,
        "n_splits": len(folds),
        "architecture": {
            "tabular": True,
            "catboost_ranker": True,
            "social": True,
            "bocd": True,
            "temporal_transformer": enable_temporal,
            "wallet_transaction_gnn": enable_gnn,
        },
        "folds": fold_reports,
        "aggregate_oos": aggregate,
    }
    oos.to_parquet(outdir / "oos_predictions.parquet", index=False)
    (outdir / "oos_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return oos, report
