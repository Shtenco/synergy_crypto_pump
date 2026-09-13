from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from crypto_pump.v2.branches import CatBoostRankerBranch, MultiHorizonTabularBranch, SocialBranch
from crypto_pump.v2.ensemble import CalibratedStacker
from crypto_pump.v2.features import v2_numeric_feature_columns

from .labels import DEFAULT_TRIPLE_BARRIERS, TripleBarrierSpec, add_triple_barrier_labels


@dataclass(frozen=True, slots=True)
class TradingHorizonSpec:
    """Duck-typed V2 HorizonSpec backed by V3 first-passage labels."""

    name: str
    horizon_bars: int
    return_threshold: float
    stop_loss: float

    @property
    def target_col(self) -> str:
        return f"tb_target__{self.name}"

    @property
    def mfe_col(self) -> str:
        # The ranker learns realized first-passage/timeout return, not future max high.
        return f"tb_exit_return__{self.name}"

    @property
    def mae_col(self) -> str:
        return f"tb_exit_return__{self.name}"

    @property
    def end_col(self) -> str:
        return f"tb_exit_return__{self.name}"

    @property
    def hit_col(self) -> str:
        return f"tb_exit_bar__{self.name}"


DEFAULT_TRADING_SPECS: tuple[TradingHorizonSpec, ...] = tuple(
    TradingHorizonSpec(s.name, s.horizon_bars, s.take_profit, s.stop_loss)
    for s in DEFAULT_TRIPLE_BARRIERS
)


def _label_numeric_columns(frame: pd.DataFrame) -> set[str]:
    return {
        c for c in frame.columns
        if c.startswith(("tb_target__", "tb_exit_bar__", "tb_exit_return__"))
    }


@dataclass
class V3TradingEnsemble:
    """V2 multi-branch architecture retrained on executable first-passage targets."""

    specs: Sequence[TradingHorizonSpec] = DEFAULT_TRADING_SPECS
    model_kind: str = "auto"
    random_state: int = 42
    tabular: MultiHorizonTabularBranch | None = None
    ranker: CatBoostRankerBranch | None = None
    social: SocialBranch | None = None
    stacker: CalibratedStacker | None = None
    features: list[str] = field(default_factory=list)

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> "V3TradingEnsemble":
        exclude = _label_numeric_columns(train)
        self.features = v2_numeric_feature_columns(train, exclude=exclude)
        self.features = [c for c in self.features if not c.startswith("tb_")]
        if not self.features:
            raise ValueError("V3TradingEnsemble: no leakage-safe numeric features")

        self.tabular = MultiHorizonTabularBranch(
            kind=self.model_kind, random_state=self.random_state, prefix="tabular"
        ).fit(train, self.specs, feature_subset=self.features)
        self.ranker = CatBoostRankerBranch(
            kind=self.model_kind, random_state=self.random_state
        ).fit(train, self.specs, feature_subset=self.features)
        self.social = SocialBranch(kind=self.model_kind, random_state=self.random_state).fit(
            train, self.specs
        )
        val_scores = self._base_scores(validation)
        self.stacker = CalibratedStacker().fit(val_scores, validation, self.specs)
        return self

    def _base_scores(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.tabular is None or self.ranker is None:
            raise RuntimeError("V3TradingEnsemble is not fitted")
        pieces = [self.tabular.predict(frame, self.specs), self.ranker.predict(frame, self.specs)]
        if self.social is not None and self.social.features:
            pieces.append(self.social.predict(frame, self.specs))
        scores = pd.concat(pieces, axis=1)
        if "bocd_cp_prob" in frame:
            scores["bocd_cp_prob"] = frame["bocd_cp_prob"].astype(float)
        return scores

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.stacker is None:
            raise RuntimeError("V3TradingEnsemble is not fitted")
        missing = sorted(set(self.features).difference(frame.columns))
        if missing:
            raise ValueError(f"live feature parity failure: {missing}")
        base = self._base_scores(frame)
        final = self.stacker.predict(base, self.specs)
        return pd.concat([base, final], axis=1)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @staticmethod
    def load(path: str | Path) -> "V3TradingEnsemble":
        return joblib.load(path)


def add_v3_training_labels(
    frame: pd.DataFrame,
    *,
    specs: Sequence[TradingHorizonSpec] = DEFAULT_TRADING_SPECS,
    entry_delay_bars: int = 1,
) -> pd.DataFrame:
    barriers = tuple(
        TripleBarrierSpec(s.name, s.horizon_bars, s.return_threshold, s.stop_loss)
        for s in specs
    )
    return add_triple_barrier_labels(frame, barriers, entry_delay_bars=entry_delay_bars)


def classification_report(
    labeled: pd.DataFrame,
    predictions: pd.DataFrame,
    specs: Sequence[TradingHorizonSpec] = DEFAULT_TRADING_SPECS,
    top_k: int = 10,
) -> dict[str, dict[str, float]]:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    report: dict[str, dict[str, float]] = {}
    for spec in specs:
        pcol = f"pump_probability__{spec.name}"
        if pcol not in predictions:
            report[spec.name] = {}
            continue
        use = labeled[spec.target_col].notna() & predictions[pcol].notna()
        if not use.any():
            report[spec.name] = {}
            continue
        y = labeled.loc[use, spec.target_col].astype(int).to_numpy()
        p = predictions.loc[use, pcol].to_numpy(float)
        m = {
            "rows": float(len(y)),
            "positive_rate": float(y.mean()),
            "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
        }
        if len(np.unique(y)) > 1:
            m["roc_auc"] = float(roc_auc_score(y, p))
        ranked = labeled.loc[use, ["timestamp", spec.target_col]].copy()
        ranked["p"] = p
        top = ranked.sort_values(["timestamp", "p"], ascending=[True, False]).groupby("timestamp").head(top_k)
        m[f"precision_at_{top_k}"] = float(top[spec.target_col].mean())
        report[spec.name] = m
    return report
