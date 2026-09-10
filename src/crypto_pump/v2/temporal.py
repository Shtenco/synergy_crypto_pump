from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .config import DEFAULT_HORIZONS, HorizonSpec
from .features import v2_numeric_feature_columns

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    DataLoader = TensorDataset = None
    TORCH_AVAILABLE = False


def build_causal_sequences(
    frame: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sequences ending exactly at row t and their original row indices."""
    seqs: list[np.ndarray] = []
    rows: list[int] = []
    ordered = frame.sort_values(["symbol", "timestamp"], kind="mergesort")
    for _, part in ordered.groupby("symbol", sort=False):
        values = part[feature_cols].to_numpy(dtype=np.float32)
        idx = part.index.to_numpy()
        for i in range(seq_len - 1, len(part)):
            seqs.append(values[i - seq_len + 1 : i + 1])
            rows.append(int(idx[i]))
    if not seqs:
        return np.empty((0, seq_len, len(feature_cols)), dtype=np.float32), np.empty(0, dtype=int)
    return np.stack(seqs), np.asarray(rows, dtype=int)


if TORCH_AVAILABLE:
    class CausalTemporalTransformer(nn.Module):
        def __init__(
            self,
            n_features: int,
            n_targets: int,
            d_model: int = 64,
            nhead: int = 4,
            layers: int = 2,
            dropout: float = 0.15,
            max_len: int = 512,
        ):
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
            enc = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, n_targets)

        def forward(self, x):
            n = x.shape[1]
            h = self.input_proj(x) + self.pos[:, :n]
            mask = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1)
            h = self.encoder(h, mask=mask)
            return self.head(self.norm(h[:, -1]))
else:
    class CausalTemporalTransformer:
        def __init__(self, *args, **kwargs):
            raise ImportError("Install crypto-pump[deep] to use the temporal Transformer")


@dataclass
class TemporalTransformerBranch:
    seq_len: int = 48
    epochs: int = 8
    batch_size: int = 256
    learning_rate: float = 1e-3
    random_state: int = 42
    features: list[str] = field(default_factory=list)
    medians: np.ndarray | None = None
    means: np.ndarray | None = None
    stds: np.ndarray | None = None
    model: object | None = None

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.where(np.isfinite(x), x, self.medians)
        return (x - self.means) / self.stds

    def fit(
        self,
        frame: pd.DataFrame,
        specs: Sequence[HorizonSpec] = DEFAULT_HORIZONS,
        feature_subset: list[str] | None = None,
    ) -> "TemporalTransformerBranch":
        if not TORCH_AVAILABLE:
            raise ImportError("Install crypto-pump[deep] to train the temporal branch")
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.features = feature_subset or v2_numeric_feature_columns(frame)
        raw = frame[self.features].to_numpy(float)
        self.medians = np.nanmedian(raw, axis=0)
        self.medians = np.where(np.isfinite(self.medians), self.medians, 0.0)
        filled = np.where(np.isfinite(raw), raw, self.medians)
        self.means = filled.mean(axis=0)
        self.stds = filled.std(axis=0)
        self.stds = np.where(self.stds > 1e-8, self.stds, 1.0)

        tmp = frame.copy()
        tmp.loc[:, self.features] = self._normalize(raw)
        seqs, rows = build_causal_sequences(tmp, self.features, self.seq_len)
        if not len(rows):
            raise ValueError("not enough rows to build temporal sequences")
        y = np.column_stack([
            frame.loc[rows, s.target_col].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
            for s in specs
        ])
        valid_rows = np.isfinite(y).any(axis=1)
        seqs, y = seqs[valid_rows], y[valid_rows]
        mask = np.isfinite(y).astype(np.float32)
        y = np.nan_to_num(y, nan=0.0).astype(np.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CausalTemporalTransformer(
            n_features=len(self.features), n_targets=len(specs)
        ).to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        ds = TensorDataset(
            torch.from_numpy(seqs),
            torch.from_numpy(y),
            torch.from_numpy(mask),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for _ in range(self.epochs):
            for xb, yb, mb in loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                logits = self.model(xb)
                loss_mat = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, yb, reduction="none"
                )
                loss = (loss_mat * mb).sum() / mb.sum().clamp_min(1.0)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
        self.model = self.model.cpu().eval()
        return self

    def predict(
        self,
        frame: pd.DataFrame,
        specs: Sequence[HorizonSpec] = DEFAULT_HORIZONS,
    ) -> pd.DataFrame:
        if not TORCH_AVAILABLE or self.model is None:
            raise RuntimeError("temporal branch is not fitted")
        raw = frame[self.features].to_numpy(float)
        tmp = frame.copy()
        tmp.loc[:, self.features] = self._normalize(raw)
        seqs, rows = build_causal_sequences(tmp, self.features, self.seq_len)
        out = pd.DataFrame(index=frame.index)
        for s in specs:
            out[f"temporal__{s.target_col}"] = np.nan
        if not len(rows):
            return out
        with torch.no_grad():
            p = torch.sigmoid(self.model(torch.from_numpy(seqs))).numpy()
        for j, s in enumerate(specs):
            out.loc[rows, f"temporal__{s.target_col}"] = p[:, j]
        return out
