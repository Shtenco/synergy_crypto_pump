from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .config import DEFAULT_HORIZONS, HorizonSpec

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


@dataclass(slots=True)
class GraphSnapshot:
    x: np.ndarray
    edge_index: np.ndarray
    token_nodes: dict[str, int]


def build_wallet_token_snapshot(
    transactions: pd.DataFrame,
    as_of,
    lookback: str | pd.Timedelta | None = "30min",
) -> GraphSnapshot:
    """Build a wallet<->token graph using only edges timestamped <= as_of."""
    required = {"timestamp", "wallet", "symbol"}
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"transactions missing columns: {sorted(missing)}")
    tx = transactions.copy()
    tx["timestamp"] = pd.to_datetime(tx["timestamp"], utc=True)
    t = pd.Timestamp(as_of)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    tx = tx[tx["timestamp"] <= t]
    if lookback is not None:
        delta = pd.Timedelta(lookback)
        tx = tx[tx["timestamp"] > t - delta]
    if tx.empty:
        return GraphSnapshot(np.zeros((0, 6), np.float32), np.zeros((2, 0), np.int64), {})

    wallets = sorted(tx["wallet"].astype(str).unique())
    tokens = sorted(tx["symbol"].astype(str).unique())
    wmap = {w: i for i, w in enumerate(wallets)}
    tmap = {s: len(wallets) + i for i, s in enumerate(tokens)}
    x = np.zeros((len(wallets) + len(tokens), 6), dtype=np.float32)

    amount = tx["amount"].astype(float).abs() if "amount" in tx else pd.Series(1.0, index=tx.index)
    side = tx["direction"].astype(str).str.lower() if "direction" in tx else pd.Series("buy", index=tx.index)
    buy = side.isin(["buy", "b", "1", "long"]).astype(float)

    for wallet, grp_idx in tx.groupby(tx["wallet"].astype(str)).groups.items():
        idx = wmap[wallet]
        positions = list(grp_idx)
        a = amount.loc[positions]
        x[idx, 0] = 1.0
        x[idx, 2] = np.log1p(len(positions))
        x[idx, 3] = np.log1p(a.sum())
        x[idx, 4] = buy.loc[positions].mean()
        if "wallet_smart_score" in tx:
            x[idx, 5] = tx.loc[positions, "wallet_smart_score"].astype(float).mean()

    for symbol, grp_idx in tx.groupby(tx["symbol"].astype(str)).groups.items():
        idx = tmap[symbol]
        positions = list(grp_idx)
        a = amount.loc[positions]
        x[idx, 1] = 1.0
        x[idx, 2] = np.log1p(len(positions))
        x[idx, 3] = np.log1p(a.sum())
        x[idx, 4] = buy.loc[positions].mean()
        x[idx, 5] = np.log1p(tx.loc[positions, "wallet"].astype(str).nunique())

    src, dst = [], []
    for _, row in tx.iterrows():
        wi = wmap[str(row["wallet"])]
        ti = tmap[str(row["symbol"])]
        src.extend([wi, ti])
        dst.extend([ti, wi])
    return GraphSnapshot(
        x=x,
        edge_index=np.asarray([src, dst], dtype=np.int64),
        token_nodes=tmap,
    )


if TORCH_AVAILABLE:
    class MeanGraphLayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.self_lin = nn.Linear(in_dim, out_dim)
            self.neigh_lin = nn.Linear(in_dim, out_dim)

        def forward(self, x, edge_index):
            src, dst = edge_index
            agg = torch.zeros_like(x)
            deg = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
            agg.index_add_(0, dst, x[src])
            deg.index_add_(0, dst, torch.ones((len(dst), 1), dtype=x.dtype, device=x.device))
            agg = agg / deg.clamp_min(1.0)
            return torch.relu(self.self_lin(x) + self.neigh_lin(agg))

    class WalletTokenGNN(nn.Module):
        def __init__(self, in_dim: int = 6, hidden: int = 48, n_targets: int = 4):
            super().__init__()
            self.g1 = MeanGraphLayer(in_dim, hidden)
            self.g2 = MeanGraphLayer(hidden, hidden)
            self.head = nn.Linear(hidden, n_targets)

        def forward(self, x, edge_index):
            return self.head(self.g2(self.g1(x, edge_index), edge_index))
else:
    class WalletTokenGNN:
        def __init__(self, *args, **kwargs):
            raise ImportError("Install crypto-pump[deep] to use the wallet GNN")


@dataclass
class WalletGNNBranch:
    epochs: int = 4
    learning_rate: float = 2e-3
    lookback: str = "30min"
    max_snapshots: int = 512
    random_state: int = 42
    model: object | None = None

    def fit(
        self,
        frame: pd.DataFrame,
        transactions: pd.DataFrame,
        specs: Sequence[HorizonSpec] = DEFAULT_HORIZONS,
    ) -> "WalletGNNBranch":
        if not TORCH_AVAILABLE:
            raise ImportError("Install crypto-pump[deep] to train the wallet GNN")
        torch.manual_seed(self.random_state)
        self.model = WalletTokenGNN(n_targets=len(specs))
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        times = np.array(sorted(pd.to_datetime(frame["timestamp"], utc=True).unique()))
        if len(times) > self.max_snapshots:
            times = times[-self.max_snapshots:]
        label_lookup = frame.set_index(["timestamp", "symbol"])
        self.model.train()
        for _ in range(self.epochs):
            for ts in times:
                snap = build_wallet_token_snapshot(transactions, ts, self.lookback)
                if not snap.token_nodes or snap.edge_index.shape[1] == 0:
                    continue
                x = torch.from_numpy(snap.x)
                edges = torch.from_numpy(snap.edge_index)
                logits = self.model(x, edges)
                node_ids, ys, masks = [], [], []
                for symbol, node in snap.token_nodes.items():
                    key = (pd.Timestamp(ts), symbol)
                    if key not in label_lookup.index:
                        continue
                    row = label_lookup.loc[key]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[-1]
                    vals = np.array([
                        float(row[s.target_col]) if pd.notna(row[s.target_col]) else np.nan
                        for s in specs
                    ], dtype=np.float32)
                    if not np.isfinite(vals).any():
                        continue
                    node_ids.append(node)
                    masks.append(np.isfinite(vals).astype(np.float32))
                    ys.append(np.nan_to_num(vals, nan=0.0))
                if not node_ids:
                    continue
                y = torch.tensor(np.stack(ys))
                mask = torch.tensor(np.stack(masks))
                pred = logits[torch.tensor(node_ids)]
                loss_mat = torch.nn.functional.binary_cross_entropy_with_logits(pred, y, reduction="none")
                loss = (loss_mat * mask).sum() / mask.sum().clamp_min(1.0)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        self.model.eval()
        return self

    def predict(
        self,
        frame: pd.DataFrame,
        transactions: pd.DataFrame,
        specs: Sequence[HorizonSpec] = DEFAULT_HORIZONS,
    ) -> pd.DataFrame:
        if not TORCH_AVAILABLE or self.model is None:
            raise RuntimeError("wallet GNN is not fitted")
        out = pd.DataFrame(index=frame.index)
        for s in specs:
            out[f"gnn__{s.target_col}"] = np.nan
        for ts, rows in frame.groupby("timestamp", sort=False):
            snap = build_wallet_token_snapshot(transactions, ts, self.lookback)
            if not snap.token_nodes or snap.edge_index.shape[1] == 0:
                continue
            with torch.no_grad():
                p = torch.sigmoid(self.model(
                    torch.from_numpy(snap.x), torch.from_numpy(snap.edge_index)
                )).numpy()
            for idx, row in rows.iterrows():
                node = snap.token_nodes.get(str(row["symbol"]))
                if node is None:
                    continue
                for j, s in enumerate(specs):
                    out.loc[idx, f"gnn__{s.target_col}"] = p[node, j]
        return out
