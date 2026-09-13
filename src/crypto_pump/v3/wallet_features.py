from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def causal_wallet_features(trades: pd.DataFrame, *, frequency: str = "5min") -> pd.DataFrame:
    """Build causal per-token wallet-quality proxies from observed buy flow.

    Each row uses trades at or before that bucket only. No future holder state is
    used, so the output is safe for point-in-time training and live parity.
    """
    if trades.empty:
        return pd.DataFrame()
    required = {"timestamp", "symbol", "side", "sol_amount", "trader"}
    missing = required.difference(trades.columns)
    if missing:
        raise ValueError(f"missing wallet-feature columns: {sorted(missing)}")
    d = trades.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d[d["side"].eq("buy") & d["trader"].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    d["bucket"] = d["timestamp"].dt.floor(frequency)
    d["sol_amount"] = pd.to_numeric(d["sol_amount"], errors="coerce").fillna(0.0).clip(lower=0.0)

    rows: list[dict] = []
    for symbol, g in d.groupby("symbol", sort=False):
        g = g.sort_values("timestamp", kind="mergesort")
        first_bucket = g["bucket"].min()
        balances: defaultdict[str, float] = defaultdict(float)
        counts: defaultdict[str, int] = defaultdict(int)
        first_seen: dict[str, pd.Timestamp] = {}
        total = 0.0
        for bucket, bg in g.groupby("bucket", sort=True):
            for r in bg.itertuples(index=False):
                wallet = str(r.trader)
                amount = float(r.sol_amount)
                balances[wallet] += amount
                counts[wallet] += 1
                total += amount
                first_seen.setdefault(wallet, pd.Timestamp(r.bucket))
            vals = np.array(sorted(balances.values(), reverse=True), dtype=float)
            buyers = len(vals)
            denom = total if total > 0 else np.nan
            top1 = float(vals[:1].sum() / denom) if buyers else np.nan
            top3 = float(vals[:3].sum() / denom) if buyers else np.nan
            top5 = float(vals[:5].sum() / denom) if buyers else np.nan
            repeat = float(sum(v > 1 for v in counts.values()) / buyers) if buyers else np.nan
            sniper_wallets = {
                w for w, seen in first_seen.items()
                if seen <= first_bucket + pd.Timedelta(frequency) * 2
            }
            sniper_flow = sum(balances[w] for w in sniper_wallets)
            sniper_pct = float(sniper_flow / denom) if buyers and np.isfinite(denom) else np.nan
            # Herfindahl concentration is stable and cheap enough for live use.
            shares = vals / denom if buyers and np.isfinite(denom) else np.array([], dtype=float)
            hhi = float(np.square(shares).sum()) if len(shares) else np.nan
            rows.append({
                "timestamp": bucket,
                "symbol": symbol,
                "unique_buyers_cumulative": buyers,
                "top_buyer_pct": top1,
                "top3_buyer_pct": top3,
                "top5_share": top5,
                "repeat_buyer_rate": repeat,
                "sniper_pct": sniper_pct,
                "wallet_hhi": hhi,
            })
    return pd.DataFrame(rows).sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)


def enrich_panel_with_wallet_features(panel: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel.copy()
    wf = causal_wallet_features(trades)
    if wf.empty:
        return panel.copy()
    out = panel.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    wf["timestamp"] = pd.to_datetime(wf["timestamp"], utc=True)
    return out.merge(wf, on=["timestamp", "symbol"], how="left", sort=False)
