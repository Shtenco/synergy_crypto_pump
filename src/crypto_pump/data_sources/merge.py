from __future__ import annotations

import pandas as pd

REQUIRED = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]


def _regularize_one(p: pd.DataFrame, freq: str) -> pd.DataFrame:
    p = p.sort_values("timestamp").drop_duplicates("timestamp", keep="last").copy()
    p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True).dt.floor(freq)
    p = p.drop_duplicates("timestamp", keep="last").set_index("timestamp")
    if p.empty:
        return p.reset_index()
    idx = pd.date_range(p.index.min(), p.index.max(), freq=freq, tz="UTC")
    p = p.reindex(idx)
    p.index.name = "timestamp"
    close = p["close"].ffill()
    for c in ("open", "high", "low", "close"):
        p[c] = p[c].fillna(close)
    for c in ("volume", "quote_volume", "trades", "buy_volume", "sell_volume"):
        if c in p:
            p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0.0)
    for c in p.columns:
        if c not in {"open", "high", "low", "close", "volume", "quote_volume", "trades", "buy_volume", "sell_volume"}:
            p[c] = p[c].ffill().bfill()
    return p.reset_index()


def merge_and_regularize_panels(panels: list[pd.DataFrame], *, freq: str = "5min", min_bars: int = 600) -> pd.DataFrame:
    good = [p.copy() for p in panels if p is not None and len(p)]
    if not good:
        raise ValueError("no source returned market bars")
    df = pd.concat(good, ignore_index=True, sort=False)
    missing = set(REQUIRED).difference(df.columns)
    if missing:
        raise ValueError(f"missing required panel columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    parts = []
    for symbol, p in df.groupby("symbol", sort=False):
        q = _regularize_one(p, freq)
        if len(q) >= min_bars and q["close"].notna().sum() >= min_bars:
            q["symbol"] = symbol
            parts.append(q)
    if not parts:
        raise ValueError(f"no symbols survive min_bars={min_bars}")
    out = pd.concat(parts, ignore_index=True, sort=False)
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
