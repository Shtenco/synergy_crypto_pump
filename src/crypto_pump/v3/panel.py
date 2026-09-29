from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class PanelConfig:
    frequency: str = "5min"
    max_forward_fill_bars: int = 12


def _payload_rows(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if events.empty:
        return pd.DataFrame()
    for r in events.itertuples(index=False):
        raw = getattr(r, "payload_json", None)
        try:
            p = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        except Exception:
            continue
        p["received_at"] = getattr(r, "received_at", p.get("received_at"))
        rows.append(p)
    return pd.DataFrame(rows)


def normalize_pumpportal_trades(events: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw PumpPortal payloads into a causal trade tape.

    The stream is schema-tolerant because Pump.fun/PumpSwap payloads can differ
    by event type. Unknown/missing fields stay NaN rather than being invented.
    """
    p = _payload_rows(events)
    if p.empty:
        return p
    tx_type = p.get("txType", p.get("type", pd.Series(index=p.index, dtype=object))).astype(str).str.lower()
    mask = tx_type.isin(["buy", "sell"])
    p = p.loc[mask].copy()
    if p.empty:
        return p

    p["timestamp"] = pd.to_datetime(
        p.get("timestamp", p.get("blockTime", p["received_at"])), utc=True, errors="coerce"
    )
    p["symbol"] = p.get("mint", p.get("token")).astype(str)
    p["side"] = tx_type.loc[p.index]
    p["token_amount"] = pd.to_numeric(p.get("tokenAmount"), errors="coerce")
    p["sol_amount"] = pd.to_numeric(p.get("solAmount"), errors="coerce")
    p["price_sol"] = p["sol_amount"] / p["token_amount"].replace(0, np.nan)
    p["market_cap_sol"] = pd.to_numeric(p.get("marketCapSol"), errors="coerce")
    p["current_vsol"] = pd.to_numeric(p.get("vSolInBondingCurve"), errors="coerce")
    p["current_vtokens"] = pd.to_numeric(p.get("vTokensInBondingCurve"), errors="coerce")
    p["trader"] = p.get("traderPublicKey", p.get("trader", pd.Series(index=p.index, dtype=object)))
    p["signature"] = p.get("signature", p.get("tx", p.get("transactionSignature")))
    return p[
        ["timestamp", "symbol", "side", "token_amount", "sol_amount", "price_sol",
         "market_cap_sol", "current_vsol", "current_vtokens", "trader", "signature"]
    ].dropna(subset=["timestamp", "symbol", "price_sol"])


def trades_to_panel(
    trades: pd.DataFrame,
    *,
    sol_usd: float | pd.Series | None = None,
    config: PanelConfig | None = None,
) -> pd.DataFrame:
    cfg = config or PanelConfig()
    if trades.empty:
        return pd.DataFrame()
    d = trades.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.sort_values(["symbol", "timestamp"], kind="mergesort")
    d["bucket"] = d["timestamp"].dt.floor(cfg.frequency)
    d["buy_sol"] = np.where(d["side"].eq("buy"), d["sol_amount"], 0.0)
    d["sell_sol"] = np.where(d["side"].eq("sell"), d["sol_amount"], 0.0)
    d["buy_trade"] = d["side"].eq("buy").astype(int)
    d["sell_trade"] = d["side"].eq("sell").astype(int)

    def agg(g: pd.DataFrame) -> pd.Series:
        px = g["price_sol"].astype(float)
        return pd.Series({
            "open": px.iloc[0],
            "high": px.max(),
            "low": px.min(),
            "close": px.iloc[-1],
            "volume": g["token_amount"].abs().sum(),
            "quote_volume": g["sol_amount"].abs().sum(),
            "buy_volume": g["buy_sol"].sum(),
            "sell_volume": g["sell_sol"].sum(),
            "trades": len(g),
            "buy_trade_count": g["buy_trade"].sum(),
            "sell_trade_count": g["sell_trade"].sum(),
            "unique_buyers": g.loc[g["side"].eq("buy"), "trader"].nunique(dropna=True),
            "unique_sellers": g.loc[g["side"].eq("sell"), "trader"].nunique(dropna=True),
            "market_cap_sol": g["market_cap_sol"].dropna().iloc[-1] if g["market_cap_sol"].notna().any() else np.nan,
            "current_vsol": g["current_vsol"].dropna().iloc[-1] if g["current_vsol"].notna().any() else np.nan,
            "current_vtokens": g["current_vtokens"].dropna().iloc[-1] if g["current_vtokens"].notna().any() else np.nan,
        })

    grouped_rows: list[dict[str, Any]] = []
    for (symbol, bucket), g in d.groupby(["symbol", "bucket"], sort=False):
        row = agg(g).to_dict()
        row["symbol"] = symbol
        row["timestamp"] = bucket
        grouped_rows.append(row)
    panel = pd.DataFrame(grouped_rows)

    full: list[pd.DataFrame] = []
    for symbol, g in panel.groupby("symbol", sort=False):
        g = g.set_index("timestamp").sort_index()
        idx = pd.date_range(g.index.min(), g.index.max(), freq=cfg.frequency, tz="UTC")
        g = g.reindex(idx)
        g["symbol"] = symbol
        previous_close = g["close"].ffill(limit=cfg.max_forward_fill_bars)
        for col in ["open", "high", "low", "close"]:
            g[col] = g[col].fillna(previous_close)
        for col in ["volume", "quote_volume", "buy_volume", "sell_volume", "trades", "buy_trade_count", "sell_trade_count", "unique_buyers", "unique_sellers"]:
            g[col] = g[col].fillna(0.0)
        for col in ["market_cap_sol", "current_vsol", "current_vtokens"]:
            g[col] = g[col].ffill(limit=cfg.max_forward_fill_bars)
        full.append(g.rename_axis("timestamp").reset_index())
    out = pd.concat(full, ignore_index=True) if full else pd.DataFrame()

    if sol_usd is not None and not out.empty:
        if np.isscalar(sol_usd):
            out["sol_usd"] = float(sol_usd)
        else:
            s = pd.Series(sol_usd).copy()
            s.index = pd.to_datetime(s.index, utc=True)
            out = out.set_index("timestamp")
            out["sol_usd"] = s.reindex(out.index, method="ffill")
            out = out.reset_index()
        for col in ["open", "high", "low", "close"]:
            out[f"{col}_usd"] = out[col] * out["sol_usd"]
        out["liquidity"] = out["current_vsol"] * out["sol_usd"] * 2.0
        out["liquidity_usd"] = out["liquidity"]
        out["market_cap_usd"] = out["market_cap_sol"] * out["sol_usd"]

    return out.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
