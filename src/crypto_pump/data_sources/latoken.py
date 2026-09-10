from __future__ import annotations

import concurrent.futures
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from .common import http_json

BASE = "https://api.latoken.com"


def _unwrap(payload: Any) -> Any:
    if isinstance(payload, dict):
        for k in ("result", "data", "payload"):
            if k in payload and payload[k] is not None:
                return payload[k]
    return payload


def _currencies() -> tuple[dict[str, str], dict[str, str]]:
    rows = _unwrap(http_json(f"{BASE}/v2/currency"))
    by_id: dict[str, str] = {}
    by_tag: dict[str, str] = {}
    if not isinstance(rows, list):
        return by_id, by_tag
    for r in rows:
        if not isinstance(r, dict):
            continue
        cid = str(r.get("id") or r.get("currency") or "")
        tag = str(r.get("tag") or r.get("symbol") or r.get("code") or "").upper()
        if cid and tag:
            by_id[cid.lower()] = tag
            by_tag[tag] = cid
    return by_id, by_tag


def _resolve_currency(v: Any, by_id: dict[str, str]) -> str:
    s = str(v or "")
    return by_id.get(s.lower(), s.upper())


def _pair_universe(quote: str = "USDT") -> list[tuple[str, str]]:
    by_id, _ = _currencies()
    rows = _unwrap(http_json(f"{BASE}/v2/pair"))
    out: set[tuple[str, str]] = set()
    if not isinstance(rows, list):
        raise RuntimeError("LATOKEN /v2/pair returned an unexpected payload")
    for r in rows:
        if not isinstance(r, dict):
            continue
        base_raw = r.get("baseCurrency") or r.get("base") or r.get("currency") or r.get("baseAsset")
        quote_raw = r.get("quoteCurrency") or r.get("quote") or r.get("quoteAsset")
        base = _resolve_currency(base_raw, by_id)
        q = _resolve_currency(quote_raw, by_id)
        if base and q == quote.upper() and base != q:
            out.add((base, q))
    return sorted(out, key=lambda x: (x[0] != "BTC", x[0]))


def _parse_tv(payload: Any, base: str, quote: str) -> pd.DataFrame:
    p = _unwrap(payload)
    if not isinstance(p, dict):
        return pd.DataFrame()
    ts = p.get("t") or p.get("time") or []
    o = p.get("o") or p.get("open") or []
    h = p.get("h") or p.get("high") or []
    l = p.get("l") or p.get("low") or []
    c = p.get("c") or p.get("close") or []
    v = p.get("v") or p.get("volume") or []
    n = min(map(len, [ts, o, h, l, c, v])) if all(hasattr(x, "__len__") for x in [ts,o,h,l,c,v]) else 0
    if not n:
        return pd.DataFrame()
    d = pd.DataFrame({
        "timestamp": pd.to_datetime(np.asarray(ts[:n], dtype="int64"), unit="s", utc=True),
        "open": pd.to_numeric(o[:n], errors="coerce"),
        "high": pd.to_numeric(h[:n], errors="coerce"),
        "low": pd.to_numeric(l[:n], errors="coerce"),
        "close": pd.to_numeric(c[:n], errors="coerce"),
        "volume": pd.to_numeric(v[:n], errors="coerce").fillna(0.0),
    })
    d["symbol"] = f"LATOKEN:{base}{quote}"
    d["source"] = "latoken"
    d["pair"] = f"{base}/{quote}"
    d["quote_volume"] = d["volume"] * d["close"]
    d["trades"] = np.nan
    return d.dropna(subset=["timestamp", "open", "high", "low", "close"])


def _fetch_pair(base: str, quote: str, start_s: int, end_s: int, resolution: str = "5") -> pd.DataFrame:
    payload = http_json(
        f"{BASE}/v2/tradingview/history",
        params={"symbol": f"{base}/{quote}", "resolution": resolution, "from": start_s, "to": end_s},
        timeout=40,
        retries=4,
    )
    return _parse_tv(payload, base, quote)


def collect_latoken_panel(*, days: int = 14, max_pairs: int = 120, quote: str = "USDT", end: datetime | None = None, workers: int = 8) -> tuple[pd.DataFrame, dict[str, Any]]:
    end = (end or datetime.now(timezone.utc)).replace(second=0, microsecond=0)
    start = end - timedelta(days=days)
    pairs = _pair_universe(quote=quote)
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(_fetch_pair, b, q, int(start.timestamp()), int(end.timestamp())): (b, q) for b, q in pairs}
        for fut in concurrent.futures.as_completed(futs):
            b, q = futs[fut]
            try:
                d = fut.result()
                if len(d):
                    frames.append(d)
            except Exception as exc:
                errors[f"{b}/{q}"] = str(exc)[:500]
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(panel):
        panel = panel.drop_duplicates(["symbol", "timestamp"], keep="last").sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    report = {
        "source": "latoken",
        "requested_pairs": len(pairs),
        "successful_pairs": int(panel["symbol"].nunique()) if len(panel) else 0,
        "rows": len(panel),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "errors": errors,
    }
    return panel, report
