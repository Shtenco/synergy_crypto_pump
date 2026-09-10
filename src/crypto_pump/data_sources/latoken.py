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
        raise RuntimeError(f"LATOKEN /v2/pair returned unexpected payload: {type(rows).__name__}")
    for r in rows:
        if not isinstance(r, dict):
            continue
        status = str(r.get("status") or "").upper()
        # Keep active and historical rows if the endpoint exposes them. Do not
        # filter only by today's active set: that would create survivorship bias.
        if status and "PAIR_STATUS" in status and status not in {
            "PAIR_STATUS_ACTIVE", "PAIR_STATUS_INACTIVE", "PAIR_STATUS_DISABLED"
        }:
            continue
        base_raw = r.get("baseCurrency") or r.get("base") or r.get("currency") or r.get("baseAsset")
        quote_raw = r.get("quoteCurrency") or r.get("quote") or r.get("quoteAsset")
        base = _resolve_currency(base_raw, by_id)
        q = _resolve_currency(quote_raw, by_id)
        if base and q == quote.upper() and base != q:
            out.add((base, q))
    # BTC first so the benchmark is retained even when max_pairs is small.
    return sorted(out, key=lambda x: (x[0] != "BTC", x[0]))


def _parse_tv(payload: Any, base: str, quote: str) -> pd.DataFrame:
    p = _unwrap(payload)
    if not isinstance(p, dict):
        return pd.DataFrame()
    status = str(p.get("s") or p.get("status") or "").lower()
    if status in {"no_data", "nodata", "error", "fail", "failure"}:
        return pd.DataFrame()
    ts = p.get("t") or p.get("time") or []
    o = p.get("o") or p.get("open") or []
    h = p.get("h") or p.get("high") or []
    l = p.get("l") or p.get("low") or []
    c = p.get("c") or p.get("close") or []
    v = p.get("v") or p.get("volume") or []
    arrays = [ts, o, h, l, c, v]
    n = min(map(len, arrays)) if all(hasattr(x, "__len__") for x in arrays) else 0
    if not n:
        return pd.DataFrame()

    close = pd.to_numeric(c[:n], errors="coerce")
    # LATOKEN's own candle example reports v in quote-value scale (e.g. BTC/USDT
    # volume around millions while price is ~50k). Preserve that as quote_volume;
    # derive base volume only for the common OHLCV contract.
    qvol = pd.to_numeric(v[:n], errors="coerce").fillna(0.0)
    base_vol = qvol / close.replace(0.0, np.nan)
    d = pd.DataFrame({
        "timestamp": pd.to_datetime(np.asarray(ts[:n], dtype="int64"), unit="s", utc=True),
        "open": pd.to_numeric(o[:n], errors="coerce"),
        "high": pd.to_numeric(h[:n], errors="coerce"),
        "low": pd.to_numeric(l[:n], errors="coerce"),
        "close": close,
        "volume": base_vol.fillna(0.0),
        "quote_volume": qvol,
    })
    d["symbol"] = f"LATOKEN:{base}{quote}"
    d["source"] = "latoken"
    d["pair"] = f"{base}/{quote}"
    d["trades"] = np.nan
    return d.dropna(subset=["timestamp", "open", "high", "low", "close"])


def _fetch_window(base: str, quote: str, start_s: int, end_s: int) -> pd.DataFrame:
    payload = http_json(
        f"{BASE}/v2/tradingview/history",
        params={
            "symbol": f"{base}/{quote}",
            "resolution": "1m",
            "from": int(start_s),
            "to": int(end_s),
        },
        timeout=45,
        retries=4,
    )
    return _parse_tv(payload, base, quote)


def _aggregate_5m(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    d = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").copy()
    d["bar"] = pd.to_datetime(d["timestamp"], utc=True).dt.floor("5min")
    out = d.groupby("bar", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        quote_volume=("quote_volume", "sum"),
        source=("source", "first"),
        pair=("pair", "first"),
        symbol=("symbol", "first"),
    ).reset_index().rename(columns={"bar": "timestamp"})
    out["trades"] = np.nan
    return out


def _chunk_ranges(start: datetime, end: datetime, chunk_hours: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    cur = start
    step = timedelta(hours=max(1, chunk_hours))
    while cur < end:
        nxt = min(end, cur + step)
        out.append((int(cur.timestamp()), int(nxt.timestamp())))
        cur = nxt
    return out


def collect_latoken_panel(
    *,
    days: int = 14,
    max_pairs: int = 120,
    quote: str = "USDT",
    end: datetime | None = None,
    end_offset_days: int = 0,
    workers: int = 10,
    chunk_hours: int = 72,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collect LATOKEN 1-minute public candles and aggregate causally to 5m.

    LATOKEN does not document a native 5m resolution. We therefore request the
    documented 1m resolution in bounded windows, de-duplicate, then aggregate.
    `end_offset_days` makes historical GitHub Actions sharding reproducible.
    `max_pairs=0` means all USDT pairs exposed by /v2/pair.
    """
    anchor = end or datetime.now(timezone.utc)
    end = (anchor - timedelta(days=max(0, end_offset_days))).replace(second=0, microsecond=0)
    start = end - timedelta(days=max(1, days))
    pairs = _pair_universe(quote=quote)
    universe_pairs = len(pairs)
    if max_pairs > 0:
        pairs = pairs[:max_pairs]

    ranges = _chunk_ranges(start, end, chunk_hours)
    by_pair: dict[tuple[str, str], list[pd.DataFrame]] = {p: [] for p in pairs}
    errors: dict[str, str] = {}
    requests_total = len(pairs) * len(ranges)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {
            ex.submit(_fetch_window, b, q, s, e): (b, q, s, e)
            for b, q in pairs
            for s, e in ranges
        }
        for fut in concurrent.futures.as_completed(futs):
            b, q, s, e = futs[fut]
            try:
                d = fut.result()
                if len(d):
                    by_pair[(b, q)].append(d)
            except Exception as exc:
                key = f"{b}/{q}@{s}:{e}"
                if len(errors) < 250:
                    errors[key] = str(exc)[:800]

    frames: list[pd.DataFrame] = []
    minute_rows = 0
    for (b, q), parts in by_pair.items():
        if not parts:
            continue
        raw = pd.concat(parts, ignore_index=True, sort=False)
        raw = raw.drop_duplicates("timestamp", keep="last")
        minute_rows += len(raw)
        agg = _aggregate_5m(raw)
        if len(agg):
            frames.append(agg)

    panel = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if len(panel):
        panel = panel.drop_duplicates(["symbol", "timestamp"], keep="last")
        panel = panel.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)

    report = {
        "source": "latoken",
        "resolution_raw": "1m",
        "resolution_output": "5m",
        "universe_pairs": universe_pairs,
        "requested_pairs": len(pairs),
        "successful_pairs": int(panel["symbol"].nunique()) if len(panel) else 0,
        "minute_rows": int(minute_rows),
        "rows": len(panel),
        "requests_total": requests_total,
        "chunk_hours": chunk_hours,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "end_offset_days": int(end_offset_days),
        "errors_count": len(errors),
        "errors": errors,
    }
    return panel, report
