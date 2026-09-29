from __future__ import annotations

import argparse
import io
import urllib.error
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

BASE = "https://data.binance.vision/data/spot"


def _months(start: pd.Timestamp, end: pd.Timestamp):
    cur = pd.Timestamp(start.year, start.month, 1, tz="UTC")
    stop = pd.Timestamp(end.year, end.month, 1, tz="UTC")
    while cur <= stop:
        yield cur.year, cur.month
        cur += pd.offsets.MonthBegin(1)


def _download(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "crypto-pump-v3-oos/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _parse_zip(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError("Binance archive contains no CSV")
        with z.open(names[0]) as f:
            d = pd.read_csv(f, header=None, usecols=[0, 4], names=["open_time", "sol_usd"])
    raw = pd.to_numeric(d["open_time"], errors="coerce")
    unit = "us" if raw.dropna().median() > 10**14 else "ms"
    d["timestamp"] = pd.to_datetime(raw, unit=unit, utc=True, errors="coerce")
    d["sol_usd"] = pd.to_numeric(d["sol_usd"], errors="coerce")
    return d[["timestamp", "sol_usd"]].dropna().query("sol_usd > 0")


def _daily_dates(start: pd.Timestamp, end: pd.Timestamp):
    cur = start.date()
    stop = end.date()
    while cur <= stop:
        yield cur
        cur += timedelta(days=1)


def download_sol_usd(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    monthly_ok = True
    for y, m in _months(start, end):
        url = f"{BASE}/monthly/klines/SOLUSDT/5m/SOLUSDT-5m-{y:04d}-{m:02d}.zip"
        blob = _download(url)
        if blob is None:
            monthly_ok = False
            break
        pieces.append(_parse_zip(blob))
    if not monthly_ok:
        pieces = []
        for day in _daily_dates(start, end):
            url = f"{BASE}/daily/klines/SOLUSDT/5m/SOLUSDT-5m-{day.isoformat()}.zip"
            blob = _download(url)
            if blob is None:
                raise FileNotFoundError(f"No monthly or daily Binance archive for {day}")
            pieces.append(_parse_zip(blob))
    out = pd.concat(pieces, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
    out = out[(out["timestamp"] >= start.floor("5min")) & (out["timestamp"] <= end.ceil("5min"))]
    if out.empty:
        raise ValueError("Downloaded SOL/USD series is empty for requested interval")
    expected = max(1, int((end - start) / pd.Timedelta(minutes=5)))
    coverage = len(out) / expected
    if coverage < 0.98:
        raise ValueError(f"SOL/USD 5m coverage too low: {coverage:.2%}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Download reproducible point-in-time SOL/USDT 5m history from Binance public archives")
    p.add_argument("--panel", type=Path, required=True, help="Panel used only to derive exact time range")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    d = pd.read_parquet(args.panel, columns=["timestamp"])
    ts = pd.to_datetime(d["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        raise ValueError("Panel has no valid timestamps")
    out = download_sol_usd(ts.min(), ts.max())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False, compression="zstd")
    print({"rows": len(out), "first": str(out.timestamp.min()), "last": str(out.timestamp.max()), "output": str(args.output)})


if __name__ == "__main__":
    main()
