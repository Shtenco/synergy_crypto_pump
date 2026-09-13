from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


RESERVE_CANDIDATES = (
    "v_sol_bonding_curve",
    "virtual_sol_reserves",
    "virtual_sol_reserve",
    "v_sol_reserves",
)
TIMESTAMP_CANDIDATES = ("timestamp", "trade_time", "block_time", "event_time", "created_at", "time")


def _duckdb():
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install V3 dependencies with `pip install -e '.[v3]'`") from exc
    return duckdb


def _columns(con, parquet_glob: str) -> set[str]:
    return {str(r[0]) for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet_glob}')").fetchall()}


def _first(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    return next((c for c in candidates if c in columns), None)


def _read_prices(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    timestamp = next((c for c in ("timestamp", "time", "datetime", "date") if c in d.columns), None)
    price = next((c for c in ("sol_usd", "close", "price", "SOLUSD", "SOL_USD") if c in d.columns), None)
    if timestamp is None or price is None:
        raise ValueError("SOL/USD table must contain a timestamp/time column and sol_usd/close/price column")
    out = d[[timestamp, price]].rename(columns={timestamp: "timestamp", price: "sol_usd"}).copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["sol_usd"] = pd.to_numeric(out["sol_usd"], errors="coerce")
    out = out.dropna().query("sol_usd > 0").drop_duplicates("timestamp").sort_values("timestamp")
    if out.empty:
        raise ValueError("SOL/USD table has no valid positive prices")
    return out


def enrich(panel_path: Path, corpus_dir: Path, sol_usd_path: Path, output_path: Path, report_path: Path) -> dict:
    duckdb = _duckdb()
    trades_glob = str(corpus_dir / "trades" / "*.parquet").replace("'", "''")
    if not list((corpus_dir / "trades").glob("*.parquet")):
        raise FileNotFoundError("No trades/*.parquet shards found")

    con = duckdb.connect(database=":memory:")
    trade_cols = _columns(con, trades_glob)
    reserve_col = _first(trade_cols, RESERVE_CANDIDATES)
    ts_col = _first(trade_cols, TIMESTAMP_CANDIDATES)
    if reserve_col is None:
        raise ValueError(f"No bonding-curve SOL reserve column found. Available columns: {sorted(trade_cols)}")
    if ts_col is None:
        raise ValueError(f"No direct trade timestamp found for liquidity enrichment. Available columns: {sorted(trade_cols)}")

    median_raw = con.execute(
        f"SELECT median(CAST({reserve_col} AS DOUBLE)) FROM read_parquet('{trades_glob}') WHERE {reserve_col} IS NOT NULL AND CAST({reserve_col} AS DOUBLE) > 0"
    ).fetchone()[0]
    if median_raw is None or float(median_raw) <= 0:
        raise ValueError(f"Bonding-curve reserve column {reserve_col} has no positive values")
    median_raw = float(median_raw)
    # Pump.fun on-chain virtual SOL reserve is normally stored in lamports. Fail closed
    # on ambiguous magnitudes instead of silently labelling an unknown unit as USD.
    if median_raw >= 1_000_000:
        reserve_scale = 1e9
        reserve_unit = "lamports"
    elif 0.01 <= median_raw <= 1_000_000:
        reserve_scale = 1.0
        reserve_unit = "SOL"
    else:
        raise ValueError(f"Ambiguous {reserve_col} scale: median={median_raw}; refusing liquidity conversion")

    panel = pd.read_parquet(panel_path)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce")
    if panel.empty or panel["timestamp"].isna().all():
        raise ValueError("Historical panel is empty or has no valid timestamps")

    reserve = con.execute(
        f"""
        SELECT
            CAST(mint AS VARCHAR) AS symbol,
            time_bucket(INTERVAL '5 minutes', CAST({ts_col} AS TIMESTAMPTZ)) AS timestamp,
            arg_max(CAST({reserve_col} AS DOUBLE), CAST({ts_col} AS TIMESTAMPTZ)) / {reserve_scale} AS liquidity_sol
        FROM read_parquet('{trades_glob}')
        WHERE {reserve_col} IS NOT NULL AND CAST({reserve_col} AS DOUBLE) > 0
        GROUP BY symbol, timestamp
        ORDER BY timestamp, symbol
        """
    ).df()
    con.close()
    reserve["timestamp"] = pd.to_datetime(reserve["timestamp"], utc=True)
    reserve["liquidity_sol"] = pd.to_numeric(reserve["liquidity_sol"], errors="coerce")

    prices = _read_prices(sol_usd_path)
    # Point-in-time price: use only the latest SOL/USD observation at or before the bar.
    price_for_bars = pd.merge_asof(
        panel[["timestamp"]].drop_duplicates().sort_values("timestamp"),
        prices,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("6h"),
    )
    panel = panel.merge(reserve, on=["timestamp", "symbol"], how="left", validate="one_to_one")
    panel = panel.merge(price_for_bars, on="timestamp", how="left", validate="many_to_one")
    panel["liquidity_usd"] = panel["liquidity_sol"] * panel["sol_usd"]

    finite = panel["liquidity_usd"].notna() & (panel["liquidity_usd"] > 0)
    coverage = float(finite.mean()) if len(panel) else 0.0
    price_coverage = float(panel["sol_usd"].notna().mean()) if len(panel) else 0.0
    if coverage < 0.95:
        raise ValueError(
            f"liquidity_usd coverage {coverage:.3%} is below required 95%; "
            f"price coverage={price_coverage:.3%}, reserve_col={reserve_col}"
        )

    vals = panel.loc[finite, "liquidity_usd"]
    diagnostics = {
        "rows": int(len(panel)),
        "reserve_column": reserve_col,
        "reserve_raw_median": median_raw,
        "reserve_raw_unit": reserve_unit,
        "reserve_scale_to_sol": reserve_scale,
        "liquidity_coverage": coverage,
        "sol_usd_coverage": price_coverage,
        "liquidity_usd_min": float(vals.min()),
        "liquidity_usd_median": float(vals.median()),
        "liquidity_usd_p95": float(vals.quantile(0.95)),
        "first_timestamp": str(panel["timestamp"].min()),
        "last_timestamp": str(panel["timestamp"].max()),
        "status": "PASS",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False, compression="zstd")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    return diagnostics


def main() -> None:
    p = argparse.ArgumentParser(description="Attach point-in-time Pump.fun bonding-curve liquidity in USD")
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--corpus-dir", type=Path, required=True)
    p.add_argument("--sol-usd", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(enrich(args.panel, args.corpus_dir, args.sol_usd, args.output, args.report), indent=2))


if __name__ == "__main__":
    main()
