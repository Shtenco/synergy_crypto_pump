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
    "virtual_sol_reserve_lamports",
)
TIMESTAMP_CANDIDATES = ("timestamp", "trade_time", "block_time", "event_time", "created_at", "time", "snapshot_time")
RELATIVE_TIME_CANDIDATES = ("seconds_since_launch", "seconds_since_creation", "age_seconds", "token_age_seconds")
LAUNCH_TIME_CANDIDATES = ("created_at", "launch_time", "launched_at", "first_seen_at", "created_timestamp", "timestamp")


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


def _timestamp_expression(source_cols: set[str], token_cols: set[str]) -> tuple[str, bool, str]:
    direct = _first(source_cols, TIMESTAMP_CANDIDATES)
    if direct:
        return f"CAST(s.{direct} AS TIMESTAMPTZ)", False, direct
    relative = _first(source_cols, RELATIVE_TIME_CANDIDATES)
    launch = _first(token_cols, LAUNCH_TIME_CANDIDATES)
    if relative and launch:
        return (
            f"CAST(k.{launch} AS TIMESTAMPTZ) + CAST(s.{relative} AS DOUBLE) * INTERVAL 1 SECOND",
            True,
            f"{launch}+{relative}",
        )
    raise ValueError(
        "Cannot derive absolute liquidity timestamp: expected direct timestamp or launch timestamp + relative seconds"
    )


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


def _select_curve_source(con, corpus_dir: Path) -> tuple[str, str, set[str]]:
    candidates = []
    trades = corpus_dir / "trades"
    if list(trades.glob("*.parquet")):
        candidates.append(("trades", str(trades / "*.parquet").replace("'", "''")))
    snapshots = corpus_dir / "snapshots.parquet"
    if snapshots.exists():
        candidates.append(("snapshots", str(snapshots).replace("'", "''")))
    diagnostics: dict[str, list[str]] = {}
    for name, parquet in candidates:
        cols = _columns(con, parquet)
        diagnostics[name] = sorted(cols)
        if _first(cols, RESERVE_CANDIDATES):
            return name, parquet, cols
    raise ValueError(f"No bonding-curve SOL reserve column found in trades or snapshots. Schemas={diagnostics}")


def enrich(panel_path: Path, corpus_dir: Path, sol_usd_path: Path, output_path: Path, report_path: Path) -> dict:
    duckdb = _duckdb()
    tokens_path = corpus_dir / "tokens.parquet"
    if not tokens_path.exists():
        raise FileNotFoundError("tokens.parquet is required for robust point-in-time liquidity reconstruction")

    con = duckdb.connect(database=":memory:")
    source_name, source_path, source_cols = _select_curve_source(con, corpus_dir)
    token_path_sql = str(tokens_path).replace("'", "''")
    token_cols = _columns(con, token_path_sql)
    reserve_col = _first(source_cols, RESERVE_CANDIDATES)
    assert reserve_col is not None
    ts_expr, needs_token_join, timestamp_source = _timestamp_expression(source_cols, token_cols)
    token_join = f"JOIN read_parquet('{token_path_sql}') k USING (mint)" if needs_token_join else ""

    median_raw = con.execute(
        f"SELECT median(CAST(s.{reserve_col} AS DOUBLE)) FROM read_parquet('{source_path}') s WHERE s.{reserve_col} IS NOT NULL AND CAST(s.{reserve_col} AS DOUBLE) > 0"
    ).fetchone()[0]
    if median_raw is None or float(median_raw) <= 0:
        raise ValueError(f"Bonding-curve reserve column {reserve_col} has no positive values")
    median_raw = float(median_raw)
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
        WITH curve AS (
            SELECT
                CAST(s.mint AS VARCHAR) AS symbol,
                {ts_expr} AS event_timestamp,
                CAST(s.{reserve_col} AS DOUBLE) / {reserve_scale} AS liquidity_sol
            FROM read_parquet('{source_path}') s
            {token_join}
            WHERE s.{reserve_col} IS NOT NULL AND CAST(s.{reserve_col} AS DOUBLE) > 0
        )
        SELECT
            symbol,
            time_bucket(INTERVAL '5 minutes', event_timestamp) AS timestamp,
            arg_max(liquidity_sol, event_timestamp) AS liquidity_sol
        FROM curve
        WHERE event_timestamp IS NOT NULL
        GROUP BY symbol, timestamp
        ORDER BY timestamp, symbol
        """
    ).df()
    con.close()
    reserve["timestamp"] = pd.to_datetime(reserve["timestamp"], utc=True)
    reserve["liquidity_sol"] = pd.to_numeric(reserve["liquidity_sol"], errors="coerce")

    prices = _read_prices(sol_usd_path)
    price_for_bars = pd.merge_asof(
        panel[["timestamp"]].drop_duplicates().sort_values("timestamp"),
        prices,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("10min"),
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
            f"price coverage={price_coverage:.3%}, source={source_name}, reserve_col={reserve_col}"
        )

    vals = panel.loc[finite, "liquidity_usd"]
    diagnostics = {
        "rows": int(len(panel)),
        "curve_source": source_name,
        "reserve_column": reserve_col,
        "timestamp_source": timestamp_source,
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
