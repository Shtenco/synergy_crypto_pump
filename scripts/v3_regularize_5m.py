#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _duckdb():
    import duckdb
    return duckdb


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def regularize(source: Path, output: Path, report_path: Path, *, threads: int = 4, memory_limit: str = "7GB") -> dict:
    duckdb = _duckdb()
    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={int(threads)}")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    src = str(source).replace("'", "''")
    out = str(output).replace("'", "''")

    columns = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{src}')").fetchall()}
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume", "liquidity_usd"}
    missing = required - columns
    if missing:
        raise ValueError(f"regularization source missing columns: {sorted(missing)}")

    zero_cols = [
        c for c in (
            "volume", "quote_volume", "buy_volume", "sell_volume", "trades",
            "buy_trade_count", "sell_trade_count", "unique_buyers", "unique_sellers",
        ) if c in columns
    ]
    carry_numeric = [c for c in ("liquidity_sol", "sol_usd", "liquidity_usd") if c in columns]
    carry_text = [c for c in ("data_source", "source_regime") if c in columns]

    raw_rows = int(con.execute(f"SELECT COUNT(*) FROM read_parquet('{src}')").fetchone()[0])
    max_gap = con.execute(
        f"""
        WITH x AS (
          SELECT epoch(timestamp - lag(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp))/60.0 AS gap_min
          FROM read_parquet('{src}')
        )
        SELECT max(gap_min) FROM x
        """
    ).fetchone()[0]

    zero_sql = ",\n                ".join(f"COALESCE({c}, 0) AS {c}" for c in zero_cols)
    carry_num_sql = ",\n                ".join(
        f"last_value({c} IGNORE NULLS) OVER w AS {c}" for c in carry_numeric
    )
    carry_txt_sql = ",\n                ".join(
        f"last_value({c} IGNORE NULLS) OVER w AS {c}" for c in carry_text
    )
    optional_sections = [x for x in (zero_sql, carry_num_sql, carry_txt_sql) if x]
    optional_sql = (",\n                " + ",\n                ".join(optional_sections)) if optional_sections else ""

    con.execute(
        f"""
        COPY (
          WITH raw AS (
            SELECT * FROM read_parquet('{src}')
          ),
          bounds AS (
            SELECT symbol, min(timestamp) AS first_ts, max(timestamp) AS last_ts
            FROM raw
            GROUP BY symbol
          ),
          grid AS (
            SELECT b.symbol, gs AS timestamp
            FROM bounds b,
            LATERAL generate_series(b.first_ts, b.last_ts, INTERVAL '5 minutes') t(gs)
          ),
          j AS (
            SELECT
              g.symbol,
              g.timestamp,
              r.open, r.high, r.low, r.close,
              {", ".join("r." + c for c in zero_cols + carry_numeric + carry_text)}
            FROM grid g
            LEFT JOIN raw r USING(symbol, timestamp)
          ),
          f AS (
            SELECT
              *,
              last_value(close IGNORE NULLS) OVER w AS causal_close
            FROM j
            WINDOW w AS (
              PARTITION BY symbol ORDER BY timestamp
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )
          )
          SELECT
            timestamp,
            symbol,
            COALESCE(open, causal_close) AS open,
            COALESCE(high, causal_close) AS high,
            COALESCE(low, causal_close) AS low,
            COALESCE(close, causal_close) AS close
            {optional_sql}
          FROM f
          WINDOW w AS (
            PARTITION BY symbol ORDER BY timestamp
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
          )
          WHERE causal_close IS NOT NULL
          ORDER BY symbol, timestamp
        ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    output_rows = int(con.execute(f"SELECT COUNT(*) FROM read_parquet('{out}')").fetchone()[0])
    bad_steps = int(con.execute(
        f"""
        WITH x AS (
          SELECT epoch(timestamp - lag(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp))/60.0 AS gap_min
          FROM read_parquet('{out}')
        )
        SELECT count(*) FROM x WHERE gap_min IS NOT NULL AND abs(gap_min - 5.0) > 1e-9
        """
    ).fetchone()[0])
    liq_coverage = float(con.execute(
        f"SELECT avg(CASE WHEN liquidity_usd IS NOT NULL AND liquidity_usd > 0 THEN 1.0 ELSE 0.0 END) FROM read_parquet('{out}')"
    ).fetchone()[0] or 0.0)
    symbols = int(con.execute(f"SELECT count(DISTINCT symbol) FROM read_parquet('{out}')").fetchone()[0])
    first_ts, last_ts = con.execute(f"SELECT min(timestamp), max(timestamp) FROM read_parquet('{out}')").fetchone()
    con.close()

    report = {
        "input_rows": raw_rows,
        "output_rows": output_rows,
        "inserted_zero_flow_rows": output_rows - raw_rows,
        "symbols": symbols,
        "source_max_gap_minutes": None if max_gap is None else float(max_gap),
        "non_5m_steps_after_regularization": bad_steps,
        "liquidity_coverage": liq_coverage,
        "timestamp_min": str(first_ts),
        "timestamp_max": str(last_ts),
        "sha256": _sha256(output),
        "status": "PASS" if bad_steps == 0 and liq_coverage >= 0.95 else "FAIL",
    }
    if report["status"] != "PASS":
        raise RuntimeError(f"regularization fail-closed: {report}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Causally regularize Pump.fun history to an exact 5-minute grid")
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--memory-limit", default="7GB")
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(regularize(args.source, args.output, args.report, threads=args.threads, memory_limit=args.memory_limit), indent=2))


if __name__ == "__main__":
    main()
