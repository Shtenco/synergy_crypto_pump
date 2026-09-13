from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

SYSTEM_PROGRAM_WALLET = "BwWK17cbHxwWBKZkUYvzxLcNQ1YVyaFezduWbtm2de6s"
HF_REPO_ID = "Slinky21/Pumpfun_Memecoin_Corpus"

# Documented breaks in the fixed Jun-Jul 2026 corpus. These are used for
# diagnostics/stratification only and are intentionally emitted as strings so
# they cannot silently enter the numeric model feature set.
REGIME_BREAKS = (
    ("2026-06-09T00:00:00Z", "holder_cap_change"),
    ("2026-06-20T00:00:00Z", "mayhem_rollout"),
    ("2026-06-23T00:00:00Z", "concentration_bug_window"),
    ("2026-06-25T00:00:00Z", "silence_rate_shift"),
    ("2026-07-04T00:00:00Z", "post_outage_curve_shift"),
)


@dataclass(frozen=True, slots=True)
class CorpusQualityPolicy:
    exclude_system_program: bool = True
    require_positive_price: bool = True
    require_positive_sol_amount: bool = True
    exclude_suspect_token_concentration: bool = True
    reject_sol_price_ratio_outside: tuple[float, float] = (0.01, 100.0)


@dataclass(frozen=True, slots=True)
class CorpusBuildReport:
    input_trades: int
    clean_trades: int
    rejected_trades: int
    symbols: int
    bars: int
    first_timestamp: str | None
    last_timestamp: str | None
    output_path: str
    sha256: str
    quality_policy: dict


def _duckdb():
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("V3 historical corpus work requires `pip install -e '.[v3]'`") from exc
    return duckdb


def download_hf_corpus(
    destination: str | Path,
    *,
    include_trades: bool = True,
    token: str | None = None,
) -> Path:
    """Download the public PumpFun corpus using Hugging Face/Xet-aware tooling.

    Raw data is deliberately kept outside the repository. The downloader is
    resumable/cached by huggingface_hub and is safer for multi-GB Xet objects
    than hand-rolled HTTP downloads.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install V3 data dependencies: `pip install -e '.[v3]'`") from exc

    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    patterns = [
        "README.md",
        "KNOWN_ISSUES.md",
        "tokens.parquet",
        "snapshots.parquet",
        "postgard_snapshots.parquet",
        "postgard_outcomes.parquet",
        "wallet_stats.parquet",
        "migrations.parquet",
        "quickstart.py",
    ]
    if include_trades:
        patterns.append("trades/*.parquet")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=patterns,
        token=token,
    )
    return dest


def _columns(con, parquet_glob: str) -> set[str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet_glob}')").fetchall()
    return {str(r[0]) for r in rows}


def _first(columns: set[str], names: Iterable[str]) -> str | None:
    return next((name for name in names if name in columns), None)


def _timestamp_expression(trade_cols: set[str], token_cols: set[str]) -> tuple[str, bool]:
    direct = _first(
        trade_cols,
        ("timestamp", "trade_time", "block_time", "event_time", "created_at", "time"),
    )
    if direct:
        return f"CAST(t.{direct} AS TIMESTAMPTZ)", False
    seconds = _first(trade_cols, ("seconds_since_launch", "seconds_since_creation"))
    launch = _first(
        token_cols,
        ("created_at", "launch_time", "launched_at", "first_seen_at", "created_timestamp", "timestamp"),
    )
    if seconds and launch:
        return f"CAST(k.{launch} AS TIMESTAMPTZ) + CAST(t.{seconds} AS DOUBLE) * INTERVAL 1 SECOND", True
    raise ValueError(
        "Cannot derive an absolute trade timestamp. Expected a direct trade timestamp or "
        "seconds_since_launch plus a token launch timestamp. Inspect the corpus schema before continuing."
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _regime_case(ts_expr: str = "timestamp") -> str:
    parts = ["CASE"]
    for when, name in reversed(REGIME_BREAKS):
        parts.append(f"WHEN {ts_expr} >= TIMESTAMPTZ '{when}' THEN '{name}'")
    parts.append("ELSE 'initial_regime' END")
    return " ".join(parts)


def build_clean_hf_panel(
    data_dir: str | Path,
    output_path: str | Path,
    *,
    frequency: str = "5 minutes",
    start: str | None = None,
    end: str | None = None,
    policy: CorpusQualityPolicy | None = None,
    threads: int = 8,
    memory_limit: str = "8GB",
) -> CorpusBuildReport:
    """Build a leakage-aware OHLCV panel directly from the corpus trade ledger.

    Mandatory quality rules come from the corpus' own KNOWN_ISSUES audit:
    system-program pseudo-wallet rows are excluded, invalid/missing prices and
    SOL amounts are rejected, token-level concentration anomalies are removed,
    and internally inconsistent SOL amount rows are filtered using the documented
    sol_amount/(token_amount*price_sol) range.

    `wallet_stats.parquet` is intentionally not used because its activity/volume
    fields are documented as stale. Wallet features must be recomputed causally
    from the clean trade ledger instead.
    """
    duckdb = _duckdb()
    cfg = policy or CorpusQualityPolicy()
    root = Path(data_dir)
    trades_glob = str(root / "trades" / "*.parquet").replace("'", "''")
    tokens_path = str(root / "tokens.parquet").replace("'", "''")
    if not (root / "tokens.parquet").exists() or not list((root / "trades").glob("*.parquet")):
        raise FileNotFoundError("Expected tokens.parquet and trades/*.parquet in the corpus directory")

    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={int(threads)}")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute("PRAGMA preserve_insertion_order=false")
    trade_cols = _columns(con, trades_glob)
    token_cols = _columns(con, tokens_path)

    required = {"mint", "price_sol", "sol_amount", "is_buy"}
    missing = required.difference(trade_cols)
    if missing:
        raise ValueError(f"Corpus trade schema missing required columns: {sorted(missing)}")

    wallet_col = _first(trade_cols, ("user_wallet", "wallet", "trader", "user"))
    token_amount_col = _first(trade_cols, ("token_amount", "base_amount", "amount"))
    ts_expr, join_tokens_for_ts = _timestamp_expression(trade_cols, token_cols)
    join = f"JOIN read_parquet('{tokens_path}') k USING (mint)" if join_tokens_for_ts else f"LEFT JOIN read_parquet('{tokens_path}') k USING (mint)"

    filters = ["t.price_sol IS NOT NULL", "t.price_sol > 0"] if cfg.require_positive_price else []
    if cfg.require_positive_sol_amount:
        filters += ["t.sol_amount IS NOT NULL", "t.sol_amount > 0"]
    if cfg.exclude_system_program and wallet_col:
        filters.append(f"COALESCE(CAST(t.{wallet_col} AS VARCHAR), '') <> '{SYSTEM_PROGRAM_WALLET}'")
    if cfg.exclude_suspect_token_concentration and "top10_pct_suspect" in token_cols:
        filters.append("NOT COALESCE(k.top10_pct_suspect, FALSE)")
    lo, hi = cfg.reject_sol_price_ratio_outside
    if token_amount_col:
        filters.append(
            f"(t.{token_amount_col} IS NULL OR t.{token_amount_col} <= 0 OR "
            f"t.sol_amount / NULLIF(t.{token_amount_col} * t.price_sol, 0) BETWEEN {float(lo)} AND {float(hi)})"
        )
    if start:
        filters.append(f"{ts_expr} >= TIMESTAMPTZ '{pd.Timestamp(start).isoformat()}'")
    if end:
        filters.append(f"{ts_expr} < TIMESTAMPTZ '{pd.Timestamp(end).isoformat()}'")
    where = " AND ".join(filters) if filters else "TRUE"

    user_expr = f"CAST(t.{wallet_col} AS VARCHAR)" if wallet_col else "NULL::VARCHAR"
    token_expr = f"CAST(t.{token_amount_col} AS DOUBLE)" if token_amount_col else "NULL::DOUBLE"

    con.execute(
        f"""
        CREATE TEMP VIEW clean_trades AS
        SELECT
            {ts_expr} AS timestamp,
            CAST(t.mint AS VARCHAR) AS symbol,
            CAST(t.is_buy AS BOOLEAN) AS is_buy,
            CAST(t.price_sol AS DOUBLE) AS price_sol,
            CAST(t.sol_amount AS DOUBLE) AS sol_amount,
            {token_expr} AS token_amount,
            {user_expr} AS user_wallet,
            {('CAST(t.seconds_since_launch AS DOUBLE)' if 'seconds_since_launch' in trade_cols else 'NULL::DOUBLE')} AS seconds_since_launch
        FROM read_parquet('{trades_glob}') t
        {join}
        WHERE {where}
        """
    )

    input_trades = int(con.execute(f"SELECT COUNT(*) FROM read_parquet('{trades_glob}')").fetchone()[0])
    clean_trades = int(con.execute("SELECT COUNT(*) FROM clean_trades").fetchone()[0])

    # time_bucket is deterministic and makes the historical grid compatible
    # with the live five-minute panel.
    bucket_expr = f"time_bucket(INTERVAL '{frequency}', timestamp)"
    regime = _regime_case("timestamp")
    con.execute(
        f"""
        CREATE TEMP TABLE panel AS
        WITH b AS (
            SELECT
                symbol,
                {bucket_expr} AS timestamp,
                arg_min(price_sol, timestamp) AS open,
                max(price_sol) AS high,
                min(price_sol) AS low,
                arg_max(price_sol, timestamp) AS close,
                sum(COALESCE(token_amount, 0)) AS volume,
                sum(sol_amount) AS quote_volume,
                sum(CASE WHEN is_buy THEN sol_amount ELSE 0 END) AS buy_volume,
                sum(CASE WHEN NOT is_buy THEN sol_amount ELSE 0 END) AS sell_volume,
                count(*) AS trades,
                count(*) FILTER (WHERE is_buy) AS buy_trade_count,
                count(*) FILTER (WHERE NOT is_buy) AS sell_trade_count,
                count(DISTINCT user_wallet) FILTER (WHERE is_buy) AS unique_buyers,
                count(DISTINCT user_wallet) FILTER (WHERE NOT is_buy) AS unique_sellers
            FROM clean_trades
            GROUP BY symbol, {bucket_expr}
        )
        SELECT *, 'hf_pumpfun_corpus' AS data_source, {regime} AS source_regime
        FROM b
        ORDER BY timestamp, symbol
        """
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    escaped_out = str(out).replace("'", "''")
    con.execute(f"COPY panel TO '{escaped_out}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    stats = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT symbol), min(timestamp), max(timestamp) FROM panel"
    ).fetchone()
    digest = _sha256(out)
    report = CorpusBuildReport(
        input_trades=input_trades,
        clean_trades=clean_trades,
        rejected_trades=input_trades - clean_trades,
        symbols=int(stats[1] or 0),
        bars=int(stats[0] or 0),
        first_timestamp=None if stats[2] is None else str(stats[2]),
        last_timestamp=None if stats[3] is None else str(stats[3]),
        output_path=str(out),
        sha256=digest,
        quality_policy=asdict(cfg),
    )
    manifest = out.with_suffix(out.suffix + ".manifest.json")
    manifest.write_text(json.dumps(asdict(report), indent=2, sort_keys=True, default=str), encoding="utf-8")
    con.close()
    return report
