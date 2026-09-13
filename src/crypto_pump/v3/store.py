from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


SCHEMA_VERSION = 1


def _json(value: Any) -> str:
    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _event_id(source: str, payload: dict[str, Any]) -> str:
    explicit = payload.get("signature") or payload.get("tx") or payload.get("transactionSignature")
    event_type = payload.get("event_type") or payload.get("txType") or payload.get("type") or "event"
    mint = payload.get("mint") or payload.get("token") or ""
    if explicit:
        raw = f"{source}|{event_type}|{mint}|{explicit}"
    else:
        raw = f"{source}|{_json(payload)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class V3Store:
    """Small durable store for point-in-time events, signals and shadow positions.

    SQLite is deliberately used here instead of an in-memory dataframe so a
    collector can restart without losing idempotency or open shadow positions.
    Raw event JSON is retained unchanged for audit/replay.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=30.0)
        con.row_factory = sqlite3.Row
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            yield con
            con.commit()
        finally:
            con.close()

    def _init_schema(self) -> None:
        with self.connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    received_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    event_type TEXT,
                    mint TEXT,
                    signature TEXT,
                    slot INTEGER,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_mint_time
                    ON events(mint, received_at);
                CREATE INDEX IF NOT EXISTS idx_events_type_time
                    ON events(event_type, received_at);

                CREATE TABLE IF NOT EXISTS signals (
                    signal_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    feature_timestamp TEXT NOT NULL,
                    mint TEXT NOT NULL,
                    horizon TEXT NOT NULL,
                    probability REAL NOT NULL,
                    rank_value REAL,
                    model_version TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_signals_mint_time
                    ON signals(mint, created_at);

                CREATE TABLE IF NOT EXISTS shadow_positions (
                    position_id TEXT PRIMARY KEY,
                    signal_id TEXT UNIQUE NOT NULL,
                    mint TEXT NOT NULL,
                    opened_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    requested_notional_usd REAL NOT NULL,
                    approved_notional_usd REAL NOT NULL,
                    input_lamports INTEGER,
                    token_amount_atomic INTEGER,
                    entry_sol REAL,
                    entry_usd REAL,
                    entry_impact_bps REAL,
                    entry_quote_json TEXT,
                    closed_at TEXT,
                    exit_lamports INTEGER,
                    exit_sol REAL,
                    net_return REAL,
                    net_pnl_usd REAL,
                    close_reason TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_positions_status_time
                    ON shadow_positions(status, opened_at);

                CREATE TABLE IF NOT EXISTS health_samples (
                    sampled_at TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                );
                """
            )
            con.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def append_event(
        self,
        payload: dict[str, Any],
        *,
        source: str = "pumpportal",
        received_at: str | pd.Timestamp | None = None,
    ) -> bool:
        received = pd.Timestamp.now(tz="UTC") if received_at is None else pd.Timestamp(received_at)
        if received.tzinfo is None:
            received = received.tz_localize("UTC")
        else:
            received = received.tz_convert("UTC")
        event_id = _event_id(source, payload)
        event_type = payload.get("event_type") or payload.get("txType") or payload.get("type")
        mint = payload.get("mint") or payload.get("token")
        signature = payload.get("signature") or payload.get("tx") or payload.get("transactionSignature")
        slot = payload.get("slot")
        with self.connect() as con:
            cur = con.execute(
                """INSERT OR IGNORE INTO events
                (event_id, received_at, source, event_type, mint, signature, slot, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (event_id, received.isoformat(), source, event_type, mint, signature, slot, _json(payload)),
            )
            return cur.rowcount == 1

    def events(
        self,
        *,
        mint: str | None = None,
        since: str | pd.Timestamp | None = None,
        limit: int = 100_000,
    ) -> pd.DataFrame:
        clauses: list[str] = []
        args: list[Any] = []
        if mint is not None:
            clauses.append("mint = ?")
            args.append(mint)
        if since is not None:
            clauses.append("received_at >= ?")
            args.append(pd.Timestamp(since).isoformat())
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = f"SELECT * FROM events{where} ORDER BY received_at ASC LIMIT ?"
        args.append(int(limit))
        with self.connect() as con:
            return pd.read_sql_query(sql, con, params=args)

    def save_signal(self, payload: dict[str, Any]) -> tuple[str, bool]:
        feature_ts = pd.Timestamp(payload["feature_timestamp"])
        if feature_ts.tzinfo is None:
            feature_ts = feature_ts.tz_localize("UTC")
        key = "|".join([
            str(payload["mint"]), feature_ts.isoformat(), str(payload["horizon"]),
            str(payload.get("model_version") or "unknown"),
        ])
        signal_id = hashlib.sha256(key.encode("utf-8")).hexdigest()
        now = pd.Timestamp.now(tz="UTC").isoformat()
        with self.connect() as con:
            cur = con.execute(
                """INSERT OR IGNORE INTO signals
                (signal_id, created_at, feature_timestamp, mint, horizon, probability,
                 rank_value, model_version, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    signal_id, now, feature_ts.isoformat(), str(payload["mint"]),
                    str(payload["horizon"]), float(payload["probability"]),
                    payload.get("rank"), payload.get("model_version"), _json(payload),
                ),
            )
            return signal_id, cur.rowcount == 1

    def open_position(self, payload: dict[str, Any]) -> tuple[str, bool]:
        signal_id = str(payload["signal_id"])
        position_id = hashlib.sha256(f"shadow|{signal_id}".encode("utf-8")).hexdigest()
        with self.connect() as con:
            cur = con.execute(
                """INSERT OR IGNORE INTO shadow_positions
                (position_id, signal_id, mint, opened_at, status,
                 requested_notional_usd, approved_notional_usd, input_lamports,
                 token_amount_atomic, entry_sol, entry_usd, entry_impact_bps,
                 entry_quote_json, payload_json)
                VALUES (?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    position_id, signal_id, str(payload["mint"]), str(payload["opened_at"]),
                    float(payload["requested_notional_usd"]), float(payload["approved_notional_usd"]),
                    payload.get("input_lamports"), payload.get("token_amount_atomic"),
                    payload.get("entry_sol"), payload.get("entry_usd"), payload.get("entry_impact_bps"),
                    _json(payload.get("entry_quote") or {}), _json(payload),
                ),
            )
            return position_id, cur.rowcount == 1

    def open_positions(self) -> pd.DataFrame:
        with self.connect() as con:
            return pd.read_sql_query(
                "SELECT * FROM shadow_positions WHERE status='OPEN' ORDER BY opened_at ASC", con
            )

    def close_position(
        self,
        position_id: str,
        *,
        closed_at: str,
        exit_lamports: int | None,
        exit_sol: float | None,
        net_return: float | None,
        net_pnl_usd: float | None,
        close_reason: str,
    ) -> bool:
        with self.connect() as con:
            cur = con.execute(
                """UPDATE shadow_positions
                   SET status='CLOSED', closed_at=?, exit_lamports=?, exit_sol=?,
                       net_return=?, net_pnl_usd=?, close_reason=?
                   WHERE position_id=? AND status='OPEN'""",
                (
                    closed_at, exit_lamports, exit_sol, net_return,
                    net_pnl_usd, close_reason, position_id,
                ),
            )
            return cur.rowcount == 1

    def record_health(self, payload: dict[str, Any], sampled_at: str | None = None) -> None:
        ts = sampled_at or pd.Timestamp.now(tz="UTC").isoformat()
        with self.connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO health_samples(sampled_at, payload_json) VALUES (?, ?)",
                (ts, _json(payload)),
            )
