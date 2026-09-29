from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

from .metrics import financial_report
from .store import V3Store


def dashboard_payload(store: V3Store) -> dict:
    open_positions = store.open_positions()
    with store.connect() as con:
        closed = pd.read_sql_query(
            "SELECT * FROM shadow_positions WHERE status='CLOSED' ORDER BY closed_at ASC", con
        )
        signals = pd.read_sql_query("SELECT * FROM signals ORDER BY created_at ASC", con)
        health = pd.read_sql_query("SELECT * FROM health_samples ORDER BY sampled_at ASC", con)

    metrics = financial_report(closed) if not closed.empty else {"trades": 0}
    days = 0
    if not closed.empty:
        opened = pd.to_datetime(closed["opened_at"], utc=True, errors="coerce")
        closed_at = pd.to_datetime(closed["closed_at"], utc=True, errors="coerce")
        if opened.notna().any() and closed_at.notna().any():
            days = max(1, int((closed_at.max() - opened.min()).total_seconds() // 86400) + 1)

    health_status = None
    health_reasons: list[str] = []
    if not health.empty:
        try:
            last = json.loads(health.iloc[-1]["payload_json"])
            health_status = last.get("status")
            health_reasons = list(last.get("reasons") or [])
        except Exception:
            health_status = "unparseable"

    return {
        "closed_trades": int(len(closed)),
        "open_positions": int(len(open_positions)),
        "signals": int(len(signals)),
        "shadow_days": days,
        "metrics": metrics,
        "health_status": health_status,
        "health_reasons": health_reasons,
    }


def write_dashboard(store_path: str | Path, output_path: str | Path) -> dict:
    store = V3Store(store_path)
    payload = dashboard_payload(store)
    rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in payload["metrics"].items()
    )
    reasons = ", ".join(payload["health_reasons"]) or "none"
    doc = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Crypto Pump V3 Shadow Dashboard</title>
<style>body{{font-family:system-ui;max-width:1100px;margin:40px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px;text-align:left}}.ok{{color:#087f23}}.bad{{color:#b00020}}</style></head>
<body><h1>Crypto Pump V3 — Shadow Validation</h1>
<p>Closed trades: <b>{payload['closed_trades']}</b> · Open positions: <b>{payload['open_positions']}</b> · Signals: <b>{payload['signals']}</b> · Shadow days: <b>{payload['shadow_days']}</b></p>
<p>Latest health: <b>{html.escape(str(payload['health_status']))}</b> · reasons: {html.escape(reasons)}</p>
<h2>Financial metrics</h2><table>{rows}</table>
<p><small>This dashboard is observational only. It cannot sign or broadcast transactions.</small></p>
</body></html>"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload
