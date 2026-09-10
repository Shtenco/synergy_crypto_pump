#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

root = Path("artifacts/real_v2")
data = json.loads((root / "data_report.json").read_text()) if (root / "data_report.json").exists() else {}
oos = json.loads((root / "oos/report.json").read_text()) if (root / "oos/report.json").exists() else {}
lines = ["# CRYPTO PUMP V2 — real-data run", "", "## Dataset", "", f"- rows: **{data.get('panel_rows', 0):,}**", f"- symbols: **{data.get('panel_symbols', 0)}**", f"- LATOKEN symbols: **{data.get('latoken_symbols', 0)}**", f"- Uniswap symbols: **{data.get('uniswap_symbols', 0)}**", f"- range: `{data.get('timestamp_min')}` → `{data.get('timestamp_max')}`", "", "## OOS", "", "```json", json.dumps(oos, indent=2)[:20000], "```", "", "## Source status", "", "```json", json.dumps(data.get('sources', []), indent=2)[:12000], "```"]
(root / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
