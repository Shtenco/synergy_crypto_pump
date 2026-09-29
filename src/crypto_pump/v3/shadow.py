from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .risk import RiskManager


@dataclass(frozen=True, slots=True)
class ShadowOrder:
    timestamp: str
    symbol: str
    side: str
    requested_notional_usd: float
    approved_notional_usd: float
    reference_price: float
    probability: float | None
    decision: str
    metadata: dict[str, Any]


class ShadowExecutor:
    """Record intended orders without signing or broadcasting transactions."""

    def __init__(self, journal_path: str | Path, risk: RiskManager | None = None):
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.risk = risk or RiskManager()

    def submit(
        self,
        *,
        timestamp,
        symbol: str,
        reference_price: float,
        requested_notional_usd: float,
        equity_usd: float,
        liquidity_usd: float,
        estimated_impact_bps: float,
        estimated_slippage_bps: float,
        probability: float | None = None,
        data_fresh: bool = True,
        model_healthy: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> ShadowOrder:
        decision = self.risk.evaluate(
            equity_usd=equity_usd,
            requested_notional_usd=requested_notional_usd,
            liquidity_usd=liquidity_usd,
            estimated_impact_bps=estimated_impact_bps,
            estimated_slippage_bps=estimated_slippage_bps,
            data_fresh=data_fresh,
            model_healthy=model_healthy,
        )
        order = ShadowOrder(
            timestamp=str(pd.Timestamp(timestamp)),
            symbol=str(symbol),
            side="BUY",
            requested_notional_usd=float(requested_notional_usd),
            approved_notional_usd=float(decision.notional_usd),
            reference_price=float(reference_price),
            probability=None if probability is None else float(probability),
            decision=decision.reason,
            metadata=metadata or {},
        )
        with self.journal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(order), sort_keys=True) + "\n")
        return order

    def read_journal(self) -> pd.DataFrame:
        if not self.journal_path.exists():
            return pd.DataFrame()
        with self.journal_path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return pd.DataFrame(rows)
