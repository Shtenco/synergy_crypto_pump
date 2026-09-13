from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .risk import RiskManager
from .solana import JupiterQuoteClient
from .store import V3Store

SOL_MINT = "So11111111111111111111111111111111111111112"
LAMPORTS_PER_SOL = 1_000_000_000


@dataclass(frozen=True, slots=True)
class ShadowStrategyConfig:
    probability_column: str = "pump_probability__20pct_1h"
    horizon: str = "20pct_1h"
    min_probability: float = 0.70
    top_k: int = 10
    requested_notional_usd: float = 50.0
    tp_return: float = 0.20
    sl_return: float = -0.08
    max_hold_minutes: int = 60
    slippage_bps: int = 100
    min_quote_out_lamports: int = 1
    model_version: str = "v2"


@dataclass(slots=True)
class ShadowSignalEngine:
    """Convert model ranking into idempotent, quote-checked shadow positions."""

    store: V3Store
    quote_client: JupiterQuoteClient
    risk: RiskManager = field(default_factory=RiskManager)
    config: ShadowStrategyConfig = field(default_factory=ShadowStrategyConfig)

    @staticmethod
    def _impact_bps(quote: dict[str, Any]) -> float:
        try:
            return abs(float(quote.get("priceImpactPct", 0.0))) * 10_000.0
        except (TypeError, ValueError):
            return float("inf")

    def candidates(self, ranking: pd.DataFrame) -> pd.DataFrame:
        if ranking.empty:
            return ranking.copy()
        if self.config.probability_column not in ranking:
            raise ValueError(f"missing probability column: {self.config.probability_column}")
        d = ranking.copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        d = d[pd.to_numeric(d[self.config.probability_column], errors="coerce") >= self.config.min_probability]
        if d.empty:
            return d
        latest = d["timestamp"].max()
        d = d[d["timestamp"] == latest].copy()
        return d.sort_values(self.config.probability_column, ascending=False).head(self.config.top_k)

    def submit_ranking(
        self,
        ranking: pd.DataFrame,
        *,
        equity_usd: float,
        sol_usd: float,
        liquidity_column: str = "liquidity_usd",
    ) -> list[dict[str, Any]]:
        if sol_usd <= 0:
            raise ValueError("sol_usd must be > 0")
        accepted: list[dict[str, Any]] = []
        current_open = self.store.open_positions()
        self.risk.set_open_exposure(
            float(current_open["approved_notional_usd"].sum()) if not current_open.empty else 0.0
        )

        for rank, row in enumerate(self.candidates(ranking).itertuples(index=False), start=1):
            payload = row._asdict()
            mint = str(payload.get("mint") or payload.get("symbol"))
            if not mint or mint == "None":
                continue
            probability = float(payload[self.config.probability_column])
            feature_ts = pd.Timestamp(payload["timestamp"])
            signal_id, inserted = self.store.save_signal({
                "feature_timestamp": feature_ts,
                "mint": mint,
                "horizon": self.config.horizon,
                "probability": probability,
                "rank": rank,
                "model_version": self.config.model_version,
                "raw": payload,
            })
            if not inserted:
                continue

            requested = self.config.requested_notional_usd
            preliminary_lamports = max(1, int(requested / sol_usd * LAMPORTS_PER_SOL))
            quote = self.quote_client.quote(
                input_mint=SOL_MINT,
                output_mint=mint,
                amount_atomic=preliminary_lamports,
                slippage_bps=self.config.slippage_bps,
            )
            impact_bps = self._impact_bps(quote)
            out_amount = int(quote.get("outAmount") or 0)
            liquidity_usd = float(payload.get(liquidity_column) or np.nan)
            # Jupiter route availability is a hard executability gate. Liquidity
            # remains an independent risk input from the point-in-time panel.
            if out_amount < self.config.min_quote_out_lamports:
                continue
            decision = self.risk.evaluate(
                equity_usd=equity_usd,
                requested_notional_usd=requested,
                liquidity_usd=liquidity_usd,
                estimated_impact_bps=impact_bps,
                estimated_slippage_bps=float(self.config.slippage_bps),
                data_fresh=True,
                model_healthy=True,
            )
            if not decision.approved:
                continue

            approved_lamports = max(1, int(decision.notional_usd / sol_usd * LAMPORTS_PER_SOL))
            if approved_lamports != preliminary_lamports:
                quote = self.quote_client.quote(
                    input_mint=SOL_MINT,
                    output_mint=mint,
                    amount_atomic=approved_lamports,
                    slippage_bps=self.config.slippage_bps,
                )
                impact_bps = self._impact_bps(quote)
                out_amount = int(quote.get("outAmount") or 0)
                if out_amount <= 0:
                    continue
            opened_at = pd.Timestamp.now(tz="UTC")
            position = {
                "signal_id": signal_id,
                "mint": mint,
                "opened_at": opened_at.isoformat(),
                "requested_notional_usd": requested,
                "approved_notional_usd": decision.notional_usd,
                "input_lamports": approved_lamports,
                "token_amount_atomic": out_amount,
                "entry_sol": approved_lamports / LAMPORTS_PER_SOL,
                "entry_usd": decision.notional_usd,
                "entry_impact_bps": impact_bps,
                "entry_quote": quote,
                "probability": probability,
                "tp_return": self.config.tp_return,
                "sl_return": self.config.sl_return,
                "max_hold_minutes": self.config.max_hold_minutes,
            }
            position_id, created = self.store.open_position(position)
            if created:
                accepted.append({"position_id": position_id, **position})
                self.risk.set_open_exposure(self.risk.open_exposure_usd + decision.notional_usd)
        return accepted

    def mark_open_positions(self, *, sol_usd: float) -> list[dict[str, Any]]:
        now = pd.Timestamp.now(tz="UTC")
        actions: list[dict[str, Any]] = []
        open_positions = self.store.open_positions()
        for row in open_positions.itertuples(index=False):
            token_amount = int(row.token_amount_atomic or 0)
            if token_amount <= 0:
                continue
            quote = self.quote_client.quote(
                input_mint=str(row.mint),
                output_mint=SOL_MINT,
                amount_atomic=token_amount,
                slippage_bps=self.config.slippage_bps,
            )
            exit_lamports = int(quote.get("outAmount") or 0)
            if exit_lamports <= 0:
                continue
            entry_lamports = int(row.input_lamports or 0)
            if entry_lamports <= 0:
                continue
            gross_return = exit_lamports / entry_lamports - 1.0
            # Quote already reflects route/AMM price impact. Apply a conservative
            # residual haircut for network/MEV that the quote cannot guarantee.
            residual_cost = (15.0 + self.config.slippage_bps) / 10_000.0
            net_return = gross_return - residual_cost
            hold_minutes = (now - pd.Timestamp(row.opened_at)).total_seconds() / 60.0
            reason = None
            if net_return >= self.config.tp_return:
                reason = "take_profit"
            elif net_return <= self.config.sl_return:
                reason = "stop_loss"
            elif hold_minutes >= self.config.max_hold_minutes:
                reason = "timeout"
            if reason is None:
                actions.append({
                    "position_id": row.position_id,
                    "action": "HOLD",
                    "mark_return": net_return,
                    "quote": quote,
                })
                continue
            pnl_usd = float(row.approved_notional_usd) * net_return
            closed = self.store.close_position(
                str(row.position_id),
                closed_at=now.isoformat(),
                exit_lamports=exit_lamports,
                exit_sol=exit_lamports / LAMPORTS_PER_SOL,
                net_return=net_return,
                net_pnl_usd=pnl_usd,
                close_reason=reason,
            )
            if closed:
                self.risk.record_fill_result(pnl_usd=pnl_usd, failed=False)
                actions.append({
                    "position_id": row.position_id,
                    "action": "CLOSE",
                    "reason": reason,
                    "net_return": net_return,
                    "net_pnl_usd": pnl_usd,
                    "quote": quote,
                })
        return actions
