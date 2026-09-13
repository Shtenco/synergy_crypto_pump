from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class CanaryLimits:
    max_trade_usd: float = 20.0
    max_daily_notional_usd: float = 100.0
    max_open_positions: int = 2
    max_daily_loss_usd: float = 25.0


@dataclass(frozen=True, slots=True)
class CanaryIntent:
    intent_id: str
    created_at: str
    mint: str
    side: str
    notional_usd: float
    max_slippage_bps: int
    expires_at: str
    metadata: dict[str, Any]
    mac: str


class CanaryIntentService:
    """Create authenticated execution intents without exposing a transaction signer.

    This service never stores a private key and has no Solana sendTransaction
    capability. A future isolated executor may consume intents only after the
    shadow->canary promotion gate has passed. Absolute caps are enforced again
    here even if upstream RiskManager values are larger.
    """

    def __init__(self, queue_path: str | Path, *, limits: CanaryLimits | None = None, secret: str | None = None):
        self.queue_path = Path(queue_path)
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        self.limits = limits or CanaryLimits()
        self.secret = (secret or os.getenv("CANARY_INTENT_HMAC_SECRET") or "").encode()
        if len(self.secret) < 32:
            raise ValueError("CANARY_INTENT_HMAC_SECRET must be at least 32 bytes")

    def _rows(self) -> list[dict[str, Any]]:
        if not self.queue_path.exists():
            return []
        with self.queue_path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _sign(self, payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hmac.new(self.secret, raw, hashlib.sha256).hexdigest()

    def create_intent(
        self,
        *,
        mint: str,
        notional_usd: float,
        max_slippage_bps: int,
        promotion_allowed: bool,
        current_open_positions: int,
        realized_today_usd: float,
        ttl_seconds: int = 60,
        metadata: dict[str, Any] | None = None,
    ) -> CanaryIntent:
        if not promotion_allowed:
            raise PermissionError("shadow_to_canary_gate_not_passed")
        if current_open_positions >= self.limits.max_open_positions:
            raise PermissionError("canary_open_position_limit")
        if realized_today_usd <= -abs(self.limits.max_daily_loss_usd):
            raise PermissionError("canary_daily_loss_limit")
        amount = min(float(notional_usd), self.limits.max_trade_usd)
        if amount <= 0:
            raise ValueError("notional_usd must be positive")

        now = pd.Timestamp.now(tz="UTC")
        today = now.date().isoformat()
        used_today = sum(
            float(r.get("notional_usd") or 0.0)
            for r in self._rows()
            if str(r.get("created_at", "")).startswith(today)
        )
        if used_today + amount > self.limits.max_daily_notional_usd:
            raise PermissionError("canary_daily_notional_limit")

        core = {
            "created_at": now.isoformat(),
            "mint": str(mint),
            "side": "BUY",
            "notional_usd": amount,
            "max_slippage_bps": int(max_slippage_bps),
            "expires_at": (now + pd.Timedelta(seconds=int(ttl_seconds))).isoformat(),
            "metadata": metadata or {},
        }
        intent_id = hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()
        signed_payload = {"intent_id": intent_id, **core}
        intent = CanaryIntent(**signed_payload, mac=self._sign(signed_payload))
        with self.queue_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(intent), sort_keys=True, default=str) + "\n")
        return intent

    def verify(self, row: dict[str, Any]) -> bool:
        payload = dict(row)
        provided = str(payload.pop("mac", ""))
        return hmac.compare_digest(provided, self._sign(payload))
