from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from .solana import PumpPortalStreamConfig
from .store import V3Store


@dataclass(slots=True)
class StreamStats:
    connected: bool = False
    reconnects: int = 0
    messages: int = 0
    inserted_events: int = 0
    duplicates: int = 0
    last_message_monotonic: float | None = None
    last_error: str | None = None


@dataclass(slots=True)
class PumpStreamCollector:
    """Resilient read-only Pump.fun/PumpSwap event collector.

    Uses one websocket connection and dynamically manages token subscriptions.
    This is intentionally observation-only: it never signs or broadcasts.
    """

    store: V3Store
    config: PumpPortalStreamConfig = field(default_factory=PumpPortalStreamConfig)
    max_active_mints: int = 200
    token_ttl_seconds: int = 6 * 3600
    reconnect_min_seconds: float = 1.0
    reconnect_max_seconds: float = 30.0
    stats: StreamStats = field(default_factory=StreamStats)
    _active_mints: OrderedDict[str, float] = field(default_factory=OrderedDict)

    async def _send(self, ws: Any, method: str, keys: list[str] | None = None) -> None:
        payload: dict[str, Any] = {"method": method}
        if keys:
            payload["keys"] = keys
        await ws.send(json.dumps(payload))

    async def _subscribe_baseline(self, ws: Any) -> None:
        await self._send(ws, self.config.subscribe_new_token_method)
        if self._active_mints:
            await self._send(ws, self.config.subscribe_token_trade_method, list(self._active_mints))

    async def watch_mints(self, ws: Any, mints: list[str]) -> list[str]:
        now = time.time()
        added: list[str] = []
        for mint in mints:
            mint = str(mint).strip()
            if not mint:
                continue
            is_new = mint not in self._active_mints
            self._active_mints[mint] = now
            self._active_mints.move_to_end(mint)
            if is_new:
                added.append(mint)
        while len(self._active_mints) > self.max_active_mints:
            self._active_mints.popitem(last=False)
        if added:
            await self._send(ws, self.config.subscribe_token_trade_method, added)
        return added

    def expire_mints(self) -> list[str]:
        cutoff = time.time() - self.token_ttl_seconds
        expired = [mint for mint, touched in self._active_mints.items() if touched < cutoff]
        for mint in expired:
            self._active_mints.pop(mint, None)
        return expired

    def _normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        out = dict(payload)
        tx_type = str(out.get("txType") or out.get("type") or "unknown")
        if tx_type == "create":
            out["event_type"] = "NEW_TOKEN"
        elif tx_type in {"buy", "sell"}:
            out["event_type"] = "TOKEN_TRADE"
        else:
            out["event_type"] = tx_type.upper()
        if out.get("mint"):
            self._active_mints[str(out["mint"])] = time.time()
            self._active_mints.move_to_end(str(out["mint"]))
        return out

    async def run(self, *, stop_event: asyncio.Event | None = None) -> StreamStats:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("PumpStreamCollector requires the 'websockets' package") from exc

        stop_event = stop_event or asyncio.Event()
        backoff = self.reconnect_min_seconds
        while not stop_event.is_set():
            try:
                async with websockets.connect(
                    self.config.websocket_url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10,
                    max_queue=10_000,
                ) as ws:
                    self.stats.connected = True
                    self.stats.last_error = None
                    await self._subscribe_baseline(ws)
                    backoff = self.reconnect_min_seconds
                    while not stop_event.is_set():
                        raw = await asyncio.wait_for(ws.recv(), timeout=60)
                        self.stats.messages += 1
                        self.stats.last_message_monotonic = time.monotonic()
                        payload = json.loads(raw)
                        if not isinstance(payload, dict):
                            continue
                        normalized = self._normalize(payload)
                        inserted = self.store.append_event(normalized, source="pumpportal")
                        if inserted:
                            self.stats.inserted_events += 1
                        else:
                            self.stats.duplicates += 1
                        if normalized.get("event_type") == "NEW_TOKEN" and normalized.get("mint"):
                            await self.watch_mints(ws, [str(normalized["mint"])])
                        self.expire_mints()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.stats.connected = False
                self.stats.reconnects += 1
                self.stats.last_error = repr(exc)
                if stop_event.is_set():
                    break
                await asyncio.sleep(backoff)
                backoff = min(self.reconnect_max_seconds, backoff * 2.0)
        self.stats.connected = False
        return self.stats
