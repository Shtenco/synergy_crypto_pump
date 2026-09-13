from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator

import pandas as pd

SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
LAMPORTS_PER_SOL = 1_000_000_000
USDC_ATOMIC = 1_000_000


def _json_request(url: str, *, payload: dict | None = None, timeout: int = 20) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json", "User-Agent": "crypto-pump-v3/0.1"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


@dataclass(slots=True)
class SolanaRPC:
    url: str

    def call(self, method: str, params: list[Any] | None = None) -> Any:
        result = _json_request(
            self.url,
            payload={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or []},
        )
        if "error" in result:
            raise RuntimeError(f"Solana RPC error: {result['error']}")
        return result.get("result")

    def recent_signatures(
        self,
        address: str,
        limit: int = 100,
        *,
        before: str | None = None,
        until: str | None = None,
    ) -> list[dict]:
        opts: dict[str, Any] = {"limit": min(1000, int(limit))}
        if before:
            opts["before"] = before
        if until:
            opts["until"] = until
        return self.call("getSignaturesForAddress", [address, opts]) or []

    def iter_signatures(
        self,
        address: str,
        *,
        max_signatures: int = 10_000,
        page_size: int = 1000,
        until: str | None = None,
        sleep_s: float = 0.0,
    ) -> Iterator[dict]:
        before = None
        emitted = 0
        seen: set[str] = set()
        while emitted < max_signatures:
            page = self.recent_signatures(
                address,
                min(page_size, max_signatures - emitted),
                before=before,
                until=until,
            )
            if not page:
                break
            for item in page:
                sig = item.get("signature")
                if not sig or sig in seen:
                    continue
                seen.add(sig)
                emitted += 1
                yield item
                if emitted >= max_signatures:
                    break
            last = page[-1].get("signature")
            if not last or last == before or len(page) < min(page_size, max_signatures - emitted + len(page)):
                break
            before = last
            if sleep_s:
                time.sleep(sleep_s)

    def transaction(self, signature: str) -> dict | None:
        return self.call(
            "getTransaction",
            [signature, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
        )

    def wallet_transactions(self, address: str, limit: int = 100, sleep_s: float = 0.0) -> pd.DataFrame:
        rows = []
        for item in self.iter_signatures(address, max_signatures=limit, sleep_s=sleep_s):
            sig = item.get("signature")
            if not sig:
                continue
            rows.append({
                "wallet": address,
                "signature": sig,
                "slot": item.get("slot"),
                "block_time": item.get("blockTime"),
                "err": item.get("err"),
                "transaction": self.transaction(sig),
            })
            if sleep_s:
                time.sleep(sleep_s)
        return pd.DataFrame(rows)


@dataclass(slots=True)
class JupiterQuoteClient:
    """Read-only Jupiter quote adapter for shadow execution and impact checks."""

    base_url: str = "https://lite-api.jup.ag/swap/v1"

    def quote(
        self,
        *,
        input_mint: str,
        output_mint: str,
        amount_atomic: int,
        slippage_bps: int = 100,
    ) -> dict:
        query = urllib.parse.urlencode({
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": int(amount_atomic),
            "slippageBps": int(slippage_bps),
        })
        return _json_request(f"{self.base_url}/quote?{query}")

    def sol_usd_price(self, *, slippage_bps: int = 50) -> float:
        quote = self.quote(
            input_mint=SOL_MINT,
            output_mint=USDC_MINT,
            amount_atomic=LAMPORTS_PER_SOL,
            slippage_bps=slippage_bps,
        )
        out = int(quote.get("outAmount") or 0)
        if out <= 0:
            raise RuntimeError("Jupiter returned no executable SOL/USDC output")
        return out / USDC_ATOMIC


@dataclass(slots=True)
class PumpPortalStreamConfig:
    """Read-only Pump.fun/PumpSwap stream metadata; no signing or broadcasting."""

    websocket_url: str = "wss://pumpportal.fun/api/data"
    subscribe_new_token_method: str = "subscribeNewToken"
    subscribe_token_trade_method: str = "subscribeTokenTrade"
    subscribe_account_trade_method: str = "subscribeAccountTrade"
