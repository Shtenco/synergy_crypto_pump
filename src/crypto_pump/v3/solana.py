from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

import pandas as pd


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

    def recent_signatures(self, address: str, limit: int = 100) -> list[dict]:
        return self.call("getSignaturesForAddress", [address, {"limit": int(limit)}]) or []

    def transaction(self, signature: str) -> dict | None:
        return self.call(
            "getTransaction",
            [signature, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
        )

    def wallet_transactions(self, address: str, limit: int = 100, sleep_s: float = 0.0) -> pd.DataFrame:
        rows = []
        for item in self.recent_signatures(address, limit=limit):
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


@dataclass(slots=True)
class PumpPortalStreamConfig:
    """Read-only Pump.fun/PumpSwap stream metadata; no signing or broadcasting."""

    websocket_url: str = "wss://pumpportal.fun/api/data"
    subscribe_new_token_method: str = "subscribeNewToken"
    subscribe_token_trade_method: str = "subscribeTokenTrade"
    subscribe_account_trade_method: str = "subscribeAccountTrade"
