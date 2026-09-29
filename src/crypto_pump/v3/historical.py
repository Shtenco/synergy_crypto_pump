from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator

import pandas as pd


@dataclass(slots=True)
class HeliusHistoricalClient:
    """Read-only Helius gTFA client for chronological Solana history replay.

    Uses Helius' getTransactionsForAddress method because it supports time/slot
    filters, oldest-first ordering and pagination tokens. No signing endpoints
    are exposed from this class.
    """

    api_key: str | None = None
    base_url: str = "https://mainnet.helius-rpc.com"
    timeout: int = 60

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("HELIUS_API_KEY")
        if not self.api_key:
            raise ValueError("HELIUS_API_KEY is required for Helius historical backfill")

    @property
    def url(self) -> str:
        return f"{self.base_url}/?api-key={self.api_key}"

    def call(self, method: str, params: list[Any]) -> Any:
        payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
        req = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "crypto-pump-v3/0.3"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if "error" in body:
            raise RuntimeError(f"Helius RPC error: {body['error']}")
        return body.get("result")

    def transactions_page(
        self,
        address: str,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
        pagination_token: str | None = None,
        full: bool = True,
        limit: int | None = None,
        status: str = "succeeded",
    ) -> dict[str, Any]:
        detail = "full" if full else "signatures"
        max_limit = 100 if full else 1000
        opts: dict[str, Any] = {
            "transactionDetails": detail,
            "sortOrder": "asc",
            "commitment": "finalized",
            "limit": min(max_limit, int(limit or max_limit)),
            "filters": {"status": status},
        }
        block_time: dict[str, int] = {}
        if start_time is not None:
            block_time["gte"] = int(start_time)
        if end_time is not None:
            block_time["lte"] = int(end_time)
        if block_time:
            opts["filters"]["blockTime"] = block_time
        if pagination_token:
            opts["paginationToken"] = pagination_token
        return self.call("getTransactionsForAddress", [address, opts]) or {"data": []}

    def iter_transactions(
        self,
        address: str,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
        full: bool = True,
        sleep_s: float = 0.05,
        max_pages: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        token: str | None = None
        pages = 0
        while True:
            page = self.transactions_page(
                address,
                start_time=start_time,
                end_time=end_time,
                pagination_token=token,
                full=full,
            )
            data = page.get("data") or []
            for tx in data:
                yield tx
            pages += 1
            token = page.get("paginationToken")
            if not token or not data or (max_pages is not None and pages >= max_pages):
                break
            if sleep_s:
                time.sleep(sleep_s)

    def to_frame(self, address: str, **kwargs) -> pd.DataFrame:
        rows = list(self.iter_transactions(address, **kwargs))
        if not rows:
            return pd.DataFrame()
        frame = pd.json_normalize(rows)
        frame["queried_address"] = address
        return frame


def utc_epoch(value: str | pd.Timestamp) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp())
