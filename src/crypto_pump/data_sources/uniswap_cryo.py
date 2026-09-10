from __future__ import annotations

import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import rpc_json

UNISWAP_V3_FACTORY = "0x1f98431c8ad98523631ae4a59f267346ea31f984"
POOL_CREATED_TOPIC = "0x783cca1c0412dd0d695e784568c96da2e9c22ff989357a2e8b1d9b2b4e6b7118"
SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
POOL_CREATED_SIG = "PoolCreated(address indexed token0, address indexed token1, uint24 indexed fee, int24 tickSpacing, address pool)"
SWAP_SIG = "Swap(address indexed sender, address indexed recipient, int256 amount0, int256 amount1, uint160 sqrtPriceX96, uint128 liquidity, int24 tick)"

QUOTES = {
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": ("WETH", 18),
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": ("USDC", 6),
    "0xdac17f958d2ee523a2206206994597c13d831ec7": ("USDT", 6),
    "0x6b175474e89094c44da98b954eedeac495271d0f": ("DAI", 18),
}
DECIMALS_SELECTOR = "0x313ce567"


def _clean_hex(x: Any) -> str:
    if isinstance(x, bytes):
        return "0x" + x.hex()
    s = str(x or "").lower()
    return s if s.startswith("0x") else "0x" + s


def _word(data: str, i: int) -> str:
    h = _clean_hex(data)[2:]
    return h[i * 64:(i + 1) * 64].rjust(64, "0")


def _address_from_word(w: str) -> str:
    return "0x" + w[-40:].lower()


def _address_from_topic(x: Any) -> str:
    return "0x" + _clean_hex(x)[2:][-40:].lower()


def _uint_word(w: str) -> int:
    return int(w or "0", 16)


def _int_word(w: str, bits: int = 256) -> int:
    u = int(w or "0", 16)
    return u - (1 << bits) if u >= (1 << (bits - 1)) else u


def _hex_int(x: Any, default: int = 0) -> int:
    if x is None:
        return default
    try:
        s = str(x)
        return int(s, 16) if s.startswith("0x") else int(s)
    except Exception:
        return default


def _read_csv_tree(root: Path) -> pd.DataFrame:
    parts = []
    for f in sorted(root.rglob("*.csv")):
        try:
            d = pd.read_csv(f)
            if len(d):
                parts.append(d)
        except Exception:
            continue
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def _run_cryo(args: list[str], *, rpc_url: str, cwd: Path) -> None:
    """Optional Cryo accelerator retained for research/debugging.

    Production public-RPC collection below uses adaptive direct eth_getLogs,
    because public providers frequently reject Cryo's large batched log scans.
    """
    cmd = [
        "cryo", *args,
        "--rpc", rpc_url,
        "--requests-per-second", "4",
        "--max-concurrent-requests", "4",
        "--max-retries", "8",
    ]
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode:
        tail = (proc.stdout or "")[-6000:]
        raise RuntimeError(f"Cryo failed ({proc.returncode}): {' '.join(cmd[:8])}\n{tail}")


def _raw_log_row(log: dict[str, Any]) -> dict[str, Any]:
    topics = list(log.get("topics") or [])
    return {
        "block_number": _hex_int(log.get("blockNumber")),
        "transaction_index": _hex_int(log.get("transactionIndex")),
        "log_index": _hex_int(log.get("logIndex")),
        "address": str(log.get("address") or "").lower(),
        "topic0": topics[0] if len(topics) > 0 else None,
        "topic1": topics[1] if len(topics) > 1 else None,
        "topic2": topics[2] if len(topics) > 2 else None,
        "topic3": topics[3] if len(topics) > 3 else None,
        "data": log.get("data") or "0x",
    }


def _get_logs_recursive(
    rpc_url: str,
    start: int,
    end: int,
    addresses: list[str],
    topic0: str,
    *,
    timeout: int = 60,
) -> list[dict[str, Any]]:
    """Fetch logs and recursively split requests rejected by provider limits."""
    if end < start or not addresses:
        return []
    filt: dict[str, Any] = {
        "fromBlock": hex(int(start)),
        "toBlock": hex(int(end)),
        "address": addresses[0] if len(addresses) == 1 else addresses,
        "topics": [topic0],
    }
    try:
        result = rpc_json(rpc_url, "eth_getLogs", [filt], timeout=timeout, retries=2)
        if result is None:
            return []
        if not isinstance(result, list):
            raise RuntimeError(f"eth_getLogs returned {type(result).__name__}")
        return [x for x in result if isinstance(x, dict)]
    except Exception:
        # Prefer splitting the block span first: most public RPC limits are
        # expressed as max block range / response size. If one block still
        # fails, split the address list. Never silently drop a request.
        if start < end:
            mid = (start + end) // 2
            return (
                _get_logs_recursive(rpc_url, start, mid, addresses, topic0, timeout=timeout)
                + _get_logs_recursive(rpc_url, mid + 1, end, addresses, topic0, timeout=timeout)
            )
        if len(addresses) > 1:
            mid = len(addresses) // 2
            return (
                _get_logs_recursive(rpc_url, start, end, addresses[:mid], topic0, timeout=timeout)
                + _get_logs_recursive(rpc_url, start, end, addresses[mid:], topic0, timeout=timeout)
            )
        raise


def _collect_logs_adaptive(
    rpc_url: str,
    *,
    start: int,
    end: int,
    addresses: list[str],
    topic0: str,
    block_chunk: int = 2_000,
    address_batch: int = 25,
) -> pd.DataFrame:
    """Bound request sizes up-front, then recursively shrink on RPC rejection."""
    rows: list[dict[str, Any]] = []
    unique_addresses = sorted({a.lower() for a in addresses if a})
    for ai in range(0, len(unique_addresses), max(1, address_batch)):
        addr_chunk = unique_addresses[ai:ai + max(1, address_batch)]
        cur = int(start)
        while cur <= int(end):
            stop = min(int(end), cur + max(1, block_chunk) - 1)
            logs = _get_logs_recursive(rpc_url, cur, stop, addr_chunk, topic0)
            rows.extend(_raw_log_row(x) for x in logs)
            cur = stop + 1
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.drop_duplicates(["block_number", "transaction_index", "log_index", "address"], keep="last")


def _decode_pools(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    rows = []
    for _, r in raw.iterrows():
        try:
            token0 = str(r.get("token0") or "").lower()
            token1 = str(r.get("token1") or "").lower()
            pool = str(r.get("pool") or "").lower()
            fee = r.get("fee")
            if not token0:
                token0 = _address_from_topic(r["topic1"])
            if not token1:
                token1 = _address_from_topic(r["topic2"])
            if not pool:
                pool = _address_from_word(_word(r["data"], 1))
            if fee is None or (isinstance(fee, float) and np.isnan(fee)):
                fee = int(_clean_hex(r["topic3"]), 16)
            rows.append({
                "block_number": int(r["block_number"]),
                "token0": token0,
                "token1": token1,
                "fee": int(fee),
                "pool": pool,
            })
        except Exception:
            continue
    return pd.DataFrame(rows).drop_duplicates("pool") if rows else pd.DataFrame()


def _rpc_batch(rpc_url: str, calls: list[tuple[str, list[Any]]], *, timeout: int = 60, retries: int = 4) -> list[Any]:
    if not calls:
        return []
    body = [
        {"jsonrpc": "2.0", "id": i, "method": method, "params": params}
        for i, (method, params) in enumerate(calls)
    ]
    data = json.dumps(body).encode()
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            rpc_url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "crypto-pump-v2/0.5"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read().decode("utf-8"))
            if not isinstance(payload, list):
                raise RuntimeError(f"RPC batch returned {type(payload).__name__}")
            by_id = {int(x.get("id")): x for x in payload if isinstance(x, dict) and "id" in x}
            out = []
            for i in range(len(calls)):
                row = by_id.get(i, {})
                if row.get("error"):
                    out.append(None)
                else:
                    out.append(row.get("result"))
            return out
        except Exception as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(min(8.0, 2.0 ** attempt))
    raise RuntimeError(f"RPC batch failed after {retries} attempts: {last}")


def _token_decimals(rpc_url: str, tokens: list[str], batch_size: int = 40) -> tuple[dict[str, int], list[str]]:
    out: dict[str, int] = {a: meta[1] for a, meta in QUOTES.items()}
    failed: list[str] = []
    todo = sorted({t.lower() for t in tokens if t and t.lower() not in out})
    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        calls = [("eth_call", [{"to": t, "data": DECIMALS_SELECTOR}, "latest"]) for t in chunk]
        try:
            results = _rpc_batch(rpc_url, calls)
        except Exception:
            results = [None] * len(chunk)
        for token, raw in zip(chunk, results):
            try:
                dec = int(str(raw), 16)
                if 0 <= dec <= 36:
                    out[token] = dec
                else:
                    failed.append(token)
            except Exception:
                failed.append(token)
    return out, failed


def _block_timestamps(rpc_url: str, block_numbers: list[int], batch_size: int = 50) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    nums = sorted({int(x) for x in block_numbers})
    for i in range(0, len(nums), batch_size):
        chunk = nums[i:i + batch_size]
        calls = [("eth_getBlockByNumber", [hex(n), False]) for n in chunk]
        try:
            results = _rpc_batch(rpc_url, calls, timeout=90, retries=5)
        except Exception:
            # Some public providers reject JSON-RPC batches. Preserve correctness
            # by falling back to single retry-safe calls rather than dropping time.
            results = []
            for n in chunk:
                try:
                    results.append(rpc_json(rpc_url, "eth_getBlockByNumber", [hex(n), False], timeout=60, retries=4))
                except Exception:
                    results.append(None)
        for n, block in zip(chunk, results):
            if isinstance(block, dict) and block.get("timestamp"):
                rows.append({"block_number": n, "timestamp": int(block["timestamp"], 16)})
    return pd.DataFrame(rows)


def _decode_swaps(raw: pd.DataFrame, pools: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    if raw.empty or pools.empty or blocks.empty:
        return pd.DataFrame()
    addr_col = "address" if "address" in raw else "contract_address"
    bmap = blocks[["block_number", "timestamp"]].drop_duplicates("block_number")
    meta = pools.set_index("pool").to_dict("index")
    rows = []
    for _, r in raw.iterrows():
        try:
            pool = _clean_hex(r[addr_col]).lower()
            m = meta.get(pool)
            if not m:
                continue
            amount0 = r.get("amount0")
            amount1 = r.get("amount1")
            sqrtp = r.get("sqrtPriceX96")
            liq = r.get("liquidity")
            if amount0 is None or (isinstance(amount0, float) and np.isnan(amount0)):
                amount0 = _int_word(_word(r["data"], 0))
                amount1 = _int_word(_word(r["data"], 1))
                sqrtp = _uint_word(_word(r["data"], 2))
                liq = _uint_word(_word(r["data"], 3))
            amount0, amount1, sqrtp = int(amount0), int(amount1), int(sqrtp)
            raw_ratio_1_per_0 = (sqrtp / float(2 ** 96)) ** 2
            t0, t1 = m["token0"], m["token1"]
            d0, d1 = int(m["decimals0"]), int(m["decimals1"])

            if t1 in QUOTES and t0 not in QUOTES:
                quote, qdec = QUOTES[t1]
                price = raw_ratio_1_per_0 * (10.0 ** (d0 - d1))
                token_amt, quote_amt = amount0, amount1
                token_addr = t0
            elif t0 in QUOTES and t1 not in QUOTES:
                quote, qdec = QUOTES[t0]
                price = (1.0 / raw_ratio_1_per_0) * (10.0 ** (d1 - d0)) if raw_ratio_1_per_0 > 0 else np.nan
                token_amt, quote_amt = amount1, amount0
                token_addr = t1
            else:
                continue
            if not np.isfinite(price) or price <= 0:
                continue
            side = "buy" if token_amt < 0 else "sell"
            qvol = abs(float(quote_amt)) / (10 ** qdec)
            rows.append({
                "block_number": int(r["block_number"]),
                "transaction_index": int(r.get("transaction_index", 0) or 0),
                "log_index": int(r.get("log_index", 0) or 0),
                "pool": pool,
                "token": token_addr,
                "quote": quote,
                "symbol": f"UNISWAP:{token_addr}/{quote}:{pool[:10]}",
                "price": float(price),
                "quote_volume": qvol,
                "side": side,
                "liquidity": float(liq or 0),
            })
        except Exception:
            continue
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d = d.merge(bmap, on="block_number", how="left")
    d["timestamp"] = pd.to_datetime(pd.to_numeric(d["timestamp"], errors="coerce"), unit="s", utc=True)
    return d.dropna(subset=["timestamp", "price"])


def _bars(swaps: pd.DataFrame, min_swaps: int = 20) -> pd.DataFrame:
    if swaps.empty:
        return pd.DataFrame()
    keep = swaps.groupby("symbol").size()
    keep = set(keep[keep >= min_swaps].index)
    d = swaps[swaps["symbol"].isin(keep)].copy()
    if d.empty:
        return d
    d = d.sort_values(["symbol", "timestamp", "block_number", "transaction_index", "log_index"])
    d["bar"] = d["timestamp"].dt.floor("5min")
    d["buy_q"] = np.where(d["side"].eq("buy"), d["quote_volume"], 0.0)
    d["sell_q"] = np.where(d["side"].eq("sell"), d["quote_volume"], 0.0)
    bars = d.groupby(["symbol", "bar"], sort=True).agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        quote_volume=("quote_volume", "sum"),
        trades=("price", "size"),
        buy_volume=("buy_q", "sum"),
        sell_volume=("sell_q", "sum"),
        liquidity=("liquidity", "last"),
        pool=("pool", "first"),
        token=("token", "first"),
        quote=("quote", "first"),
    ).reset_index().rename(columns={"bar": "timestamp"})
    bars["volume"] = bars["quote_volume"] / bars["close"].replace(0.0, np.nan)
    bars["volume"] = bars["volume"].fillna(0.0)
    bars["source"] = "uniswap"
    return bars


def collect_uniswap_v3_panel(
    *,
    rpc_url: str,
    lookback_blocks: int = 80_000,
    end_block_offset: int = 0,
    pool_lookback_blocks: int | None = None,
    max_pools: int = 100,
    safe_blocks: int = 64,
    workdir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collect one historical Uniswap V3 shard with adaptive direct JSON-RPC.

    Pool discovery and Swap retrieval are chunked and recursively split when a
    public RPC rejects a block/address range. `end_block_offset` permits
    non-overlapping GitHub Actions shards. Token decimals are fetched on-chain
    before converting sqrtPriceX96 to human quote/token prices.
    """
    latest_hex = rpc_json(rpc_url, "eth_blockNumber", timeout=30, retries=4)
    latest = int(latest_hex, 16)
    canonical_end = max(1, latest - safe_blocks)
    end = max(12_369_622, canonical_end - max(0, end_block_offset))
    start = max(12_369_621, end - max(1, lookback_blocks))
    pool_span = int(pool_lookback_blocks or max(lookback_blocks * 6, 500_000))
    pool_start = max(12_369_621, end - pool_span)

    root = Path(workdir or tempfile.mkdtemp(prefix="crypto-pump-uniswap-"))
    root.mkdir(parents=True, exist_ok=True)

    pools_raw = _collect_logs_adaptive(
        rpc_url,
        start=pool_start,
        end=end,
        addresses=[UNISWAP_V3_FACTORY],
        topic0=POOL_CREATED_TOPIC,
        block_chunk=2_000,
        address_batch=1,
    )
    pools = _decode_pools(pools_raw)
    if pools.empty:
        return pd.DataFrame(), {
            "source": "uniswap_cryo", "error": "no PoolCreated rows decoded",
            "collector_mode": "direct_eth_getLogs_adaptive",
            "latest": latest, "start_block": start, "end_block": end,
            "pool_scan_start": pool_start, "pool_raw_rows": len(pools_raw),
        }

    mask = pools["token0"].isin(QUOTES) ^ pools["token1"].isin(QUOTES)
    pools = pools[mask].sort_values("block_number", ascending=False).copy()
    decimals, failed_decimals = _token_decimals(
        rpc_url, pools["token0"].tolist() + pools["token1"].tolist()
    )
    pools["decimals0"] = pools["token0"].map(decimals)
    pools["decimals1"] = pools["token1"].map(decimals)
    pools = pools.dropna(subset=["decimals0", "decimals1"])
    if max_pools > 0:
        pools = pools.head(max_pools)
    contracts = pools["pool"].tolist()
    if not contracts:
        return pd.DataFrame(), {
            "source": "uniswap_cryo", "error": "no quote-token pools with resolvable decimals",
            "collector_mode": "direct_eth_getLogs_adaptive",
            "latest": latest, "start_block": start, "end_block": end,
            "pool_scan_start": pool_start, "failed_decimals": len(failed_decimals),
        }

    swaps_raw = _collect_logs_adaptive(
        rpc_url,
        start=start,
        end=end,
        addresses=contracts,
        topic0=SWAP_TOPIC,
        block_chunk=2_000,
        address_batch=25,
    )
    if swaps_raw.empty:
        return pd.DataFrame(), {
            "source": "uniswap_cryo", "error": "no Swap rows returned",
            "collector_mode": "direct_eth_getLogs_adaptive",
            "latest": latest, "start_block": start, "end_block": end,
            "pool_scan_start": pool_start, "selected_pools": len(pools),
            "pool_raw_rows": len(pools_raw),
        }

    block_numbers = pd.to_numeric(swaps_raw.get("block_number"), errors="coerce").dropna().astype(int).unique().tolist()
    blocks = _block_timestamps(rpc_url, block_numbers)
    swaps = _decode_swaps(swaps_raw, pools, blocks)
    panel = _bars(swaps)
    report = {
        "source": "uniswap_cryo",
        "collector_mode": "direct_eth_getLogs_adaptive",
        "latest": latest,
        "canonical_end": canonical_end,
        "start_block": start,
        "end_block": end,
        "end_block_offset": int(end_block_offset),
        "pool_scan_start": pool_start,
        "pool_raw_rows": len(pools_raw),
        "selected_pools": len(pools),
        "failed_token_decimals": len(failed_decimals),
        "raw_swap_rows": len(swaps_raw),
        "decoded_swaps": len(swaps),
        "timestamped_blocks": len(blocks),
        "bar_symbols": int(panel["symbol"].nunique()) if len(panel) else 0,
        "bar_rows": len(panel),
    }
    return panel, report
