from __future__ import annotations

import subprocess
import tempfile
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


def _read_csv_tree(root: Path) -> pd.DataFrame:
    parts = []
    for f in sorted(root.rglob("*.csv")):
        try:
            d = pd.read_csv(f)
            if len(d):
                parts.append(d)
        except Exception:
            continue
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _run_cryo(args: list[str], *, rpc_url: str, cwd: Path) -> None:
    cmd = ["cryo", *args, "--rpc", rpc_url, "--requests-per-second", "3", "--max-concurrent-requests", "3", "--max-retries", "6"]
    subprocess.run(cmd, cwd=cwd, check=True)


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
            rows.append({"block_number": int(r["block_number"]), "token0": token0, "token1": token1, "fee": int(fee), "pool": pool})
        except Exception:
            continue
    return pd.DataFrame(rows).drop_duplicates("pool") if rows else pd.DataFrame()


def _decode_swaps(raw: pd.DataFrame, pools: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    if raw.empty or pools.empty or blocks.empty:
        return pd.DataFrame()
    addr_col = "address" if "address" in raw else "contract_address"
    bnum_col = "number" if "number" in blocks else "block_number"
    bmap = blocks[[bnum_col, "timestamp"]].drop_duplicates(bnum_col).rename(columns={bnum_col: "block_number"})
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
            ratio_1_per_0 = (sqrtp / float(2 ** 96)) ** 2
            t0, t1 = m["token0"], m["token1"]
            if t1 in QUOTES and t0 not in QUOTES:
                quote, qdec = QUOTES[t1]
                price = ratio_1_per_0
                token_amt, quote_amt = amount0, amount1
                token_addr = t0
            elif t0 in QUOTES and t1 not in QUOTES:
                quote, qdec = QUOTES[t0]
                price = 1.0 / ratio_1_per_0 if ratio_1_per_0 > 0 else np.nan
                token_amt, quote_amt = amount1, amount0
                token_addr = t1
            else:
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
        open=("price", "first"), high=("price", "max"), low=("price", "min"), close=("price", "last"),
        volume=("quote_volume", "sum"), quote_volume=("quote_volume", "sum"), trades=("price", "size"),
        buy_volume=("buy_q", "sum"), sell_volume=("sell_q", "sum"), liquidity=("liquidity", "last"),
        pool=("pool", "first"), token=("token", "first"), quote=("quote", "first"),
    ).reset_index().rename(columns={"bar": "timestamp"})
    bars["source"] = "uniswap"
    return bars


def collect_uniswap_v3_panel(*, rpc_url: str, lookback_blocks: int = 80_000, pool_lookback_blocks: int | None = None, max_pools: int = 100, safe_blocks: int = 64, workdir: str | Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    latest_hex = rpc_json(rpc_url, "eth_blockNumber")
    latest = int(latest_hex, 16)
    end = max(1, latest - safe_blocks)
    start = max(12_369_621, end - lookback_blocks)
    pool_start = max(12_369_621, end - int(pool_lookback_blocks or max(lookback_blocks * 4, 250_000)))
    root = Path(workdir or tempfile.mkdtemp(prefix="crypto-pump-cryo-"))
    root.mkdir(parents=True, exist_ok=True)
    pools_dir, swaps_dir, blocks_dir = root / "pools", root / "swaps", root / "blocks"
    for p in (pools_dir, swaps_dir, blocks_dir):
        p.mkdir(parents=True, exist_ok=True)

    _run_cryo(["logs", "--blocks", f"{pool_start}:{end}", "--contract", UNISWAP_V3_FACTORY, "--topic0", POOL_CREATED_TOPIC, "--event-signature", POOL_CREATED_SIG, "--hex", "--csv", "--output-dir", str(pools_dir), "--overwrite", "--inner-request-size", "1000"], rpc_url=rpc_url, cwd=root)
    pools = _decode_pools(_read_csv_tree(pools_dir))
    if pools.empty:
        return pd.DataFrame(), {"source": "uniswap_cryo", "error": "no PoolCreated rows decoded", "latest": latest, "start": start, "end": end}
    mask = pools["token0"].isin(QUOTES) ^ pools["token1"].isin(QUOTES)
    pools = pools[mask].sort_values("block_number", ascending=False)
    if max_pools > 0:
        pools = pools.head(max_pools)
    contracts = pools["pool"].tolist()
    if not contracts:
        return pd.DataFrame(), {"source": "uniswap_cryo", "error": "no quote-token pools", "latest": latest, "start": start, "end": end}

    _run_cryo(["logs", "--blocks", f"{start}:{end}", "--contract", *contracts, "--topic0", SWAP_TOPIC, "--event-signature", SWAP_SIG, "--hex", "--csv", "--output-dir", str(swaps_dir), "--overwrite", "--inner-request-size", "500"], rpc_url=rpc_url, cwd=root)
    _run_cryo(["blocks", "--blocks", f"{start}:{end}", "--include-columns", "timestamp", "--csv", "--output-dir", str(blocks_dir), "--overwrite"], rpc_url=rpc_url, cwd=root)
    swaps = _decode_swaps(_read_csv_tree(swaps_dir), pools, _read_csv_tree(blocks_dir))
    panel = _bars(swaps)
    report = {
        "source": "uniswap_cryo", "latest": latest, "start_block": start, "end_block": end,
        "pool_scan_start": pool_start, "selected_pools": len(pools), "raw_swaps": len(swaps),
        "bar_symbols": int(panel["symbol"].nunique()) if len(panel) else 0, "bar_rows": len(panel),
    }
    return panel, report
