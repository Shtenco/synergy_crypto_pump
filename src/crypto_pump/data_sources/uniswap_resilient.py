from __future__ import annotations

from typing import Any

from .common import rpc_json
from . import uniswap_cryo as _impl


def _token_decimals_resilient(
    rpc_url: str,
    tokens: list[str],
    batch_size: int = 40,
) -> tuple[dict[str, int], list[str]]:
    """Resolve ERC-20 decimals with batch first and verified single-call fallback.

    Several public Ethereum RPCs accept eth_getLogs and single eth_call but reject
    JSON-RPC batches. The original collector treated a rejected batch as if every
    token lacked decimals, which could erase an otherwise valid pool universe.
    """
    out: dict[str, int] = {a: meta[1] for a, meta in _impl.QUOTES.items()}
    failed: list[str] = []
    todo = sorted({t.lower() for t in tokens if t and t.lower() not in out})

    for i in range(0, len(todo), max(1, batch_size)):
        chunk = todo[i:i + max(1, batch_size)]
        calls = [
            ("eth_call", [{"to": token, "data": _impl.DECIMALS_SELECTOR}, "latest"])
            for token in chunk
        ]
        try:
            results: list[Any] = _impl._rpc_batch(rpc_url, calls, timeout=45, retries=2)
        except Exception:
            results = [None] * len(chunk)

        for pos, token in enumerate(chunk):
            raw = results[pos] if pos < len(results) else None
            if raw is None:
                try:
                    raw = rpc_json(
                        rpc_url,
                        "eth_call",
                        [{"to": token, "data": _impl.DECIMALS_SELECTOR}, "latest"],
                        timeout=30,
                        retries=3,
                    )
                except Exception:
                    raw = None
            try:
                dec = int(str(raw), 16)
                if 0 <= dec <= 36:
                    out[token] = dec
                else:
                    failed.append(token)
            except Exception:
                failed.append(token)
    return out, failed


def collect_uniswap_v3_panel(**kwargs):
    # Patch only the decimals resolver; all price/event decoding remains in the
    # audited native collector. This can be removed once the implementation is
    # folded back into uniswap_cryo.py.
    original = _impl._token_decimals
    try:
        _impl._token_decimals = _token_decimals_resilient
        return _impl.collect_uniswap_v3_panel(**kwargs)
    finally:
        _impl._token_decimals = original
