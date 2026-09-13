from __future__ import annotations

import argparse
import asyncio

from crypto_pump.v3.store import V3Store
from crypto_pump.v3.stream import PumpStreamCollector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect Pump.fun/PumpSwap events into the V3 SQLite store")
    p.add_argument("--store", default="data/v3/live.sqlite")
    p.add_argument("--max-active-mints", type=int, default=200)
    p.add_argument("--token-ttl-seconds", type=int, default=6 * 3600)
    return p.parse_args()


async def main_async() -> None:
    args = parse_args()
    store = V3Store(args.store)
    collector = PumpStreamCollector(
        store=store,
        max_active_mints=args.max_active_mints,
        token_ttl_seconds=args.token_ttl_seconds,
    )
    stats = await collector.run()
    print(stats)


if __name__ == "__main__":
    asyncio.run(main_async())
