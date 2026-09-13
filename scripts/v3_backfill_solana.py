from __future__ import annotations

import argparse
import os
import time

from crypto_pump.v3.solana import SolanaRPC
from crypto_pump.v3.store import V3Store


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill raw Solana transactions for programs or wallets into V3 SQLite")
    p.add_argument("--rpc-url", default=os.environ.get("SOLANA_RPC_URL"))
    p.add_argument("--store", default="data/v3/live.sqlite")
    p.add_argument("--address", action="append", required=True, help="Program or wallet address; repeatable")
    p.add_argument("--max-signatures", type=int, default=10_000)
    p.add_argument("--page-size", type=int, default=1000)
    p.add_argument("--sleep", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.rpc_url:
        raise SystemExit("--rpc-url or SOLANA_RPC_URL is required")
    rpc = SolanaRPC(args.rpc_url)
    store = V3Store(args.store)
    inserted = 0
    fetched = 0
    for address in args.address:
        for item in rpc.iter_signatures(
            address,
            max_signatures=args.max_signatures,
            page_size=args.page_size,
            sleep_s=args.sleep,
        ):
            signature = item.get("signature")
            if not signature:
                continue
            fetched += 1
            tx = rpc.transaction(signature)
            payload = {
                "event_type": "PROGRAM_TX",
                "program_or_wallet": address,
                "signature": signature,
                "slot": item.get("slot"),
                "blockTime": item.get("blockTime"),
                "err": item.get("err"),
                "transaction": tx,
            }
            if store.append_event(payload, source=f"solana_rpc:{address}"):
                inserted += 1
            if args.sleep:
                time.sleep(args.sleep)
    print({"fetched": fetched, "inserted": inserted, "store": args.store})


if __name__ == "__main__":
    main()
