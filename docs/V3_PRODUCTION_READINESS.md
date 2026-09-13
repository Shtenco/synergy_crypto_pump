# CRYPTO PUMP V3 — execution, risk and shadow validation

V3 is additive: V2 remains the predictive research stack. V3 answers the next
question: **does a V2 signal survive realistic execution costs and risk gates?**

## New gates before live capital

1. Triple-barrier first-passage labels: TP must occur before SL.
2. Delayed entry: no impossible fill at the same close that generated a signal.
3. Execution model: DEX fee, gas, slippage, notional/liquidity impact, MEV haircut,
   expected failed-transaction cost and configurable latency.
4. Risk manager: fixed cap, equity-fraction cap, daily loss, portfolio exposure,
   liquidity, impact, slippage, stale-data/model-health and kill switch.
5. Shadow executor: writes intended orders to an append-only JSONL journal and
   never signs or broadcasts a transaction.
6. Financial validation: net PnL, expectancy, win rate, profit factor, drawdown,
   per-trade Sharpe, capacity curve and winner-concentration stress.
7. Stress runs: costs x2, latency x2, both x2, plus removal of best 1, best 3,
   and top 1% of winning trades.
8. Read-only Solana adapters: RPC wallet-history collection and Jupiter quotes.
   Signing keys are intentionally out of scope until shadow/live parity is proven.

## Required promotion sequence

`V2 OOS -> V3 execution backtest -> 30d shadow -> tiny fixed-size canary -> scale`

A live canary is allowed only if:

- at least 300 shadow trades and at least 30 shadow days are collected;
- aggregate OOS/shadow net expectancy is positive after base costs;
- profit factor is at least 1.05;
- costs x2 does not make aggregate PnL negative;
- removing the top 1% winners does not make aggregate PnL negative;
- capacity curve identifies a positive range for the requested notional;
- shadow quote-vs-observed-fill drift stays below the configured tolerance;
- no risk/kill-switch violations occur;
- CI is green on the exact commit.

These gates are codified in `v3/promotion.py` rather than being an informal checklist.

## CI note

The pre-V3 `main` Action failure happened before any workflow step: GitHub
reported empty `steps` and `runner_id=0` for both jobs. Treat this as a runner /
Actions infrastructure gate, not evidence that pytest failed. Promotion remains
blocked until GitHub allocates a runner and the exact V3 commit passes tests.

## Solana scope

`v3/solana.py` is read-only by design. It includes:

- Solana JSON-RPC signature and transaction collection;
- Jupiter quote retrieval for shadow price-impact checks;
- PumpPortal websocket endpoint metadata for a dedicated collector.

The next dataset milestone is a point-in-time Solana launch table containing token
creation, Pump.fun/PumpSwap trades, Raydium/Jupiter executable quote snapshots,
wallet edges, liquidity and creator history. Raw ingest should be immutable and
features must use observed timestamps only.

## Longer point-in-time data

The existing 14-day/short-lookback real-data job remains useful as a smoke test,
but it is not sufficient evidence for capital. Production research should collect
rolling immutable shards covering multiple market regimes, preferably 3-6 months
for a first serious decision and longer when available.

## Canary rule

V3 deliberately does not include a transaction signer. A canary executor should be
added only after promotion gates pass. The first canary should use fixed, very small
notional and the exact same signal, quote, risk and journal path as shadow mode so
shadow-vs-live drift can be measured instead of guessed.
