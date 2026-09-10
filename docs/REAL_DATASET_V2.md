# REAL DATASET V2 — Uniswap + LATOKEN only

This pipeline intentionally contains only two market sources.

## LATOKEN CEX

The collector uses LATOKEN public API v2 directly:

- `/v2/currency` — currency id/tag mapping
- `/v2/pair` — pair universe
- `/v2/tradingview/history` — historical OHLCV at 5-minute resolution

The benchmark symbol is normalized to `LATOKEN:BTCUSDT`.

## Uniswap V3 DEX

Cryo collects canonical Ethereum logs from the Uniswap V3 factory and selected pools:

1. `PoolCreated` from the V3 factory.
2. Select pools where exactly one side is WETH, USDC, USDT or DAI.
3. `Swap` logs for those pools.
4. Block timestamps.
5. Decode raw ABI fields, orient every pool as token/quote and aggregate swaps to 5-minute OHLCV + buy/sell quote flow.

The workflow uses `ETH_RPC_URL` secret if available and otherwise a public RPC fallback. A private/archive RPC is strongly recommended for larger backfills.

## Envio

Envio HyperSync is an optional accelerator. Current HyperSync clients require `ENVIO_API_TOKEN`; therefore the reproducible no-secret path is Cryo. Add an `ENVIO_API_TOKEN` repository secret before enabling the Envio accelerator in large runs.

## Point-in-time rule

Sparse DEX series are explicitly reindexed to a true 5-minute grid before labels are created. Missing bars forward-fill price and have zero volume/trades. This prevents `+288 rows` from accidentally meaning much more than 24 hours for illiquid pools.

## Training

The GitHub Action runs:

`collect -> merge/regularize -> features-v2 -> labels-v2 -> purged walk-forward OOS -> final V2 fit -> artifact upload`.

The action is a research training pipeline, not an execution or market-manipulation system.
