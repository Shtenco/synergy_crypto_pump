# V3 historical data sources

V3 uses multiple independent sources so a profitable result cannot depend on one vendor or one snapshot.

## 1. PumpFun Launch-to-Graduation Corpus — bootstrap research set

Repository: `Slinky21/Pumpfun_Memecoin_Corpus` on Hugging Face.

Documented coverage at integration time:

- 798,430 Pump.fun launches;
- 33,581,765 trade rows;
- 26,934,864 pre-graduation bonding-curve snapshots;
- 1,392,133 post-graduation snapshots;
- 5,669 labeled post-graduation outcomes;
- fixed 39-day collection window, 2026-06-05 through 2026-07-14;
- approximately 6.7 GB across the published parquet corpus.

The corpus explicitly documents quality problems. V3 therefore does **not** train directly on all raw rows. `crypto_pump.v3.corpus` applies mandatory filters and writes a manifest/hash for every derived panel.

Key rules:

- exclude the documented System Program pseudo-wallet rows;
- reject missing/non-positive prices and SOL amounts;
- reject internally inconsistent SOL amount / token amount / price rows;
- exclude token rows flagged as suspect concentration where available;
- recompute wallet behavior causally from the trade ledger instead of trusting stale aggregate wallet statistics;
- keep documented platform regime breaks as metadata for slicing and falsification, not numeric alpha features.

Raw corpus data must live outside git under `data/` or another mounted data directory.

## 2. Helius getTransactionsForAddress — long historical extension

V3 includes a read-only Helius archival client using `getTransactionsForAddress` with:

- oldest-first (`sortOrder=asc`) replay;
- block-time range filters;
- succeeded-transaction filter;
- pagination tokens;
- full or signature-only modes.

The intended role is to extend beyond the fixed 39-day corpus, replay Pump/PumpSwap program history and fill gaps without changing the model interface.

Environment variable: `HELIUS_API_KEY`.

No signing or transaction submission functionality exists in this client.

## 3. Dune `dex_solana.trades` — independent execution validation

Dune's curated Solana DEX trade table contains raw trade segments across Solana DEX protocols, including timestamps, token mints, USD amounts, fees, program IDs, traders and transaction IDs.

V3 treats Dune as an independent control surface for:

- verifying post-graduation PumpSwap/Raydium/Jupiter execution paths;
- validating fee/route assumptions;
- checking volume and liquidity regimes independently from PumpPortal/Hugging Face;
- selecting adversarial holdout dates for OOS falsification.

Dune is not required for the core training run and is intentionally separated from the primary corpus.

## Validation hierarchy

A strategy may move from research toward canary only in this order:

1. cleaned fixed corpus OOS;
2. extended Helius historical OOS on later/earlier regimes;
3. independent Dune execution sanity checks;
4. live PumpPortal + Jupiter shadow trading;
5. 300+ closed shadow trades across >=30 days and all promotion gates;
6. tiny canary intents with absolute hard caps;
7. scale only after the separate canary-to-scale gate.

A strong score on the fixed corpus alone is never sufficient evidence for live capital.
