# CRYPTO PUMP — Unified ML Research Stack

Research monorepo for **early pump / abnormal-upside prediction** in low-cap cryptocurrencies and memecoins.

The goal is not to detect a pump after it is obvious. The core target is strictly forward-looking:

```text
P(max_return(symbol, t+1 ... t+H) >= R | information available at or before t)
```

and a cross-sectional ranking of the whole tradable universe at each timestamp.

> Research only. Pump-and-dump markets are highly manipulated and execution costs, slippage, delistings, survivorship bias and data leakage can destroy apparent backtest alpha. The repository deliberately separates prediction quality from trading PnL.

## Architecture

```text
CEX OHLCV/order flow ─┐
DEX/on-chain wallets ─┼─> feature store ─> horizon labels ─> models ─> calibrated ensemble
Social/news activity ─┘                                      │
                                                             v
                                                cross-sectional Top-K ranking
                                                             │
                                                             v
                                         walk-forward / cost-aware evaluation
```

The native `src/crypto_pump` layer is clean-room integration code. The 15 selected research repositories are pinned as Git submodules under `upstreams/`, so one recursive clone restores the exact upstream revisions selected for this project without copying or silently relicensing their source. `scripts/bootstrap_upstreams.py` is also provided for a separate ignored `external/upstreams/` research workspace and writes the resolved SHAs to a local lock file.

## Upstream research set (15)

1. `B0R0koko/pump-and-dump-prediction` — pre-pump cross-sectional target ranking and microstructure.
2. `Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency` — sequence-based target coin prediction.
3. `cakcora/Chainoba` — classical pump prediction / anomaly feature engineering.
4. `baseershah7/Solana_Memecoin_Graduation` — early Pump.fun graduation prediction.
5. `gayatri1505/Solana-Memcoin-Prediction-Using-Machine-Learning` — early-block Solana memecoin ensemble.
6. `Based-LTD/graduate-oracle` — graduation probability / wallet-quality methodology and public artifacts.
7. `Wendy1890/crypto_pump_predictor_bot` — multi-exchange ML scanner.
8. `Derposoft/crypto_pump_and_dump_with_deep_learning` — Conv-LSTM / Transformer anomaly models.
9. `uasth/CryptoAnomalyDetection` — high-frequency temporal anomaly detection.
10. `lidialosavio-dotcom/crypto_fraud_stGNN` — spatio-temporal graph learning.
11. `Confringo233/PumpWatcher` — graph + contrastive learning for DeFi pump/dump patterns.
12. `mfzhang/20260115-Pump_and_Dump_detect_signal` — market + social/news signal modelling.
13. `Mehrnoom/Cryptocurrency-Pump-Dump` — large social/Telegram research dataset and analysis.
14. `lodetomasi/the-intelligent-llm-investor` — social-momentum / coordination detection.
15. `RaibekTussupbekov/binance-pump-dump-detector` — Bayesian change-point regime detector.

See `upstreams.yaml` and `docs/UPSTREAMS.md`.

## Quick start

```bash
# Exact unified research tree including all 15 pinned upstreams
git clone --recurse-submodules https://github.com/Shtenco/crypto_pump.git
cd crypto_pump

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .

# Alternative: clone the same research sources into an ignored workspace
python scripts/bootstrap_upstreams.py

# Build leakage-safe features from a normalized CSV
crypto-pump features data/raw/market.csv data/processed/features.parquet

# Create future-horizon labels
crypto-pump labels data/processed/features.parquet data/processed/labeled.parquet --horizon 12 --threshold 0.20

# Train a baseline cross-sectional probability model
crypto-pump train data/processed/labeled.parquet artifacts/model.joblib

# Rank every symbol at the latest timestamp
crypto-pump rank data/processed/labeled.parquet artifacts/model.joblib artifacts/latest_ranking.csv --top-k 20
```

If the main repository was already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Canonical normalized schema

Minimum columns:

```text
timestamp,symbol,open,high,low,close,volume
```

Optional columns are automatically consumed when available:

```text
trades,buy_volume,sell_volume,bid_depth,ask_depth,spread,
unique_buyers,unique_sellers,holders,liquidity,
social_mentions,social_users,creator_success,top5_share,sniper_share
```

## Leakage rules

1. Features at time `t` may use only observations with timestamp `<= t`.
2. Future windows are used only by the label/evaluation layer.
3. Model selection must be chronological/walk-forward, never random row shuffle across time.
4. Cross-sectional ranking is computed independently inside each timestamp.

## Default feature families

- Price/volatility: returns, realized volatility, ATR proxy, range, drawdown, compression.
- Volume/flow: volume z-score, acceleration, trade acceleration, buy/sell imbalance.
- Liquidity: spread, bid/ask depth imbalance, liquidity growth.
- On-chain: buyer/holder velocity, concentration, creator history, sniper fraction.
- Social: mention velocity/acceleration, unique-user velocity and market/social divergence.
- Regime: relative strength vs universe, BTC-relative features when supplied, change intensity.

## Targets

A single dataset can be labelled at several horizons, for example:

```text
+10% within 15m
+20% within 1h
+50% within 6h
+100% within 24h
```

For each horizon the system can also estimate adverse excursion before target so that a high pump probability is not automatically treated as a tradable signal.

## Evaluation

Primary prediction metrics:

```text
PR-AUC, ROC-AUC, Brier score, log loss,
Precision@1, Precision@5, Precision@10, Recall@K, NDCG@K
```

Trading evaluation should additionally include fees, spread, slippage, market impact, latency, rejected fills and delisted/dead tokens.

## Repository layout

```text
configs/                 model/label configuration
src/crypto_pump/         unified clean-room ML pipeline
upstreams/               15 pinned Git submodules
scripts/                 bootstrap and research utilities
docs/                    source map and methodology
external/upstreams/      optional cloned research workspace (gitignored)
data/                    local datasets (gitignored)
artifacts/               local models/reports (gitignored)
tests/                   leakage and feature tests
```

## Status

V1 contains the 15 pinned upstream repositories plus a native leakage-safe feature/label/ranking core. The next stage is to add source-specific adapters, walk-forward ensemble calibration, graph embeddings, social encoders and exchange/DEX collectors.
