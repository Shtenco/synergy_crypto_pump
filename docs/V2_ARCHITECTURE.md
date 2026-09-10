# CRYPTO PUMP V2 architecture

## One model, independent branches

```text
point-in-time market panel ──> V2FeatureBuilder ───────────────┐
                                                              │
                           ┌─> multi-horizon CatBoost ---------┤
                           ├─> CatBoost YetiRank / NDCG -------┤
past-only sequences ───────┼─> causal Temporal Transformer ----┤
wallet/token edge stream ──┼─> causal Wallet↔Token GNN --------┤
social/news aggregates ────┼─> social-only classifier ---------┤
BTC-relative residual ─────┴─> BOCD change probability --------┤
                                                              v
                                               validation-only stacker
                                                              │
                                               isotonic calibration
                                                              │
                                                              v
                              P(+10%/15m), P(+20%/1h),
                              P(+50%/6h), P(+100%/24h)
                                                              │
                                                              v
                                          per-timestamp Top-K ranking
```

The branches are intentionally separable. `oos_predictions.parquet` contains their individual scores so a researcher can remove a branch and measure the change instead of hiding all signal sources in one black box.

## Branch 1 — tabular boosting

`MultiHorizonTabularBranch` trains one probability model per horizon. With the `boosting`/`v2` extra installed it uses CatBoost; otherwise a scikit-learn histogram-gradient fallback makes the pipeline reproducible in CPU-only CI.

## Branch 2 — cross-sectional ranker

`CatBoostRankerBranch` groups all candidates at the same timestamp and optimizes a YetiRank/NDCG objective. Its relevance label is the within-timestamp percentile of future MFE, higher-is-better. The fallback is a histogram gradient regressor over the same relevance target.

This branch answers a different question than the binary classifier: given everything tradable now, which coins should be at the top of the list?

## Branch 3 — causal Temporal Transformer

`TemporalTransformerBranch` builds fixed-length sequences independently per symbol. Each training sequence ends at its prediction row. The encoder receives an upper-triangular causal mask, so attention cannot see a later element even inside a batch.

One shared representation feeds four horizon logits. This is multi-task learning: short and long pump horizons can share temporal structure while retaining separate probabilities.

Install with `pip install -e '.[deep,boosting]'`.

## Branch 4 — wallet/transaction GNN

`build_wallet_token_snapshot()` creates a bipartite wallet↔token graph for a given `as_of` timestamp and filters `timestamp <= as_of` before creating nodes or edges. Wallet nodes contain activity/amount/buy-rate/smart-score fields; token nodes contain activity/amount/buy-rate/unique-wallet summaries.

`WalletTokenGNN` performs two mean-neighbor message-passing layers and emits the four horizon logits. `WalletGNNBranch` trains over historical graph snapshots. An edge table requires `timestamp,wallet,symbol` and can additionally provide `amount,direction,wallet_smart_score`.

## Branch 5 — social-only model

Social information is not silently mixed into market features. `SocialBranch` selects the social/sentiment/coordination family and produces its own per-horizon score. This makes social alpha measurable in isolation.

## BOCD regime score

V2 runs a bounded Bayesian online changepoint process over a causal standardized market residual. If BTC is present, `market_residual_1 = asset_return_1 - BTC_return_1`. BOCD is not asked to predict the full pump; it answers the narrower question of whether the asset has just entered a statistically different regime relative to the market.

## Stacking and calibration

For each horizon the meta-model receives available branch scores. It is fitted only on the validation interval. If there is enough positive/negative support, the resulting probability is calibrated by isotonic regression, inspired by the public Graduate Oracle methodology.

No test prediction participates in fitting the base branches, stacker or calibrator.

## Purged walk-forward OOS

For the default 24-hour longest target on 5-minute bars:

```text
TRAIN | 288-bar purge | VALIDATION | 288-bar purge + embargo | TEST
```

Later folds expand the train window. Every test row is therefore produced by a model whose complete information set predates the test interval.

Run:

```bash
crypto-pump features-v2 data/raw/market.parquet data/processed/features_v2.parquet
crypto-pump labels-v2 data/processed/features_v2.parquet data/processed/labeled_v2.parquet
crypto-pump walk-forward-v2 data/processed/labeled_v2.parquet artifacts/v2_oos --model-kind auto

crypto-pump walk-forward-v2 \
  data/processed/labeled_v2.parquet artifacts/v2_oos_deep \
  --transactions data/processed/wallet_edges.parquet \
  --enable-temporal --enable-gnn
```

Artifacts are `fold_N_model.joblib`, `oos_predictions.parquet` and `oos_metrics.json`. Primary metrics per horizon are PR-AUC, ROC-AUC, Brier score, log loss, Precision@K and NDCG@K.
