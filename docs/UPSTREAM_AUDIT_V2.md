# CRYPTO PUMP V2 — upstream source audit and distillation

Date: 2026-09-10. Upstreams are pinned in `.gitmodules` / `upstreams/`.

This document records what is actually distilled into the native V2 implementation. We do not assume that an upstream backtest is causal simply because its model is useful. Code, features, labels and validation are audited separately.

| # | Upstream | Distilled into V2 | What we deliberately do not inherit |
|---|---|---|---|
| 1 | B0R0koko/pump-and-dump-prediction | Cross-sectional ranking; YetiRank/NDCG; return/quote z-scores; long-trade share; power-law/slippage/flow imbalance; number of trades; prior-pump context; Top-K evaluation | Any feature unavailable at inference time; V2 fixes relevance direction explicitly (higher future excursion = higher relevance) |
| 2 | Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency | Historical target/pump sequence context; positional/target-aware attention concept | TensorFlow-1 implementation; V2 uses a modern causal Transformer over past-only sequences |
| 3 | cakcora/Chainoba | Classical coin/exchange/pump-history factors and class-imbalance awareness | Whole-dataset shuffle + shuffled StratifiedKFold; not valid for forward trading OOS |
| 4 | baseershah7/Solana_Memecoin_Graduation | tx/sec; time-decayed activity; creator reputation; wallet Gini/top-wallet share; virtual liquidity; flow imbalance; gas-efficiency idea | Random stratified CV for a time-evolving trading problem |
| 5 | gayatri1505/Solana-Memcoin-Prediction-Using-Machine-Learning | buy/sell counts; amount/virtual balance; first/last buy/sell timing; creator deployment count; ensemble/calibration idea | Random train/test and shuffled folds; README claims are separated from what `main.py` actually implements |
| 6 | Based-LTD/graduate-oracle | current liquidity/growth; unique buyers; top buyer concentration; smart-wallet/bot fields; validation-only isotonic calibration; forward-prediction ledger discipline | Closed-source score/thresholds; V2 is a clean-room implementation from public methodology only |
| 7 | Wendy1890/crypto_pump_predictor_bot | volume ratio; current candle return; RSI; volatility; short trend; liquidity/universe prefilters; not-already-moved-too-far concept | Hard-coded alert thresholds as proof of alpha |
| 8 | Derposoft/crypto_pump_and_dump_with_deep_learning | Transformer sequence encoder; anomaly-attention research direction; short-segment modelling | Reconstruction/detection objective as the final target; V2 predicts future horizons directly |
| 9 | uasth/CryptoAnomalyDetection | High-frequency anomaly-Transformer / Conv-LSTM design evidence | Duplicate detector branch in V2 core; kept as an ablation source |
| 10 | lidialosavio-dotcom/crypto_fraud_stGNN | buy pressure; dynamic volume/trade correlations; temporal graph idea; temporal split + embargo discipline | Graph construction that can include future correlations |
| 11 | Confringo233/PumpWatcher | Causal graph sampling rule (`edge.timestamp <= as_of`); temporal neighbor aggregation; wallet/transaction graph branch | Any future edge; V2 unit-tests this invariant |
| 12 | mfzhang/20260115-Pump_and_Dump_detect_signal | Market + social/news fusion; temporal learned embedding concept | Bidirectional/reconstruction architecture as a direct live signal; V2 uses causal masking |
| 13 | Mehrnoom/Cryptocurrency-Pump-Dump | Per-coin social mention count, unique users, sentiment; Telegram pump history | Raw user identity; V2 works with aggregates and rates |
| 14 | lodetomasi/the-intelligent-llm-investor | Engagement velocity, cross-platform breadth, coordination/new-account indicators | LLM judgement as the core numeric predictor; it can be an auxiliary feature source |
| 15 | RaibekTussupbekov/binance-pump-dump-detector | Altcoin-vs-BTC residual and Bayesian online changepoint probability | Expensive unbounded run-length implementation; V2 caps run length and runs BOCD on past-standardized residuals |

## Native V2 feature families

### Market and microstructure
- returns and realized volatility at multiple causal windows
- quote-volume z-score and acceleration
- current candle displacement and RSI
- buy pressure, buy/sell flow imbalance and depth imbalance
- long-trade share when trade counts are available
- slippage imbalance when signed slippage is available
- fast/slow exponentially-weighted activity ratios
- cross-sectional ranks at each timestamp

### Early launch / on-chain
- transaction velocity (`tx_per_sec`)
- liquidity velocity and virtual-liquidity growth
- creator history / number of prior deployments
- wallet concentration (Gini/top buyer/top-3/top-5)
- repeat/dust/unknown/low-history/sniper buyer rates
- smart-wallet score and bot-pressure composites

### Social
- mention and unique-user log levels
- velocity, acceleration and rolling z-scores
- sentiment level/delta/z-score
- cross-platform breadth / coordination / new-account composites

### Regime / BOCD
If `BTCUSDT` is in the panel, V2 forms a same-timestamp benchmark residual. Otherwise it uses the asset return itself. The residual is standardized using a past-only rolling window and passed through capped Bayesian online changepoint detection. `bocd_cp_prob` is supplied independently to the meta-model.

## Targets

Defaults assume 5-minute bars:

| Target | Horizon | Threshold |
|---|---:|---:|
| `target__up10_15m` | 3 bars / 15 min | +10% |
| `target__up20_1h` | 12 bars / 1 h | +20% |
| `target__up50_6h` | 72 bars / 6 h | +50% |
| `target__up100_24h` | 288 bars / 24 h | +100% |

Each target also stores `mfe__*`, `mae__*`, `end_return__*` and `hit_bar__*`. Rows without the complete future horizon are censored. V2 never converts an incomplete tail window into a negative example.

## Validation rules

1. All features at `t` use observations `<= t`.
2. Sequence examples end at `t`; the Transformer uses a causal attention mask.
3. Wallet/transaction graphs delete every edge with `timestamp > as_of`.
4. Train, validation and test are chronological.
5. Train→validation and validation→test are purged by at least the longest label horizon (288 bars by default).
6. An additional embargo follows validation before test.
7. The stacker and isotonic calibration fit on validation only; final test remains untouched.
8. Every walk-forward fold retrains from earlier data and produces immutable OOS predictions.
9. Report classification and ranking metrics separately. A high AUC is not treated as evidence of executable PnL.

## Known limitations

The V2 repository provides the training architecture, not a claim of profitable performance. Real results depend on point-in-time universe construction, delisted/dead tokens, transaction availability, timestamp alignment, fees, spread, slippage, market impact, latency and adversarial regime changes. Those must be included before a trading conclusion is made.
