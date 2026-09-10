# Upstream research map

`crypto_pump` deliberately separates **ideas/adapters** from third-party source trees. Run `python scripts/bootstrap_upstreams.py` to clone the research corpus into the ignored `external/upstreams/` directory and generate `upstreams.lock.json` with the exact resolved SHAs.

| # | Upstream | Primary value extracted for the unified design |
|---|---|---|
| 1 | B0R0koko/pump-and-dump-prediction | cross-sectional target ranking, pre-event market microstructure, liquidity-aware evaluation |
| 2 | Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency | sequence-conditioned target-coin prediction |
| 3 | cakcora/Chainoba | classical supervised pump features and anomaly research |
| 4 | baseershah7/Solana_Memecoin_Graduation | early transaction velocity, wallet concentration, creator history |
| 5 | gayatri1505/Solana-Memcoin-Prediction-Using-Machine-Learning | early-block feature ensemble |
| 6 | Based-LTD/graduate-oracle | wallet reputation / graduation-calibration methodology |
| 7 | Wendy1890/crypto_pump_predictor_bot | multi-exchange scanning and alert pipeline |
| 8 | Derposoft/crypto_pump_and_dump_with_deep_learning | temporal Conv-LSTM/Transformer anomaly representations |
| 9 | uasth/CryptoAnomalyDetection | sub-minute anomaly windows |
| 10 | lidialosavio-dotcom/crypto_fraud_stGNN | dynamic cross-asset graph structure |
| 11 | Confringo233/PumpWatcher | DeFi transaction graph + contrastive representations |
| 12 | mfzhang/20260115-Pump_and_Dump_detect_signal | social/news + market multimodal features |
| 13 | Mehrnoom/Cryptocurrency-Pump-Dump | historical social/Telegram dataset design |
| 14 | lodetomasi/the-intelligent-llm-investor | social momentum and coordinated-activity signals |
| 15 | RaibekTussupbekov/binance-pump-dump-detector | Bayesian change-point score as an independent regime feature |

## Integration policy

The native package does not copy unknown-license source code. Upstream repositories are cloned separately so their own license and Git history remain intact. For permissively licensed sources, specific adapters can later be implemented or vendored with attribution after license verification.

## Distilled signal groups

The common design distilled from these projects is:

```text
market microstructure
+ temporal anomaly state
+ low-volatility/compression state
+ DEX/on-chain wallet velocity
+ liquidity/holder concentration
+ social acceleration
+ graph coordination
+ prior pump/creator history
=> calibrated multi-horizon probability
=> timestamp-local cross-sectional rank
```

The native V1 intentionally starts with tabular leakage-safe features and a boosting model. Temporal Transformers, GNN embeddings and social encoders should enter as **out-of-fold meta-features**, not be trained on the same future labels and then reused in-sample.
