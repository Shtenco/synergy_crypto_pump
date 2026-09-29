# CRYPTO PUMP V3 — operational runbook

V3 is a research/shadow system. It must not sign or broadcast transactions until the promotion gates pass.

## 1. Install

```bash
python -m pip install -e '.[v3,dev]'
```

## 2. Start point-in-time Pump.fun / PumpSwap collection

```bash
python scripts/v3_collect_pump.py \
  --store data/v3/live.sqlite \
  --max-active-mints 200 \
  --token-ttl-seconds 21600
```

The collector uses one websocket, reconnects with exponential backoff, dynamically subscribes to token trades, and stores raw event JSON idempotently in SQLite.

## 3. Optional Solana RPC backfill

Set a production RPC endpoint and backfill wallets/programs that matter to the research universe:

```bash
export SOLANA_RPC_URL='https://...'
python scripts/v3_backfill_solana.py \
  --store data/v3/live.sqlite \
  --address <PROGRAM_OR_WALLET> \
  --max-signatures 10000
```

This stage stores raw transactions for later wallet/program decoding. Do not treat raw backfill as a decoded trade panel until a decoder has produced causal fields.

## 4. Build a historical Solana panel

The normalized trade path is:

`raw events -> normalize_pumpportal_trades -> 5m panel -> wallet features -> V2 causal feature builder`

Persist a multi-month panel before drawing conclusions. A few days of data are only a pipeline smoke test.

## 5. Train V3 on first-passage targets

```bash
python scripts/v3_train_trading.py \
  data/solana_panel.parquet \
  artifacts/v3 \
  --n-splits 3 \
  --probability-threshold 0.70 \
  --top-k 10 \
  --notional-usd 50
```

V3 target semantics are `TP before SL`, with one-bar entry delay and conservative `both hit in one OHLC bar => stop first` handling.

Outputs include:

- `v3_trading_model.joblib`
- `v3_reference_features.parquet`
- purged walk-forward fold models
- OOS probabilities
- OOS execution-aware trades
- financial metrics
- costs x2 / latency x2 stress tests
- winner-removal stress tests
- capacity curve

## 6. Start shadow decisions

```bash
python scripts/v3_shadow_loop.py \
  --store data/v3/live.sqlite \
  --model artifacts/v3/v3_trading_model.joblib \
  --reference-features artifacts/v3/v3_reference_features.parquet \
  --equity-usd 10000 \
  --notional-usd 50 \
  --min-probability 0.70
```

The live loop:

1. reads recent point-in-time events;
2. builds 5-minute bars and causal wallet features;
3. enforces feature parity, freshness, missingness and PSI drift gates;
4. scores the V3 first-passage model;
5. requests read-only Jupiter quotes;
6. applies liquidity, price-impact, slippage and portfolio risk limits;
7. records a shadow position without signing anything;
8. re-quotes open positions and closes them virtually on TP, SL or timeout.

## 7. Promotion report

After accumulating shadow observations:

```bash
python scripts/v3_promotion_report.py \
  --store data/v3/live.sqlite \
  --oos-report artifacts/v3/oos/v3_oos_report.json \
  --horizon tp20_sl08_1h \
  --equity-usd 10000 \
  --ci-green
```

Exit code `0` means the shadow-to-canary gate passed. Exit code `2` means real money remains blocked.

Minimum default canary gate:

- CI green on exact commit;
- >= 300 closed shadow trades;
- >= 30 shadow days;
- positive net expectancy;
- profit factor >= 1.05;
- max drawdown <= 5% of reference equity;
- nonnegative OOS PnL under costs x2;
- nonnegative result after removing top 1% winners;
- zero recorded risk violations.

## 8. Tiny canary is a separate stage

A future live signer must be a separate module and should remain absent from the shadow branch. The first canary should use small fixed notional and hard transaction limits. Do not scale until at least the canary-to-scale gate passes:

- >= 30 live canary trades;
- >= 7 live days;
- positive live expectancy;
- shadow/live execution drift <= 75 bps;
- no risk violations;
- no kill-switch event.

## 9. Current hard blockers

- GitHub Actions must execute normally and become green; `runner_id=0 / steps=[]` pre-run failures are infrastructure failures, not test results.
- Historical Solana/Pump.fun data still has to be accumulated/backfilled in sufficient depth.
- Executable alpha is not proven until OOS + shadow statistics satisfy the gates.
- No transaction signer belongs in the system before those gates pass.
