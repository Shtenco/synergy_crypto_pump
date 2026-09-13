from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from crypto_pump.v2.features import V2FeatureBuilder
from crypto_pump.v2.pipeline import V2Ensemble
from crypto_pump.v3.live import ShadowSignalEngine, ShadowStrategyConfig
from crypto_pump.v3.monitor import health_report
from crypto_pump.v3.panel import normalize_pumpportal_trades, trades_to_panel
from crypto_pump.v3.risk import RiskManager
from crypto_pump.v3.solana import JupiterQuoteClient
from crypto_pump.v3.store import V3Store


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Continuously score Pump.fun/PumpSwap data and record V3 shadow positions")
    p.add_argument("--store", default="data/v3/live.sqlite")
    p.add_argument("--model", required=True)
    p.add_argument("--equity-usd", type=float, default=10_000.0)
    p.add_argument("--probability-column", default="pump_probability__20pct_1h")
    p.add_argument("--horizon", default="20pct_1h")
    p.add_argument("--min-probability", type=float, default=0.70)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--notional-usd", type=float, default=50.0)
    p.add_argument("--lookback-hours", type=float, default=48.0)
    p.add_argument("--poll-seconds", type=float, default=60.0)
    p.add_argument("--once", action="store_true")
    p.add_argument("--reference-features", type=Path)
    return p.parse_args()


def expected_model_features(model: V2Ensemble) -> list[str]:
    names: set[str] = set()
    for branch_name in ("tabular", "ranker", "social"):
        branch = getattr(model, branch_name, None)
        if branch is None:
            continue
        for attr in ("features", "feature_columns"):
            value = getattr(branch, attr, None)
            if value:
                names.update(map(str, value))
    return sorted(names)


def cycle(args: argparse.Namespace, model: V2Ensemble, store: V3Store, quote: JupiterQuoteClient) -> dict:
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=args.lookback_hours)
    events = store.events(since=since)
    trades = normalize_pumpportal_trades(events)
    if trades.empty:
        report = {"status": "no_trades", "events": len(events)}
        store.record_health(report)
        return report

    sol_usd = quote.sol_usd_price()
    panel = trades_to_panel(trades, sol_usd=sol_usd)
    panel = panel.dropna(subset=["open", "high", "low", "close"])
    if panel.empty:
        report = {"status": "no_complete_bars", "trades": len(trades)}
        store.record_health(report)
        return report

    features = V2FeatureBuilder(benchmark_symbol="__NO_BENCHMARK__").transform(panel)
    expected = expected_model_features(model)
    reference = pd.read_parquet(args.reference_features) if args.reference_features else None
    health = health_report(features, expected_features=expected, reference_frame=reference)
    health_payload = {
        "status": "healthy" if health.healthy else "blocked",
        "data_age_seconds": health.data_age_seconds,
        "missing_fraction": health.missing_fraction,
        "max_psi": health.max_psi,
        "missing_features": list(health.missing_features),
        "reasons": list(health.reasons),
        "panel_rows": len(panel),
        "feature_rows": len(features),
        "sol_usd": sol_usd,
    }
    store.record_health(health_payload)
    if not health.healthy:
        return health_payload

    pred = model.predict(features)
    ranking = pd.concat([
        features[["timestamp", "symbol", "liquidity_usd"]].reset_index(drop=True),
        pred.reset_index(drop=True),
    ], axis=1)
    engine = ShadowSignalEngine(
        store=store,
        quote_client=quote,
        risk=RiskManager(),
        config=ShadowStrategyConfig(
            probability_column=args.probability_column,
            horizon=args.horizon,
            min_probability=args.min_probability,
            top_k=args.top_k,
            requested_notional_usd=args.notional_usd,
            model_version=Path(args.model).name,
        ),
    )
    opened = engine.submit_ranking(ranking, equity_usd=args.equity_usd, sol_usd=sol_usd)
    marked = engine.mark_open_positions(sol_usd=sol_usd)
    return {
        **health_payload,
        "status": "ok",
        "opened": len(opened),
        "marked": len(marked),
        "open_positions": len(store.open_positions()),
    }


def main() -> None:
    args = parse_args()
    store = V3Store(args.store)
    quote = JupiterQuoteClient()
    model = V2Ensemble.load(args.model)
    while True:
        try:
            print(json.dumps(cycle(args, model, store, quote), indent=2, default=str))
        except Exception as exc:
            payload = {"status": "error", "error": repr(exc)}
            store.record_health(payload)
            print(json.dumps(payload, indent=2))
        if args.once:
            break
        time.sleep(max(5.0, args.poll_seconds))


if __name__ == "__main__":
    main()
