#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from crypto_pump.v2.features import V2FeatureBuilder
from crypto_pump.v3.execution import ExecutionBacktester, ExecutionCostModel
from crypto_pump.v3.labels import TripleBarrierSpec, add_triple_barrier_labels
from crypto_pump.v3.metrics import financial_report, stress_test_trades
from crypto_pump.v3.model import TradingHorizonSpec, V3TradingEnsemble
from crypto_pump.v3.portfolio import PortfolioPolicy, apply_portfolio_limits


SPEC = TradingHorizonSpec("tp20_sl08_1h", 12, 0.20, 0.08)
BARRIER = TripleBarrierSpec(SPEC.name, SPEC.horizon_bars, SPEC.return_threshold, SPEC.stop_loss)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)


def _split_times(frame: pd.DataFrame, *, train_fraction: float = 0.65, validation_fraction: float = 0.15,
                 purge_bars: int = 13, embargo_bars: int = 12):
    times = np.array(sorted(pd.to_datetime(frame["timestamp"], utc=True).unique()))
    n = len(times)
    if n < 500:
        raise ValueError(f"need >=500 unique 5m timestamps, got {n}")
    oos_start = int(n * (train_fraction + validation_fraction))
    val_start_raw = int(n * train_fraction)
    val_end = oos_start - purge_bars - embargo_bars
    train_end = val_start_raw - purge_bars
    if train_end <= 0 or val_end <= val_start_raw or oos_start >= n:
        raise ValueError("invalid frozen split after purge/embargo")
    return times[:train_end], times[val_start_raw:val_end], times[oos_start:]


def _candidate_trades(
    full: pd.DataFrame,
    test: pd.DataFrame,
    pred: pd.DataFrame,
    *,
    threshold: float,
    top_k: int,
    notional_usd: float,
    costs: ExecutionCostModel,
) -> pd.DataFrame:
    pcol = f"pump_probability__{SPEC.name}"
    scored = test[["timestamp", "symbol"]].copy()
    scored["probability"] = pd.to_numeric(pred[pcol], errors="coerce").to_numpy(float)
    scored["rank"] = scored.groupby("timestamp", sort=False)["probability"].rank(method="first", ascending=False)
    scored["signal"] = (scored["probability"] >= threshold) & (scored["rank"] <= top_k)
    selected = scored[scored["signal"]].copy()
    if selected.empty:
        return pd.DataFrame()

    d = full.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.sort_values(["symbol", "timestamp"], kind="stable")
    by_symbol = {s: g.reset_index(drop=True) for s, g in d.groupby("symbol", sort=False)}
    score_lookup = selected.set_index(["timestamp", "symbol"])[["probability", "rank"]]
    rows = []

    for (signal_ts, symbol), sr in score_lookup.iterrows():
        g = by_symbol.get(symbol)
        if g is None:
            continue
        pos = g.index[g["timestamp"].eq(signal_ts)]
        if len(pos) != 1:
            continue
        i = int(pos[0])
        entry_i = i + costs.latency_bars
        stop_i = entry_i + SPEC.horizon_bars
        if entry_i >= len(g) or stop_i >= len(g):
            continue
        entry_mid = float(g.loc[entry_i, "close"])
        if not np.isfinite(entry_mid) or entry_mid <= 0:
            continue

        tp = entry_mid * (1.0 + SPEC.return_threshold)
        sl = entry_mid * (1.0 - SPEC.stop_loss)
        exit_i = stop_i
        exit_mid = float(g.loc[stop_i, "close"])
        reason = "timeout"

        for j in range(entry_i + 1, stop_i + 1):
            hi = float(g.loc[j, "high"])
            lo = float(g.loc[j, "low"])
            hit_tp = np.isfinite(hi) and hi >= tp
            hit_sl = np.isfinite(lo) and lo <= sl
            if hit_tp and hit_sl:
                exit_i, exit_mid, reason = j, sl, "both_stop_first"
                break
            if hit_sl:
                exit_i, exit_mid, reason = j, sl, "stop"
                break
            if hit_tp:
                exit_i, exit_mid, reason = j, tp, "take_profit"
                break

        liq = float(g.loc[entry_i, "liquidity_usd"]) if pd.notna(g.loc[entry_i, "liquidity_usd"]) else float("nan")
        est = ExecutionBacktester(costs).estimate_trade(entry_mid, exit_mid, notional_usd, liq)
        rows.append({
            "signal_timestamp": signal_ts,
            "entry_timestamp": g.loc[entry_i, "timestamp"],
            "exit_timestamp": g.loc[exit_i, "timestamp"],
            "symbol": symbol,
            "probability": float(sr["probability"]),
            "rank": float(sr["rank"]),
            "exit_reason": reason,
            "bars_held": int(exit_i - entry_i),
            "minutes_held": int((exit_i - entry_i) * 5),
            "notional_usd": float(notional_usd),
            "liquidity_usd": liq,
            "executable": est.executable,
            "reject_reason": None if est.executable else est.reason,
            "impact_bps": est.impact_bps,
            "entry_price": est.entry_price,
            "exit_price": est.exit_price,
            "net_return": est.net_return,
            "net_pnl_usd": est.net_pnl_usd,
        })

    return pd.DataFrame(rows)


def _metrics(trades: pd.DataFrame, initial_equity_usd: float) -> dict:
    rep = financial_report(trades)
    x = trades[trades.get("executable", True)].copy() if not trades.empty else trades
    rep["pnl_pct_initial_equity"] = (
        float(rep.get("net_pnl_usd", 0.0) or 0.0) / initial_equity_usd
        if initial_equity_usd > 0 else None
    )
    rep["average_holding_minutes"] = float(pd.to_numeric(x.get("minutes_held"), errors="coerce").mean()) if len(x) else None
    return rep


def run(source: Path, output_dir: Path, *, threshold: float = 0.70, top_k: int = 10,
        notional_usd: float = 50.0, initial_equity_usd: float = 5000.0) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = _read(source)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.sort_values(["symbol", "timestamp"], kind="stable").reset_index(drop=True)

    # Fail closed: every within-symbol step must be exactly five minutes.
    gaps = raw.groupby("symbol", sort=False)["timestamp"].diff().dropna()
    bad_gaps = int((gaps != pd.Timedelta(minutes=5)).sum())
    if bad_gaps:
        raise ValueError(f"source is not an exact 5-minute grid: bad_steps={bad_gaps}")
    liq = pd.to_numeric(raw.get("liquidity_usd"), errors="coerce")
    liq_coverage = float((liq.notna() & (liq > 0)).mean())
    if liq_coverage < 0.95:
        raise ValueError(f"liquidity_usd coverage {liq_coverage:.3%} < 95%")

    featured = V2FeatureBuilder(benchmark_symbol="__NO_BENCHMARK__").transform(raw)
    labeled = add_triple_barrier_labels(featured, specs=[BARRIER], entry_delay_bars=1)

    train_times, val_times, oos_times = _split_times(labeled)
    ts = labeled["timestamp"]
    train = labeled[ts.isin(train_times)].copy()
    validation = labeled[ts.isin(val_times)].copy()
    oos = labeled[ts.isin(oos_times)].copy()

    model = V3TradingEnsemble(specs=[SPEC], model_kind="catboost").fit(train, validation)
    frozen_path = output_dir / "frozen_pre_oos_model.joblib"
    model.save(frozen_path)

    # The model is frozen before this line. OOS is read exactly once for scoring.
    pred = model.predict(oos)
    pred_out = pd.concat([oos[["timestamp", "symbol"]].reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    pred_out.to_parquet(output_dir / "frozen_oos_predictions.parquet", index=False)

    # Historical Pump.fun bonding-curve fee: 1.25% per leg. Stress tests multiply
    # all fee/slippage/MEV components without changing the frozen signal set.
    base_costs = ExecutionCostModel(
        dex_fee_bps=125.0,
        base_slippage_bps=35.0,
        gas_usd=0.05,
        mev_haircut_bps=15.0,
        failure_rate=0.02,
        failed_tx_cost_usd=0.01,
        latency_bars=1,
        impact_coefficient=0.50,
        min_liquidity_usd=5_000.0,
        max_impact_bps=300.0,
    )
    policy = PortfolioPolicy(
        initial_equity_usd=initial_equity_usd,
        max_total_exposure_fraction=0.10,
        max_position_fraction_equity=0.02,
        max_concurrent_positions=10,
        max_daily_loss_fraction=0.02,
        one_open_position_per_symbol=True,
    )

    scenarios = {
        "base": base_costs,
        "costs_x2": base_costs.scaled(cost_mult=2.0, latency_mult=1.0),
        "latency_x2": base_costs.scaled(cost_mult=1.0, latency_mult=2.0),
        "costs_x2_latency_x2": base_costs.scaled(cost_mult=2.0, latency_mult=2.0),
    }
    scenario_reports = {}
    base_portfolio = pd.DataFrame()
    for name, costs in scenarios.items():
        candidates = _candidate_trades(
            labeled, oos, pred,
            threshold=threshold, top_k=top_k, notional_usd=notional_usd, costs=costs,
        )
        portfolio = apply_portfolio_limits(candidates, policy)
        if name == "base":
            base_portfolio = portfolio
            portfolio.to_parquet(output_dir / "frozen_oos_trades.parquet", index=False)
        scenario_reports[name] = _metrics(portfolio, initial_equity_usd)

    winner = stress_test_trades(base_portfolio)
    capacity = []
    for size in (20, 50, 100, 250, 500, 1000):
        candidates = _candidate_trades(
            labeled, oos, pred,
            threshold=threshold, top_k=top_k, notional_usd=float(size), costs=base_costs,
        )
        p = apply_portfolio_limits(candidates, policy)
        capacity.append({"notional_usd": float(size), **_metrics(p, initial_equity_usd)})

    use = base_portfolio[base_portfolio.get("executable", True)].copy() if not base_portfolio.empty else base_portfolio
    exit_counts = use["exit_reason"].value_counts().to_dict() if len(use) else {}
    report = {
        "protocol": "FROZEN_SINGLE_OOS_V1",
        "target": SPEC.name,
        "take_profit": SPEC.return_threshold,
        "stop_loss": SPEC.stop_loss,
        "horizon_bars": SPEC.horizon_bars,
        "horizon_minutes": SPEC.horizon_bars * 5,
        "threshold": threshold,
        "top_k": top_k,
        "notional_usd": notional_usd,
        "initial_equity_usd": initial_equity_usd,
        "split": {
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "oos_rows": int(len(oos)),
            "train_timestamp_max": str(pd.to_datetime(train["timestamp"], utc=True).max()),
            "validation_timestamp_min": str(pd.to_datetime(validation["timestamp"], utc=True).min()),
            "validation_timestamp_max": str(pd.to_datetime(validation["timestamp"], utc=True).max()),
            "oos_timestamp_min": str(pd.to_datetime(oos["timestamp"], utc=True).min()),
            "oos_timestamp_max": str(pd.to_datetime(oos["timestamp"], utc=True).max()),
            "purge_bars": 13,
            "embargo_bars": 12,
        },
        "data_quality": {
            "rows": int(len(raw)),
            "symbols": int(raw["symbol"].nunique()),
            "bad_5m_steps": bad_gaps,
            "liquidity_coverage": liq_coverage,
        },
        "feature_count": len(model.features),
        "features": model.features,
        "frozen_model_path": str(frozen_path),
        "financial": scenario_reports["base"],
        "execution_stress": scenario_reports,
        "winner_stress": winner,
        "capacity": capacity,
        "exit_reason_counts": {str(k): int(v) for k, v in exit_counts.items()},
        "verdict_inputs": {
            "positive_expectancy": float(scenario_reports["base"].get("expectancy_usd", 0) or 0) > 0,
            "profit_factor_ge_1_05": (
                scenario_reports["base"].get("profit_factor") is not None
                and float(scenario_reports["base"]["profit_factor"]) >= 1.05
            ),
            "costs_x2_nonnegative": float(scenario_reports["costs_x2"].get("net_pnl_usd", 0) or 0) >= 0,
            "top1pct_removed_nonnegative": float((winner.get("remove_top_1pct") or {}).get("net_pnl_usd", -1) or 0) >= 0,
            "max_dd_le_5pct": abs(min(0.0, float(scenario_reports["base"].get("max_drawdown_usd", 0) or 0))) <= initial_equity_usd * 0.05,
        },
    }
    report["historical_gate_passed"] = all(report["verdict_inputs"].values())
    (output_dir / "frozen_oos_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="One-shot pre-OOS retrain and fully frozen profitability test")
    ap.add_argument("source", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--notional-usd", type=float, default=50.0)
    ap.add_argument("--initial-equity-usd", type=float, default=5000.0)
    args = ap.parse_args()
    print(json.dumps(run(
        args.source, args.output_dir,
        threshold=args.threshold, top_k=args.top_k,
        notional_usd=args.notional_usd, initial_equity_usd=args.initial_equity_usd,
    ), indent=2, default=str))


if __name__ == "__main__":
    main()
