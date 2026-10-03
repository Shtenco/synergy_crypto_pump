#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_pump.v2.features import V2FeatureBuilder
from crypto_pump.v3.execution import ExecutionBacktester, ExecutionCostModel
from crypto_pump.v3.labels import TripleBarrierSpec, add_triple_barrier_labels
from crypto_pump.v3.metrics import financial_report, stress_test_trades
from crypto_pump.v3.model import TradingHorizonSpec, V3TradingEnsemble, classification_report
from crypto_pump.v3.portfolio import PortfolioPolicy, apply_portfolio_limits


SPEC = TradingHorizonSpec("tp20_sl08_1h", 12, 0.20, 0.08)
BARRIER = TripleBarrierSpec(SPEC.name, SPEC.horizon_bars, SPEC.return_threshold, SPEC.stop_loss)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)


def _exact_grid(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only sufficiently long symbols and causally fill to an exact 5m grid."""
    d = frame.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    parts: list[pd.DataFrame] = []
    zero_cols = [c for c in ("volume", "quote_volume", "trades", "buy_volume", "sell_volume") if c in d.columns]
    for symbol, g in d.groupby("symbol", sort=False):
        g = g.sort_values("timestamp").drop_duplicates("timestamp", keep="last").set_index("timestamp")
        if g.empty:
            continue
        idx = pd.date_range(g.index.min(), g.index.max(), freq="5min", tz="UTC")
        g = g.reindex(idx)
        g.index.name = "timestamp"
        close = pd.to_numeric(g["close"], errors="coerce").ffill()
        for c in ("open", "high", "low", "close"):
            g[c] = pd.to_numeric(g[c], errors="coerce").fillna(close)
        for c in zero_cols:
            g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0.0)
        for c in g.columns:
            if c not in {"open", "high", "low", "close", *zero_cols}:
                g[c] = g[c].ffill()
        g["symbol"] = symbol
        g = g.reset_index()
        g = g.dropna(subset=["open", "high", "low", "close"])
        if len(g) >= 300:
            parts.append(g)
    if not parts:
        raise ValueError("no LATOKEN series survive exact-grid regularization")
    out = pd.concat(parts, ignore_index=True, sort=False)
    out = out.sort_values(["symbol", "timestamp"], kind="stable").reset_index(drop=True)
    gaps = out.groupby("symbol", sort=False)["timestamp"].diff().dropna()
    bad = int((gaps != pd.Timedelta(minutes=5)).sum())
    if bad:
        raise ValueError(f"exact-grid fail-closed: {bad} non-5m steps")
    return out


def _add_causal_liquidity_proxy(frame: pd.DataFrame) -> pd.DataFrame:
    """CEX stage-1 capacity proxy from PAST one-hour quote turnover only.

    This is intentionally not claimed to be order-book depth. It exists only to
    reject extremely thin bars and make impact/capacity stress more conservative.
    The current bar is shifted out so the execution proxy uses information known
    before the signal bar.
    """
    d = frame.copy()
    qv = pd.to_numeric(d.get("quote_volume"), errors="coerce").fillna(0.0)
    d["_qv"] = qv
    d["liquidity_usd"] = (
        d.groupby("symbol", sort=False)["_qv"]
        .transform(lambda s: s.shift(1).rolling(12, min_periods=3).sum())
        .fillna(0.0)
    )
    return d.drop(columns=["_qv"])


def _split_times(frame: pd.DataFrame, *, train_fraction: float = 0.65, validation_fraction: float = 0.15,
                 purge_bars: int = 13, embargo_bars: int = 12):
    times = np.array(sorted(pd.to_datetime(frame["timestamp"], utc=True).unique()))
    n = len(times)
    if n < 500:
        raise ValueError(f"need >=500 unique 5m timestamps, got {n}")
    val_start = int(n * train_fraction)
    oos_start = int(n * (train_fraction + validation_fraction))
    train_end = val_start - purge_bars
    val_end = oos_start - purge_bars - embargo_bars
    if train_end <= 0 or val_end <= val_start or oos_start >= n:
        raise ValueError("invalid frozen split after purge/embargo")
    return times[:train_end], times[val_start:val_end], times[oos_start:]


def _candidate_trades(full: pd.DataFrame, test: pd.DataFrame, pred: pd.DataFrame, *, threshold: float,
                      top_k: int, notional_usd: float, costs: ExecutionCostModel) -> pd.DataFrame:
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
    rows: list[dict] = []
    for _, sr in selected.iterrows():
        signal_ts, symbol = sr["timestamp"], sr["symbol"]
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
        exit_i, exit_mid, reason = stop_i, float(g.loc[stop_i, "close"]), "timeout"
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


def _metrics(trades: pd.DataFrame, equity: float) -> dict:
    r = financial_report(trades)
    x = trades[trades.get("executable", True)].copy() if not trades.empty else trades
    r["pnl_pct_initial_equity"] = float(r.get("net_pnl_usd", 0.0) or 0.0) / equity if equity > 0 else None
    r["average_holding_minutes"] = float(pd.to_numeric(x.get("minutes_held"), errors="coerce").mean()) if len(x) else None
    return r


def run(source: Path, output_dir: Path, *, threshold: float = 0.70, top_k: int = 10,
        notional_usd: float = 50.0, initial_equity_usd: float = 5000.0) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = _add_causal_liquidity_proxy(_exact_grid(_read(source)))
    raw.to_parquet(output_dir / "latoken_exact_5m_with_capacity_proxy.parquet", index=False)

    featured = V2FeatureBuilder(benchmark_symbol="LATOKEN:BTCUSDT").transform(raw)
    labeled = add_triple_barrier_labels(featured, specs=[BARRIER], entry_delay_bars=1)
    train_times, val_times, oos_times = _split_times(labeled)
    ts = labeled["timestamp"]
    train = labeled[ts.isin(train_times)].copy()
    validation = labeled[ts.isin(val_times)].copy()
    oos = labeled[ts.isin(oos_times)].copy()

    target = BARRIER.target_col
    split_target_stats = {
        "train_positive_rate": float(pd.to_numeric(train[target], errors="coerce").mean()),
        "validation_positive_rate": float(pd.to_numeric(validation[target], errors="coerce").mean()),
        "oos_positive_rate": float(pd.to_numeric(oos[target], errors="coerce").mean()),
        "train_positives": int(pd.to_numeric(train[target], errors="coerce").fillna(0).sum()),
        "validation_positives": int(pd.to_numeric(validation[target], errors="coerce").fillna(0).sum()),
        "oos_positives": int(pd.to_numeric(oos[target], errors="coerce").fillna(0).sum()),
    }
    if split_target_stats["train_positives"] < 5 or split_target_stats["validation_positives"] < 2:
        raise RuntimeError(f"insufficient positive support for frozen model: {split_target_stats}")

    model = V3TradingEnsemble(specs=[SPEC], model_kind="catboost").fit(train, validation)
    frozen_path = output_dir / "frozen_pre_oos_model.joblib"
    model.save(frozen_path)
    pred = model.predict(oos)
    pred_out = pd.concat([oos[["timestamp", "symbol", target]].reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    pred_out.to_parquet(output_dir / "frozen_oos_predictions.parquet", index=False)
    predictive = classification_report(oos.reset_index(drop=True), pred.reset_index(drop=True), specs=[SPEC], top_k=top_k)

    # Stage-1 LATOKEN/CEX assumptions are inherited from the original V1/V2 config:
    # 10 bps fee + 25 bps slippage per leg. No gas/MEV. Impact is a conservative
    # turnover-based proxy, not claimed order-book depth.
    base_costs = ExecutionCostModel(
        dex_fee_bps=10.0,
        base_slippage_bps=25.0,
        gas_usd=0.0,
        mev_haircut_bps=0.0,
        failure_rate=0.0,
        failed_tx_cost_usd=0.0,
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
    scenario_reports: dict[str, dict] = {}
    base_portfolio = pd.DataFrame()
    candidate_counts: dict[str, int] = {}
    for name, costs in scenarios.items():
        candidates = _candidate_trades(labeled, oos, pred, threshold=threshold, top_k=top_k,
                                       notional_usd=notional_usd, costs=costs)
        candidate_counts[name] = int(len(candidates))
        portfolio = apply_portfolio_limits(candidates, policy)
        if name == "base":
            base_portfolio = portfolio
            portfolio.to_parquet(output_dir / "frozen_oos_trades.parquet", index=False)
        scenario_reports[name] = _metrics(portfolio, initial_equity_usd)

    winner = stress_test_trades(base_portfolio)
    capacity = []
    for size in (20, 50, 100, 250, 500, 1000):
        candidates = _candidate_trades(labeled, oos, pred, threshold=threshold, top_k=top_k,
                                       notional_usd=float(size), costs=base_costs)
        p = apply_portfolio_limits(candidates, policy)
        capacity.append({"notional_usd": float(size), **_metrics(p, initial_equity_usd)})

    use = base_portfolio[base_portfolio.get("executable", True)].copy() if not base_portfolio.empty else base_portfolio
    exit_counts = use["exit_reason"].value_counts().to_dict() if len(use) else {}
    rejects = base_portfolio.loc[~base_portfolio.get("executable", True), "reject_reason"].value_counts().to_dict() if len(base_portfolio) else {}
    base = scenario_reports["base"]
    report = {
        "protocol": "LATOKEN_STAGE1_FROZEN_SINGLE_OOS_V1",
        "source": "LATOKEN public v2 historical OHLCV",
        "target": SPEC.name,
        "take_profit": SPEC.return_threshold,
        "stop_loss": SPEC.stop_loss,
        "horizon_minutes": 60,
        "threshold": threshold,
        "top_k": top_k,
        "notional_usd": notional_usd,
        "initial_equity_usd": initial_equity_usd,
        "split": {
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "oos_rows": int(len(oos)),
            "train_timestamp_min": str(train["timestamp"].min()),
            "train_timestamp_max": str(train["timestamp"].max()),
            "validation_timestamp_min": str(validation["timestamp"].min()),
            "validation_timestamp_max": str(validation["timestamp"].max()),
            "oos_timestamp_min": str(oos["timestamp"].min()),
            "oos_timestamp_max": str(oos["timestamp"].max()),
            "purge_bars": 13,
            "embargo_bars": 12,
        },
        "target_support": split_target_stats,
        "data_quality": {
            "rows": int(len(raw)),
            "symbols": int(raw["symbol"].nunique()),
            "timestamps": int(raw["timestamp"].nunique()),
            "timestamp_min": str(raw["timestamp"].min()),
            "timestamp_max": str(raw["timestamp"].max()),
            "capacity_proxy_coverage_ge_5k": float((raw["liquidity_usd"] >= 5000).mean()),
        },
        "execution_assumptions": {
            "fee_bps_per_leg": 10.0,
            "slippage_bps_per_leg": 25.0,
            "latency_bars": 1,
            "liquidity_proxy": "causal trailing 12x5m quote_volume sum, shifted one bar",
            "impact_coefficient": 0.50,
            "min_capacity_proxy_usd": 5000.0,
            "warning": "quote turnover is not order-book depth; this is Stage-1 screening, not final execution proof",
        },
        "feature_count": len(model.features),
        "features": model.features,
        "predictive_oos": predictive,
        "candidate_counts": candidate_counts,
        "financial": base,
        "execution_stress": scenario_reports,
        "winner_stress": winner,
        "capacity": capacity,
        "exit_reason_counts": {str(k): int(v) for k, v in exit_counts.items()},
        "reject_reason_counts": {str(k): int(v) for k, v in rejects.items()},
        "verdict_inputs": {
            "min_20_executable_trades": int(base.get("trades", 0) or 0) >= 20,
            "positive_expectancy": float(base.get("expectancy_usd", 0) or 0) > 0,
            "profit_factor_ge_1_05": base.get("profit_factor") is not None and float(base["profit_factor"]) >= 1.05,
            "costs_x2_nonnegative": float(scenario_reports["costs_x2"].get("net_pnl_usd", 0) or 0) >= 0,
            "top1pct_removed_nonnegative": float((winner.get("remove_top_1pct") or {}).get("net_pnl_usd", -1) or 0) >= 0,
            "max_dd_le_5pct": abs(min(0.0, float(base.get("max_drawdown_usd", 0) or 0))) <= initial_equity_usd * 0.05,
        },
    }
    report["stage1_gate_passed"] = all(report["verdict_inputs"].values())
    (output_dir / "latoken_frozen_oos_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen 10d x 100 LATOKEN Stage-1 profitability experiment")
    ap.add_argument("source", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--notional-usd", type=float, default=50.0)
    ap.add_argument("--initial-equity-usd", type=float, default=5000.0)
    args = ap.parse_args()
    print(json.dumps(run(args.source, args.output_dir, threshold=args.threshold, top_k=args.top_k,
                         notional_usd=args.notional_usd, initial_equity_usd=args.initial_equity_usd),
                     indent=2, default=str))


if __name__ == "__main__":
    main()
