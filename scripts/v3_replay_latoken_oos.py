from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from crypto_pump.v2.config import DEFAULT_HORIZONS
from crypto_pump.v3.execution import ExecutionBacktester, ExecutionCostModel
from crypto_pump.v3.metrics import financial_report, stress_test_trades


CAPACITY_NOTIONALS = (20.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)


def _quote_turnover_usd(panel: pd.DataFrame) -> pd.Series:
    for col in ("quote_volume_usd", "quote_volume", "turnover_usd", "turnover"):
        if col in panel.columns:
            return pd.to_numeric(panel[col], errors="coerce")
    return pd.to_numeric(panel["close"], errors="coerce") * pd.to_numeric(panel["volume"], errors="coerce")


def _prepare_market(panel: pd.DataFrame, *, liquidity_fraction: float) -> pd.DataFrame:
    d = panel.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d = d.dropna(subset=["timestamp", "symbol", "close"]).sort_values(["symbol", "timestamp"], kind="mergesort")
    is_usdt = d["symbol"].astype(str).str.upper().str.contains("USDT")
    d = d[is_usdt].copy()
    turnover = _quote_turnover_usd(d).clip(lower=0)
    d["quote_turnover_usd"] = turnover
    trailing = (
        d.assign(_turnover=turnover)
        .groupby("symbol", sort=False)["_turnover"]
        .rolling(12, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    d["liquidity_usd"] = trailing * float(liquidity_fraction)
    return d


def _score_columns(pred: pd.DataFrame) -> list[tuple[object, str]]:
    found = []
    for spec in DEFAULT_HORIZONS:
        candidates = (f"pump_probability__{spec.name}", f"p__{spec.name}")
        col = next((c for c in candidates if c in pred.columns), None)
        if col is not None:
            found.append((spec, col))
    if not found:
        available = sorted(c for c in pred.columns if c.startswith(("pump_probability__", "p__")))
        raise ValueError(f"No supported frozen V2 probability columns found; available={available}")
    return found


def _signal_market(
    market: pd.DataFrame,
    predictions: pd.DataFrame,
    spec,
    score_col: str,
    *,
    threshold: float,
    top_k: int,
    notional_usd: float,
) -> pd.DataFrame:
    p = predictions[["timestamp", "symbol", score_col]].copy()
    p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True, errors="coerce")
    p["probability"] = pd.to_numeric(p[score_col], errors="coerce")
    p = p.dropna(subset=["timestamp", "symbol", "probability"])
    p["rank"] = p.groupby("timestamp", sort=False)["probability"].rank(method="first", ascending=False)
    p["signal"] = (p["probability"] >= float(threshold)) & (p["rank"] <= int(top_k))
    lookup = p.set_index(["timestamp", "symbol"])["signal"]
    out = market.copy()
    keys = pd.MultiIndex.from_frame(out[["timestamp", "symbol"]])
    out["signal"] = lookup.reindex(keys, fill_value=False).to_numpy(bool)
    out["notional_usd"] = float(notional_usd)
    # The frozen legacy artifacts contain future-MFE labels but do not preserve
    # an ordered intrahorizon price path. A fixed-horizon close is intentionally
    # used so replay never invents a TP/SL hit time from future summaries.
    out["exit_bar"] = int(spec.horizon_bars)
    return out


def _run_all(
    market: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    costs: ExecutionCostModel,
    threshold: float,
    top_k: int,
    notional_usd: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    parts: list[pd.DataFrame] = []
    markets: dict[str, pd.DataFrame] = {}
    for spec, score_col in _score_columns(predictions):
        m = _signal_market(
            market,
            predictions,
            spec,
            score_col,
            threshold=threshold,
            top_k=top_k,
            notional_usd=notional_usd,
        )
        t = ExecutionBacktester(costs).backtest(m)
        if not t.empty:
            t["horizon"] = spec.name
            t["score_column"] = score_col
            parts.append(t)
        markets[spec.name] = m
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), markets


def _scenario_metrics(markets: dict[str, pd.DataFrame], costs: ExecutionCostModel) -> dict:
    scenarios = {
        "base": costs,
        "costs_x2": costs.scaled(cost_mult=2.0, latency_mult=1.0),
        "latency_x2": costs.scaled(cost_mult=1.0, latency_mult=2.0),
        "costs_x2_latency_x2": costs.scaled(cost_mult=2.0, latency_mult=2.0),
    }
    result = {}
    for name, c in scenarios.items():
        parts = []
        for horizon, m in markets.items():
            t = ExecutionBacktester(c).backtest(m)
            if not t.empty:
                t["horizon"] = horizon
                parts.append(t)
        all_trades = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        result[name] = financial_report(all_trades)
    return result


def replay(
    panel_path: Path,
    predictions_path: Path,
    output_dir: Path,
    *,
    threshold: float = 0.70,
    top_k: int = 10,
    notional_usd: float = 50.0,
    liquidity_fraction: float = 0.05,
) -> dict:
    panel = _prepare_market(_read(panel_path), liquidity_fraction=liquidity_fraction)
    pred = _read(predictions_path)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True, errors="coerce")
    costs = ExecutionCostModel(
        dex_fee_bps=10.0,
        base_slippage_bps=20.0,
        gas_usd=0.0,
        mev_haircut_bps=0.0,
        failure_rate=0.005,
        failed_tx_cost_usd=0.0,
        latency_bars=1,
        impact_coefficient=0.25,
        min_liquidity_usd=1_000.0,
        max_impact_bps=100.0,
    )
    trades, markets = _run_all(panel, pred, costs=costs, threshold=threshold, top_k=top_k, notional_usd=notional_usd)
    execution_stress = _scenario_metrics(markets, costs)

    capacity = []
    for n in CAPACITY_NOTIONALS:
        cap_trades, _ = _run_all(panel, pred, costs=costs, threshold=threshold, top_k=top_k, notional_usd=n)
        capacity.append({"notional_usd": n, **financial_report(cap_trades)})

    output_dir.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(output_dir / "latoken_legacy_oos_trades.parquet", index=False)
    simplified_stress = {
        "costs_x2_net_pnl_usd": execution_stress.get("costs_x2", {}).get("net_pnl_usd"),
        "latency_x2_net_pnl_usd": execution_stress.get("latency_x2", {}).get("net_pnl_usd"),
        "costs_x2_latency_x2_net_pnl_usd": execution_stress.get("costs_x2_latency_x2", {}).get("net_pnl_usd"),
    }
    (output_dir / "latoken_legacy_execution_stress.json").write_text(json.dumps(simplified_stress, indent=2), encoding="utf-8")
    score_columns = [c for _, c in _score_columns(pred)]
    report = {
        "source": "frozen_legacy_v2_oos_predictions",
        "retrained": False,
        "exit_policy": "fixed_horizon_close_after_1bar_execution_latency",
        "threshold": threshold,
        "top_k": top_k,
        "notional_usd": notional_usd,
        "liquidity_proxy": {
            "definition": "liquidity_fraction * causal trailing_1h_USDT_turnover",
            "fraction": liquidity_fraction,
            "warning": "legacy LATOKEN run did not archive L2 order-book depth; this is a conservative capacity proxy, not observed depth",
        },
        "cost_model": {
            "fee_bps_per_leg": costs.dex_fee_bps,
            "base_slippage_bps_per_leg": costs.base_slippage_bps,
            "latency_bars": costs.latency_bars,
            "impact_coefficient": costs.impact_coefficient,
            "min_liquidity_usd": costs.min_liquidity_usd,
            "max_impact_bps": costs.max_impact_bps,
        },
        "financial": financial_report(trades),
        "winner_stress": stress_test_trades(trades),
        "execution_stress": execution_stress,
        "capacity": capacity,
        "score_columns": score_columns,
    }
    (output_dir / "latoken_legacy_execution_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Replay frozen LATOKEN V2 OOS probabilities through an auditable execution model without retraining")
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.70)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--notional-usd", type=float, default=50.0)
    p.add_argument("--liquidity-fraction", type=float, default=0.05)
    args = p.parse_args()
    print(json.dumps(replay(
        args.panel,
        args.predictions,
        args.output_dir,
        threshold=args.threshold,
        top_k=args.top_k,
        notional_usd=args.notional_usd,
        liquidity_fraction=args.liquidity_fraction,
    ), indent=2, default=str))


if __name__ == "__main__":
    main()
