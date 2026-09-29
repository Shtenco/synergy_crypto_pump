from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from crypto_pump.v2.splits import purged_walk_forward_folds

from .execution import ExecutionBacktester, ExecutionCostModel
from .metrics import capacity_curve, financial_report, rerun_cost_stress, stress_test_trades
from .model import DEFAULT_TRADING_SPECS, TradingHorizonSpec, V3TradingEnsemble, classification_report


def _market_with_signals(
    full_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    spec: TradingHorizonSpec,
    probability_threshold: float,
    top_k: int,
    notional_usd: float,
) -> pd.DataFrame:
    pcol = f"pump_probability__{spec.name}"
    scored = test_frame[["timestamp", "symbol"]].copy()
    scored["probability"] = predictions[pcol].to_numpy(float)
    scored["rank"] = scored.groupby("timestamp")["probability"].rank(method="first", ascending=False)
    scored["signal"] = (scored["probability"] >= probability_threshold) & (scored["rank"] <= top_k)
    lookup = scored.set_index(["timestamp", "symbol"])["signal"]

    market = full_frame.copy()
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
    keys = pd.MultiIndex.from_frame(market[["timestamp", "symbol"]])
    market["signal"] = lookup.reindex(keys, fill_value=False).to_numpy(bool)
    market["notional_usd"] = float(notional_usd)
    market["exit_bar"] = pd.to_numeric(market[spec.hit_col], errors="coerce")
    if "liquidity_usd" not in market:
        if "liquidity" in market:
            market["liquidity_usd"] = pd.to_numeric(market["liquidity"], errors="coerce")
        else:
            market["liquidity_usd"] = float("nan")
    return market


def walk_forward_v3(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    specs: Sequence[TradingHorizonSpec] = DEFAULT_TRADING_SPECS,
    n_splits: int = 3,
    min_train_fraction: float = 0.45,
    validation_fraction: float = 0.10,
    embargo_bars: int = 12,
    model_kind: str = "auto",
    probability_threshold: float = 0.70,
    top_k: int = 10,
    notional_usd: float = 50.0,
    costs: ExecutionCostModel | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Purged OOS validation with execution-aware PnL on first-passage targets."""
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    d = frame.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    purge = max(s.horizon_bars for s in specs) + 1
    folds = purged_walk_forward_folds(
        d,
        n_splits=n_splits,
        min_train_fraction=min_train_fraction,
        validation_fraction=validation_fraction,
        purge_bars=purge,
        embargo_bars=embargo_bars,
    )
    cost_model = costs or ExecutionCostModel()
    prediction_parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    fold_reports: list[dict] = []

    for fold in folds:
        train, validation, test = fold.frames(d)
        model = V3TradingEnsemble(specs=specs, model_kind=model_kind).fit(train, validation)
        pred = model.predict(test)
        pred_out = pd.concat([test[["timestamp", "symbol"]].reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
        pred_out["fold"] = fold.fold
        prediction_parts.append(pred_out)
        model.save(outdir / f"fold_{fold.fold}_v3_trading.joblib")

        class_metrics = classification_report(test, pred, specs=specs, top_k=top_k)
        spec_reports: dict[str, dict] = {}
        for spec in specs:
            market = _market_with_signals(
                d,
                test,
                pred,
                spec=spec,
                probability_threshold=probability_threshold,
                top_k=top_k,
                notional_usd=notional_usd,
            )
            trades = ExecutionBacktester(cost_model).backtest(market)
            if not trades.empty:
                trades = trades[trades["signal_timestamp"].isin(pd.to_datetime(test["timestamp"], utc=True).unique())].copy()
                trades["fold"] = fold.fold
                trades["horizon"] = spec.name
                trade_parts.append(trades)
            spec_reports[spec.name] = {
                "classification": class_metrics.get(spec.name, {}),
                "financial": financial_report(trades),
                "winner_stress": stress_test_trades(trades),
                "execution_stress": rerun_cost_stress(market, base_costs=cost_model),
            }
        fold_reports.append({
            "fold": fold.fold,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "specs": spec_reports,
        })

    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    predictions.to_parquet(outdir / "v3_oos_predictions.parquet", index=False)
    trades.to_parquet(outdir / "v3_oos_trades.parquet", index=False)

    aggregate: dict[str, dict] = {}
    for spec in specs:
        t = trades[trades["horizon"] == spec.name].copy() if not trades.empty else pd.DataFrame()
        aggregate[spec.name] = {
            "financial": financial_report(t),
            "winner_stress": stress_test_trades(t),
        }
        # Capacity is recomputed from the full signal map over OOS predictions.
        if not predictions.empty:
            test_keys = predictions[["timestamp", "symbol"]].drop_duplicates()
            p = predictions[["timestamp", "symbol", f"pump_probability__{spec.name}"]].copy()
            p = p.rename(columns={f"pump_probability__{spec.name}": "probability"})
            p["rank"] = p.groupby("timestamp")["probability"].rank(method="first", ascending=False)
            p["signal"] = (p["probability"] >= probability_threshold) & (p["rank"] <= top_k)
            signal_lookup = p.set_index(["timestamp", "symbol"])["signal"]
            market = d.copy()
            keys = pd.MultiIndex.from_frame(market[["timestamp", "symbol"]])
            market["signal"] = signal_lookup.reindex(keys, fill_value=False).to_numpy(bool)
            market["exit_bar"] = pd.to_numeric(market[spec.hit_col], errors="coerce")
            if "liquidity_usd" not in market and "liquidity" in market:
                market["liquidity_usd"] = market["liquidity"]
            aggregate[spec.name]["capacity"] = capacity_curve(
                market,
                [20, 50, 100, 250, 500, 1000, 2000, 5000],
                costs=cost_model,
            ).to_dict(orient="records")

    report = {
        "purge_bars": purge,
        "embargo_bars": embargo_bars,
        "probability_threshold": probability_threshold,
        "top_k": top_k,
        "notional_usd": notional_usd,
        "folds": fold_reports,
        "aggregate": aggregate,
    }
    (outdir / "v3_oos_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return predictions, trades, report
