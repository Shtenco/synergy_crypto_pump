from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from .features import FeatureBuilder
from .labels import add_forward_labels
from .model import PumpModel, chronological_split, evaluate

app = typer.Typer(no_args_is_help=True, help="Unified crypto pump ML research CLI")


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


@app.command()
def features(source: Path, destination: Path):
    """Build V1 leakage-safe historical and cross-sectional features."""
    write_table(FeatureBuilder().transform(read_table(source)), destination)


@app.command()
def labels(source: Path, destination: Path, horizon: int = 12, threshold: float = 0.20):
    """Add V1 future max/min/end returns and binary future-pump target."""
    write_table(add_forward_labels(read_table(source), horizon=horizon, threshold=threshold), destination)


@app.command()
def train(source: Path, model_path: Path, kind: str = "auto", top_k: int = 10):
    """V1 chronological train/validation/test fit and evaluation."""
    df = read_table(source)
    train_df, val_df, test_df = chronological_split(df)
    model = PumpModel.fit(train_df, kind=kind)
    model.save(model_path)
    report = {
        "model_kind": model.kind,
        "features": len(model.features),
        "train_rows": len(train_df),
        "validation": evaluate(model, val_df, top_k=top_k) if len(val_df) else {},
        "test": evaluate(model, test_df, top_k=top_k) if len(test_df) else {},
    }
    typer.echo(json.dumps(report, indent=2))


@app.command()
def rank(source: Path, model_path: Path, destination: Path, top_k: int = 20, latest_only: bool = True):
    """V1 rank coins by predicted future-pump probability."""
    df = read_table(source)
    if latest_only:
        latest = pd.to_datetime(df["timestamp"], utc=True).max()
        df = df[pd.to_datetime(df["timestamp"], utc=True) == latest]
    model = PumpModel.load(model_path)
    write_table(model.rank(df, top_k=top_k), destination)


@app.command("features-v2")
def features_v2(
    source: Path,
    destination: Path,
    benchmark_symbol: str = "BTCUSDT",
    z_window: int = 96,
):
    """Build distilled V2 microstructure/on-chain/social/BOCD features."""
    from .v2.features import V2FeatureBuilder
    df = V2FeatureBuilder(
        benchmark_symbol=benchmark_symbol,
        z_window=z_window,
    ).transform(read_table(source))
    write_table(df, destination)


@app.command("labels-v2")
def labels_v2(source: Path, destination: Path):
    """Create strict +10%/15m, +20%/1h, +50%/6h and +100%/24h targets (5m bars)."""
    from .v2.labels import add_multi_horizon_labels
    write_table(add_multi_horizon_labels(read_table(source)), destination)


@app.command("walk-forward-v2")
def walk_forward_v2(
    source: Path,
    output_dir: Path,
    transactions: Path | None = None,
    n_splits: int = 3,
    top_k: int = 10,
    model_kind: str = "auto",
    enable_temporal: bool = False,
    enable_gnn: bool = False,
    sequence_length: int = 48,
    temporal_epochs: int = 8,
    gnn_epochs: int = 4,
):
    """Run purged expanding-window OOS evaluation for the V2 ensemble."""
    from .v2.pipeline import walk_forward_oos
    df = read_table(source)
    tx = read_table(transactions) if transactions is not None else None
    _, report = walk_forward_oos(
        df,
        output_dir=output_dir,
        n_splits=n_splits,
        top_k=top_k,
        model_kind=model_kind,
        enable_temporal=enable_temporal,
        enable_gnn=enable_gnn,
        transactions=tx,
        sequence_length=sequence_length,
        temporal_epochs=temporal_epochs,
        gnn_epochs=gnn_epochs,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("train-v2")
def train_v2(
    source: Path,
    model_path: Path,
    transactions: Path | None = None,
    model_kind: str = "auto",
    enable_temporal: bool = False,
    enable_gnn: bool = False,
    validation_fraction: float = 0.15,
):
    """Fit a final V2 ensemble using an earlier train segment and later calibration segment."""
    from .v2.pipeline import V2Ensemble
    df = read_table(source).sort_values("timestamp")
    times = sorted(pd.to_datetime(df["timestamp"], utc=True).unique())
    cut = max(1, int(len(times) * (1.0 - validation_fraction)))
    train_times, val_times = set(times[:cut]), set(times[cut:])
    purge = 288
    if cut <= purge:
        raise ValueError("train-v2 requires more than 288 pre-validation timestamps for the 24h target purge")
    train_times = set(times[: cut - purge])
    ts = pd.to_datetime(df["timestamp"], utc=True)
    train_df = df[ts.isin(train_times)]
    val_df = df[ts.isin(val_times)]
    tx = read_table(transactions) if transactions is not None else None
    model = V2Ensemble(
        model_kind=model_kind,
        enable_temporal=enable_temporal,
        enable_gnn=enable_gnn,
    ).fit(train_df, val_df, tx)
    model.save(model_path)
    typer.echo(json.dumps({
        "saved": str(model_path),
        "train_rows": len(train_df),
        "calibration_rows": len(val_df),
        "temporal": enable_temporal,
        "gnn": enable_gnn,
    }, indent=2))


@app.command("rank-v2")
def rank_v2(
    source: Path,
    model_path: Path,
    destination: Path,
    transactions: Path | None = None,
    latest_only: bool = True,
):
    """Score/rank the universe with a fitted V2 ensemble."""
    from .v2.pipeline import V2Ensemble
    df = read_table(source)
    tx = read_table(transactions) if transactions is not None else None
    model = V2Ensemble.load(model_path)
    # Keep historical context while scoring so the causal Transformer can build
    # its sequence ending at the latest row; filter only after prediction.
    pred = model.predict(df, tx)
    out = pd.concat([df[["timestamp", "symbol"]], pred], axis=1)
    if latest_only:
        latest = pd.to_datetime(out["timestamp"], utc=True).max()
        out = out[pd.to_datetime(out["timestamp"], utc=True) == latest].copy()
    for col in [c for c in pred if c.startswith("pump_probability__")]:
        if col in out:
            out[f"rank__{col}"] = out.groupby("timestamp")[col].rank(method="first", ascending=False)
    write_table(out, destination)


if __name__ == "__main__":
    app()
