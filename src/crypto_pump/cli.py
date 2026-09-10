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
    """Build leakage-safe historical and cross-sectional features."""
    write_table(FeatureBuilder().transform(read_table(source)), destination)


@app.command()
def labels(source: Path, destination: Path, horizon: int = 12, threshold: float = 0.20):
    """Add future max/min/end returns and binary future-pump target."""
    write_table(add_forward_labels(read_table(source), horizon=horizon, threshold=threshold), destination)


@app.command()
def train(source: Path, model_path: Path, kind: str = "auto", top_k: int = 10):
    """Chronological train/validation/test fit and evaluation."""
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
    """Rank coins by predicted future-pump probability."""
    df = read_table(source)
    if latest_only:
        latest = pd.to_datetime(df["timestamp"], utc=True).max()
        df = df[pd.to_datetime(df["timestamp"], utc=True) == latest]
    model = PumpModel.load(model_path)
    write_table(model.rank(df, top_k=top_k), destination)


if __name__ == "__main__":
    app()
