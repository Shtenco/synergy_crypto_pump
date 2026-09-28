from __future__ import annotations
from typing import Any, Mapping

CONTRACT_ID = "synergy.trading.crypto-anomaly-score/v1"
AUTHORITY = "crypto_market_anomaly"


def crypto_anomaly_score_payload(
    *,
    observation_id: str,
    timestamp: str,
    symbol: str,
    pump_probability: float,
    rank: int,
    model_ref: str,
    feature_snapshot_sha256: str,
) -> dict[str, Any]:
    if not observation_id.strip() or not symbol.strip() or not model_ref.strip():
        raise ValueError("observation_id, symbol and model_ref are required")
    probability = float(pump_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("pump_probability must be within [0,1]")
    if int(rank) <= 0:
        raise ValueError("rank must be positive")
    digest = feature_snapshot_sha256.lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid feature_snapshot_sha256")
    return {
        "observation_id": observation_id,
        "timestamp": timestamp,
        "symbol": symbol.upper(),
        "pump_probability": probability,
        "rank": int(rank),
        "model_ref": model_ref,
        "feature_snapshot_sha256": digest,
        "authorization": False,
        "requires_domain_validation": True,
    }


def crypto_anomaly_score_from_ranked_row(
    row: Mapping[str, Any],
    *,
    observation_id: str,
    model_ref: str,
    feature_snapshot_sha256: str,
) -> dict[str, Any]:
    return crypto_anomaly_score_payload(
        observation_id=observation_id,
        timestamp=str(row["timestamp"]),
        symbol=str(row["symbol"]),
        pump_probability=float(row["pump_probability"]),
        rank=int(row["rank"]),
        model_ref=model_ref,
        feature_snapshot_sha256=feature_snapshot_sha256,
    )
