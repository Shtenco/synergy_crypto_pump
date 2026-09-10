from __future__ import annotations

from pathlib import Path

import yaml

ANCHORS = {
    "borokoko-pump-prediction": ["backtest/pipelines/CatboostRanker/pipe.py", "core/feature_type.py"],
    "bayi-target-coin": ["TargetCoinPrediction/SeqModel/modeling.py"],
    "chainoba": ["anomaly/pump_prediction.py"],
    "solana-graduation": ["README.md"],
    "solana-memcoin-ensemble": ["main.py"],
    "graduate-oracle": ["docs/methodology.md"],
    "multi-exchange-predictor": ["crypto_pump_predictor_bot.py"],
    "deep-pump-detection": ["models/transformer.py", "models/anomaly_transformer.py"],
    "crypto-anomaly-detection": ["models/anomaly_transformer.py", "models/conv_lstm.py"],
    "crypto-fraud-stgnn": ["dynamic_volume_corr_solution.py", "dynamic_numTrades_corr_solution.py"],
    "pumpwatcher": ["dataloading.py", "modules.py"],
    "market-social-signal": ["BiLstm_extractor.py"],
    "social-dataset": ["README.md"],
    "llm-social-momentum": ["README.md"],
    "bayesian-change-point": ["binance_spot_pump_dump_detector.py"],
}


def main() -> None:
    manifest = yaml.safe_load(Path("upstreams.yaml").read_text(encoding="utf-8"))
    root = Path(manifest["root"])
    names = [item["name"] for item in manifest["repositories"]]
    if len(names) != 15:
        raise SystemExit(f"expected 15 upstreams, found {len(names)}")
    failures = []
    for name in names:
        repo = root / name
        if not repo.is_dir():
            failures.append(f"{name}: repo directory missing")
            continue
        for rel in ANCHORS.get(name, []):
            if not (repo / rel).exists():
                failures.append(f"{name}: anchor missing: {rel}")
    if failures:
        raise SystemExit("\n".join(failures))
    print("V2 source audit anchors verified for 15/15 upstream repositories.")


if __name__ == "__main__":
    main()
