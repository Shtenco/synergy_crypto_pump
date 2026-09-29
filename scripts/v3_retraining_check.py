from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from crypto_pump.v3.retraining import decide_retraining


def main() -> None:
    p = argparse.ArgumentParser(description="Decide whether the V3 model should be retrained")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--health-json", type=Path)
    args = p.parse_args()

    frame = pd.read_parquet(args.dataset)
    manifest = json.loads(args.manifest.read_text()) if args.manifest and args.manifest.exists() else None
    health = json.loads(args.health_json.read_text()) if args.health_json and args.health_json.exists() else None
    decision = decide_retraining(frame, previous_manifest=manifest, latest_health=health)
    print(json.dumps(asdict(decision), indent=2, default=str))
    raise SystemExit(0 if decision.should_retrain else 2)


if __name__ == "__main__":
    main()
