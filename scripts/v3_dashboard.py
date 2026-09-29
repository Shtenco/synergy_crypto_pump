from __future__ import annotations

import argparse
import json
from pathlib import Path

from crypto_pump.v3.dashboard import write_dashboard


def main() -> None:
    p = argparse.ArgumentParser(description="Render V3 shadow-validation HTML/JSON dashboard")
    p.add_argument("--store", type=Path, default=Path("data/v3/live.sqlite"))
    p.add_argument("--output", type=Path, default=Path("data/v3/dashboard/index.html"))
    args = p.parse_args()
    payload = write_dashboard(args.store, args.output)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
