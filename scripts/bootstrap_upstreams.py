#!/usr/bin/env python3
"""Clone/update the 15 research repositories declared in upstreams.yaml.

Third-party code stays outside this repository's tracked tree. A lock file with
resolved commit SHAs is produced locally for reproducibility.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "upstreams.yaml"


def run(*args: str, cwd: Path | None = None) -> str:
    p = subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True)
    return p.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--only", nargs="*", default=None, help="Optional upstream names")
    parser.add_argument("--update", action="store_true", help="Fetch/update existing clones")
    args = parser.parse_args()

    cfg = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    root = ROOT / cfg.get("root", "external/upstreams")
    root.mkdir(parents=True, exist_ok=True)
    selected = set(args.only or [])
    lock: dict[str, dict[str, str]] = {}
    failures: list[tuple[str, str]] = []

    for item in cfg["repositories"]:
        name, url = item["name"], item["url"]
        if selected and name not in selected:
            continue
        dst = root / name
        try:
            if (dst / ".git").exists():
                if args.update:
                    print(f"[update] {name}")
                    run("git", "fetch", "--all", "--prune", cwd=dst)
                    branch = run("git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD", cwd=dst).split("/")[-1]
                    run("git", "checkout", branch, cwd=dst)
                    run("git", "pull", "--ff-only", cwd=dst)
                else:
                    print(f"[keep]   {name}")
            else:
                print(f"[clone]  {name}")
                run("git", "clone", "--depth", str(args.depth), url, str(dst))

            sha = run("git", "rev-parse", "HEAD", cwd=dst)
            remote = run("git", "remote", "get-url", "origin", cwd=dst)
            lock[name] = {"repo": item["repo"], "url": remote, "sha": sha, "role": item["role"]}
        except Exception as exc:  # continue so one stale upstream doesn't block all research
            failures.append((name, str(exc)))
            print(f"[FAIL]   {name}: {exc}")

    (ROOT / "upstreams.lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nResolved {len(lock)} upstreams; failures={len(failures)}")
    if failures:
        for name, err in failures:
            print(f" - {name}: {err}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
