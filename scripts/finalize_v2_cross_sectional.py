#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import duckdb

RANK_SOURCES = (
    "ret_1",
    "volume_z",
    "volume_accel",
    "quote_abs_zscore",
    "flow_imbalance",
    "slippage_imbalance",
    "tx_per_sec",
    "wallet_concentration_mean",
    "social_mentions_z",
    "social_users_z",
    "bocd_cp_prob",
    "pre_pump_activity_score",
)


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def rank_expr(source: str, alias: str) -> str:
    c = qident(source)
    a = qident(alias)
    return (
        f"CASE WHEN {c} IS NULL THEN NULL ELSE "
        f"(RANK() OVER (PARTITION BY timestamp ORDER BY {c} ASC NULLS LAST) + "
        f"(COUNT(*) OVER (PARTITION BY timestamp, {c}) - 1) / 2.0) / "
        f"COUNT({c}) OVER (PARTITION BY timestamp) END AS {a}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Recompute exact full-universe V2 cross-sectional ranks")
    ap.add_argument("input_glob")
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--memory-limit", default="5GB")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(args.output_dir / "rank_finalize.duckdb"))
    con.execute(f"PRAGMA threads={args.threads}")
    con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
    con.execute(f"PRAGMA temp_directory='{str(args.output_dir / 'tmp')}'")

    describe = con.execute(
        "DESCRIBE SELECT * FROM read_parquet(?, union_by_name=true)",
        [args.input_glob],
    ).fetchall()
    columns = [r[0] for r in describe]

    replacements = {f"xrank_{src}": rank_expr(src, f"xrank_{src}") for src in RANK_SOURCES if src in columns and f"xrank_{src}" in columns}

    coord_sources = [c for c in ("coordination_score", "cross_platform_breadth", "new_account_rate") if c in columns]
    prelude = "SELECT *, filename AS __source_file FROM read_parquet(?, union_by_name=true, filename=true)"
    if coord_sources and "social_coordination_composite" in columns:
        rank_terms = [
            f"CASE WHEN {qident(c)} IS NULL THEN NULL ELSE "
            f"(RANK() OVER (PARTITION BY timestamp ORDER BY {qident(c)} ASC NULLS LAST) + "
            f"(COUNT(*) OVER (PARTITION BY timestamp, {qident(c)}) - 1) / 2.0) / "
            f"COUNT({qident(c)}) OVER (PARTITION BY timestamp) END"
            for c in coord_sources
        ]
        coord_expr = "(" + " + ".join(f"COALESCE({x}, 0)" for x in rank_terms) + f") / {len(rank_terms)}.0"
    else:
        coord_expr = None

    select_exprs: list[str] = []
    for col in columns:
        if col in replacements:
            select_exprs.append(replacements[col])
        elif col == "social_coordination_composite" and coord_expr is not None:
            select_exprs.append(f"{coord_expr} AS {qident(col)}")
        else:
            select_exprs.append(qident(col))

    source_glob_sql = args.input_glob.replace("'", "''")
    con.execute(
        f"CREATE OR REPLACE TEMP VIEW ranked_all AS "
        f"SELECT {', '.join(select_exprs)}, "
        f"CAST(regexp_extract(filename, 'bucket-([0-9]+)', 1) AS INTEGER) AS __bucket "
        f"FROM read_parquet('{source_glob_sql}', union_by_name=true, filename=true)"
    )

    for bucket in range(64):
        n = con.execute("SELECT COUNT(*) FROM ranked_all WHERE __bucket=?", [bucket]).fetchone()[0]
        if not n:
            continue
        out = args.output_dir / f"bucket-{bucket:02d}.parquet"
        out_sql = str(out).replace("'", "''")
        con.execute(
            f"COPY (SELECT * EXCLUDE (__bucket) FROM ranked_all WHERE __bucket={bucket}) "
            f"TO '{out_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        print({"bucket": bucket, "rows": int(n), "output": str(out)})

    con.close()


if __name__ == "__main__":
    main()
