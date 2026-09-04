#!/usr/bin/env python3
"""Derive wall-at-best-p summaries from the 56 paper CSVs.

Best p = argmin(pf_elapsed_s / n_pfs). Never use mean_pf_runtime_s.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "outputs_julia" / "full_matrix"
LONG_HEADER = [
    "setup",
    "grid",
    "solver",
    "p",
    "n_pfs",
    "wall_s",
    "wall_us",
    "init_s",
    "failed_count",
    "successful_count",
    "mean_pf_solve_time_s",
    "mean_parse_time_s",
]


def rows_from_csv(path: Path, setup: str, grid: str, solver: str) -> dict:
    best = None
    with path.open(newline="") as fh:
        for rec in csv.DictReader(fh):
            n = int(float(rec["n_pfs"]))
            wall_s = float(rec["pf_elapsed_s"])
            metric = wall_s / n
            cand = {
                "setup": setup,
                "grid": grid,
                "solver": solver,
                "p": int(rec["p"]),
                "n_pfs": n,
                "wall_s": wall_s,
                "wall_us": metric * 1e6,
                "init_s": float(rec["init_elapsed_s"]),
                "failed_count": int(float(rec["failed_count"])),
                "successful_count": int(float(rec["successful_count"])),
                "mean_pf_solve_time_s": float(rec["mean_pf_solve_time_s"]),
                "mean_parse_time_s": float(rec["mean_parse_time_s"] or 0.0),
            }
            if best is None or metric < best["wall_us"] / 1e6:
                best = cand
    if best is None:
        raise SystemExit(f"empty CSV: {path}")
    return best


def collect() -> list[dict]:
    out = []
    for setup in ("setup1", "setup2"):
        for scope in ("small", "large"):
            d = ROOT / scope / setup
            for path in sorted(d.glob("benchmark_*_*.csv")):
                name = path.stem[len("benchmark_") :]
                grid, solver = name.rsplit("_", 1)
                out.append(rows_from_csv(path, setup, grid, solver))
    out.sort(key=lambda r: (r["setup"], r["grid"], r["solver"]))
    return out


def check(rows: list[dict]) -> None:
    existing = ROOT / "wall_at_best_p_long.csv"
    with existing.open(newline="") as fh:
        old = list(csv.DictReader(fh))
    if len(old) != len(rows):
        raise SystemExit(f"row count {len(rows)} != {len(old)} in {existing}")
    for a, b in zip(rows, old):
        if a["setup"] != b["setup"] or a["grid"] != b["grid"] or a["solver"] != b["solver"]:
            raise SystemExit(f"order mismatch: {a} vs {b}")
        if int(a["p"]) != int(float(b["p"])):
            raise SystemExit(f"best p mismatch {a['grid']} {a['solver']}: {a['p']} vs {b['p']}")
        if abs(a["wall_s"] - float(b["wall_s"])) > 1e-6:
            raise SystemExit(
                f"wall_s mismatch {a['grid']} {a['solver']}: {a['wall_s']} vs {b['wall_s']}"
            )
    print(f"OK: {len(rows)} best-p rows match {existing}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="verify against committed long CSV")
    args = p.parse_args()
    rows = collect()
    if args.check:
        check(rows)
        return
    writer = csv.DictWriter(
        (ROOT / "wall_at_best_p_long.csv").open("w", newline=""),
        fieldnames=LONG_HEADER,
    )
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
