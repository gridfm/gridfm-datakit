#!/usr/bin/env python3
"""Run juliacall PF/OPF process-count sweeps on large GOC cases."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "benchmark_case118_dynamic_pf_sweep_juliacall.py"
GRIDS_DIR = REPO_ROOT / "gridfm_datakit" / "grids"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "large"

NETWORKS = [
    "case500_goc",
    "case2000_goc",
    "case10000_goc",
]

# (mode, n_solves, pf_fast)
BENCHMARK_SPECS: list[tuple[str, int, bool | None]] = [
    ("pf", 10_000, False),
    ("dcpf", 100_000, None),
    ("opf", 1_000, None),
    ("dcopf", 10_000, None),
]


def case_file_for_network(network: str) -> Path:
    case_file = GRIDS_DIR / f"pglib_opf_{network}_corrected.m"
    if not case_file.exists():
        raise FileNotFoundError(f"Corrected case file not found: {case_file}")
    return case_file


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extra_args = sys.argv[1:]

    for network in NETWORKS:
        case_file = case_file_for_network(network)

        for mode, n_solves, pf_fast in BENCHMARK_SPECS:
            output_csv = OUTPUT_DIR / f"benchmark_{network}_{mode}.csv"
            plots_dir = OUTPUT_DIR / "plots" / f"{network}_{mode}"

            print()
            print(f"=== Running {network} / {mode} (n_solves={n_solves:,}) ===")

            command = [
                sys.executable,
                str(BENCHMARK_SCRIPT),
                "--case-file",
                str(case_file),
                "--mode",
                mode,
                "--output-csv",
                str(output_csv),
                "--plots-dir",
                str(plots_dir),
                "--n-pfs",
                str(n_solves),
                *extra_args,
            ]
            if mode == "pf":
                command.append("--pf-fast" if pf_fast else "--no-pf-fast")

            subprocess.run(command, check=True)

    print()
    print("Large benchmark matrix complete.")
    print(f"CSVs written to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
