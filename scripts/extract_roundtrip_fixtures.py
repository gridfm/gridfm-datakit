#!/usr/bin/env python3
"""Extract parquet roundtrip fixtures (scenarios 3 and 4) from finetuning datasets.

Writes compact parquet slices under tests/fixtures/parquet_json_roundtrip/ for use by
tests/test_parquet_to_powermodels.py (7 cases × pf/opf × 2 scenarios).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from gridfm_datakit.convert.roundtrip_check import CASES, PARQUET_TABLES, SORT_COLUMNS
from gridfm_datakit.utils.utils import n_scenario_per_partition

PF_BASE = Path("/dccstor/gridfm/powermodels_data/v4/finetuning/pf")
OPF_BASE = Path("/dccstor/gridfm/powermodels_data/v4/finetuning/opf")
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "parquet_json_roundtrip"


def extract_scenarios(
    scenarios: list[int],
    out_dir: Path,
    cases: tuple[str, ...] = CASES,
) -> None:
    partition = min(scenarios) // n_scenario_per_partition
    filters = [("scenario_partition", "=", partition)]

    for case in cases:
        for dataset, base in [("pf", PF_BASE), ("opf", OPF_BASE)]:
            src = base / case / "raw"
            if not src.is_dir():
                print(f"skip missing {src}")
                continue

            dst = out_dir / dataset / case / "raw"
            dst.mkdir(parents=True, exist_ok=True)

            for name in PARQUET_TABLES:
                path = src / f"{name}_data.parquet"
                df = pd.read_parquet(path, filters=filters, engine="pyarrow")
                sub = df.loc[df["scenario"].isin(scenarios)].copy()
                sub = (
                    sub.sort_values(["scenario", *SORT_COLUMNS[name]], kind="stable")
                    .reset_index(drop=True)
                )
                sub.to_parquet(dst / f"{name}_data.parquet", index=False)
                print(f"wrote {dataset}/{case}/{name}_data.parquet ({len(sub)} rows)")

            args_src = src / "args.log"
            if args_src.exists():
                shutil.copy2(args_src, dst / "args.log")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        type=int,
        default=[3, 4],
        help="Scenario indices to extract (default: 3 4)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Fixture output directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=list(CASES),
        help="Cases to extract (default: all 7)",
    )
    args = parser.parse_args()
    extract_scenarios(args.scenarios, args.output_dir, tuple(args.cases))


if __name__ == "__main__":
    main()
