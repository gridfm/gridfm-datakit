#!/usr/bin/env python3
"""Convert the first N scenarios from finetuning pf/opf datasets to PowerModels JSON."""

import argparse
from os import cpu_count
from pathlib import Path

from gridfm_datakit.convert.batch_parquet_to_powermodels import convert_case

PF_BASE = "/dccstor/gridfm/powermodels_data/v4/finetuning/pf"
OPF_BASE = "/dccstor/gridfm/powermodels_data/v4/finetuning/opf"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-samples", type=int, default=10_000)
    p.add_argument("--workers", type=int, default=cpu_count() or 1)
    p.add_argument("--pf-base", default=PF_BASE)
    p.add_argument("--opf-base", default=OPF_BASE)
    args = p.parse_args()

    for base in (args.pf_base, args.opf_base):
        for case_dir in sorted(Path(base).iterdir()):
            if not (case_dir / "raw" / "bus_data.parquet").exists():
                continue
            print(case_dir.name)
            convert_case(case_dir, max_samples=args.max_samples, workers=args.workers)


if __name__ == "__main__":
    main()
