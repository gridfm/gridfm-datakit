#!/usr/bin/env python3
"""Convert one gridfm-datakit parquet scenario to PowerModels JSON."""

import argparse

from gridfm_datakit.convert import parquet_to_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert one parquet scenario to mixed-units PowerModels JSON.",
    )
    parser.add_argument("data_dir", help="Directory with bus/gen/branch_data.parquet")
    parser.add_argument("scenario", type=int, help="Scenario index")
    parser.add_argument("-o", "--output", required=True, help="Output .json path")
    parser.add_argument("--base-mva", type=float, default=100.0)
    args = parser.parse_args()

    parquet_to_json(args.data_dir, args.scenario, args.output, base_mva=args.base_mva)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
