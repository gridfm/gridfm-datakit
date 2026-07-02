#!/usr/bin/env python3
"""
Convert old non-partitioned parquet files to new partitioned format.

Usage:
    python scripts/convert_to_partitioned_parquet.py /path/to/raw/data
"""

import pandas as pd
import argparse
from pathlib import Path


def convert_data_directory(data_dir: str):
    """Convert all parquet files in a data directory to partitioned format."""
    data_dir = Path(data_dir)
    output_dir = Path(str(data_dir) + "_chunked")
    output_dir.mkdir(exist_ok=True)

    # Files to convert
    files = [
        "bus_data.parquet",
        "branch_data.parquet",
        "gen_data.parquet",
        "y_bus_data.parquet",
        "runtime_data.parquet",
    ]

    n_scenarios = None

    for filename in files:
        input_path = data_dir / filename
        if not input_path.exists():
            continue

        print(f"Converting {filename}...")
        df = pd.read_parquet(input_path, engine="pyarrow")

        # Add partition column (100 scenarios per partition)
        df["scenario_partition"] = (df["scenario"] // 100).astype("int64")

        # Write partitioned format to output directory
        output_path = output_dir / filename
        df.to_parquet(
            output_path,
            partition_cols=["scenario_partition"],
            engine="pyarrow",
            index=False,
        )

        # Track scenario count from first file
        if n_scenarios is None:
            n_scenarios = int(df["scenario"].max()) + 1
        else:
            assert n_scenarios == int(df["scenario"].max()) + 1, (
                "Scenario count mismatch"
            )

    # Write metadata file
    n_scenarios_file = output_dir / "n_scenarios.txt"
    with open(n_scenarios_file, "w") as f:
        f.write(str(n_scenarios))

    print(f"✓ Conversion complete! Total scenarios: {n_scenarios}")
    print(f"✓ Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert parquet files to partitioned format (100 scenarios/partition)",
    )
    parser.add_argument("data_dir", type=str, help="Path to data directory")
    args = parser.parse_args()

    convert_data_directory(args.data_dir)
    return 0


if __name__ == "__main__":
    exit(main())
