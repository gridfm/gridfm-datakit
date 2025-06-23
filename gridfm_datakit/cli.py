#!/usr/bin/env python3
"""Command-line interface for generating power flow data."""

import argparse
from gridfm_datakit.generate import (
    generate_power_flow_data_distributed,
)


def main():
    """Command-line interface for the data generation script."""
    parser = argparse.ArgumentParser(
        description="Generate power flow data for grid analysis",
    )

    parser.add_argument(
        "config",
        type=str,
        help="Path to config file",
    )

    args = parser.parse_args()
    file_paths = generate_power_flow_data_distributed(args.config)

    print("\nData generation complete.")
    print("Generated files:")
    for key, path in file_paths.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
