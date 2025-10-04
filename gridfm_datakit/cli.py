#!/usr/bin/env python3
"""Command-line interface for generating and validating power flow data."""

import argparse
import sys
import yaml
from pathlib import Path
from gridfm_datakit.generate import (
    generate_power_flow_data_distributed,
)
from gridfm_datakit.validation import validate_generated_data


def validate_data_directory(data_path, n_scenarios=100):
    """
    Validate generated power flow data in a directory.

    Args:
        data_path (str): Path to directory containing generated CSV files
        n_scenarios (int): Number of scenarios to sample for validation (0 = all scenarios)

    Returns:
        bool: True if all validations pass, False otherwise
    """
    data_path = Path(data_path)

    # Expected file names for validation
    expected_files = {
        "bus_data": "bus_data.csv",
        "branch_data": "branch_data.csv",
        "gen_data": "gen_data.csv",
        "y_bus_data": "y_bus_data.csv",
    }

    # read mode from args_log
    try:
        with open(data_path / "args.log", "r") as f:
            lines = f.readlines()
            # Skip the first two lines (empty line and timestamp)
            yaml_content = "".join(lines[2:])
            args = yaml.safe_load(yaml_content)
        mode = args["settings"]["mode"]
        print(f"   Found mode: {mode}")
    except Exception as e:
        print(f"   Could not read mode from args.log: {e}")
        print("   Using default mode: unsecure")
        mode = "unsecure"

    # Check if all required files exist
    file_paths = {}
    missing_files = []

    for key, filename in expected_files.items():
        file_path = data_path / filename
        if file_path.exists():
            file_paths[key] = str(file_path)
        else:
            missing_files.append(filename)

    if missing_files:
        print(f"ERROR: Missing required files: {', '.join(missing_files)}")
        print(f"   Expected files in {data_path}:")
        for filename in expected_files.values():
            print(f"   - {filename}")
        return False

    print(f"Found all required data files in {data_path}")

    try:
        # Run validation
        print(f"Running validation tests (mode: {mode})...")
        validate_generated_data(file_paths, mode, n_scenarios=n_scenarios)
        print("All validation tests passed!")
        return True

    except AssertionError as e:
        print(f"ERROR: Validation failed: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Error during validation: {e}")
        return False


def main():
    """Command-line interface for generating and validating power flow data."""
    parser = argparse.ArgumentParser(
        description="Generate or validate power flow data for grid analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate data from config file
  gridfm-datakit generate config.yaml

  # Validate existing data (sample 100 scenarios)
  gridfm-datakit validate /path/to/data/

  # Validate with custom scenario sampling
  gridfm-datakit validate /path/to/data/ --n-scenarios 50

  # Validate all scenarios
  gridfm-datakit validate /path/to/data/ --n-scenarios 0
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate power flow data from configuration file",
    )
    generate_parser.add_argument(
        "config",
        type=str,
        help="Path to configuration file (.yaml)",
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate existing generated power flow data",
    )
    validate_parser.add_argument(
        "data_path",
        type=str,
        help="Path to directory containing generated CSV files (bus_data.csv, branch_data.csv, gen_data.csv, y_bus_data.csv)",
    )
    validate_parser.add_argument(
        "--n-scenarios",
        type=int,
        default=100,
        help="Number of scenarios to sample for validation (default: 100). Use 0 to validate all scenarios.",
    )

    args = parser.parse_args()

    if args.command == "generate":
        print(f"Generating power flow data from {args.config}...")
        file_paths = generate_power_flow_data_distributed(args.config)

        print("\nData generation complete!")
        print("Generated files:")
        for key, path in file_paths.items():
            print(f"  - {key}: {path}")

    elif args.command == "validate":
        print(f"Validating data in {args.data_path}...")
        if args.n_scenarios > 0:
            print(f"Sampling {args.n_scenarios} scenarios for validation...")
        else:
            print("Validating all scenarios...")
        success = validate_data_directory(args.data_path, n_scenarios=args.n_scenarios)

        if success:
            print("\nData validation completed successfully!")
            sys.exit(0)
        else:
            print("\nData validation failed!")
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
