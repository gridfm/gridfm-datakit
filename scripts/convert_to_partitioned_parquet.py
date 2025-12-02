#!/usr/bin/env python3
"""
Convert old non-partitioned parquet files to new partitioned format.

This script converts existing parquet data files to the new partitioned format
with scenario_partition column (100 scenarios per partition) and creates the
n_scenarios.txt metadata file.

Usage:
    python scripts/convert_to_partitioned_parquet.py /path/to/raw/data
"""

import os
import pandas as pd
import argparse
from pathlib import Path
import shutil


def convert_parquet_file(input_path: str, output_path: str, backup: bool = True) -> int:
    """Convert a single parquet file to partitioned format.
    
    Args:
        input_path: Path to old non-partitioned parquet file
        output_path: Path to new partitioned parquet directory
        backup: Whether to backup the original file
        
    Returns:
        Number of unique scenarios in the data
    """
    print(f"  Reading: {os.path.basename(input_path)}")
    
    # Read old format
    df = pd.read_parquet(input_path, engine="pyarrow")
    
    if df.empty:
        raise ValueError(f"Empty parquet file: {input_path}")
    
    if "scenario" not in df.columns:
        raise ValueError(f"No 'scenario' column in {input_path}")
    
    # Get unique scenarios
    unique_scenarios = df["scenario"].unique()
    n_scenarios = len(unique_scenarios)
    
    print(f"    Found {n_scenarios} unique scenarios")
    
    # Add partition column for scenario-based partitioning (100 scenarios per partition)
    df["scenario_partition"] = (df["scenario"] // 100).astype("int64")
    
    # Backup old file if exists
    if os.path.exists(output_path) and backup:
        backup_path = output_path + ".backup"
        print(f"    Backing up existing directory to {backup_path}")
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        shutil.move(output_path, backup_path)
    
    # Write partitioned format
    print(f"  Writing to: {output_path}")
    df.to_parquet(
        output_path,
        partition_cols=["scenario_partition"],
        engine="pyarrow",
        index=False,
    )
    
    print(f"    ✓ Converted to partitioned format")
    
    return int(n_scenarios)


def convert_data_directory(data_dir: str, backup: bool = True) -> bool:
    """Convert all parquet files in a data directory to partitioned format.
    
    Args:
        data_dir: Path to data directory containing parquet files
        backup: Whether to backup original files
        
    Returns:
        True if successful
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        return False
    
    # Files to convert
    files_to_convert = {
        "bus_data.parquet": "bus_data.parquet",
        "branch_data.parquet": "branch_data.parquet",
        "gen_data.parquet": "gen_data.parquet",
        "y_bus_data.parquet": "y_bus_data.parquet",
        "runtime_data.parquet": "runtime_data.parquet",
    }
    
    n_scenarios = None
    
    for old_name, new_name in files_to_convert.items():
        old_path = data_dir / old_name
        
        if not old_path.exists():
            print(f"  Skipping (not found): {old_name}")
            continue
        
        # Check if already partitioned (directory vs file)
        if old_path.is_dir():
            # Check if already has partition structure
            partition_dirs = list(old_path.glob("scenario_partition=*"))
            if partition_dirs:
                print(f"  Skipping (already partitioned): {old_name}")
                continue
            print(f"  ERROR: {old_name} is a directory but not properly partitioned")
            return False
        
        new_path = data_dir / new_name
        
        try:
            count = convert_parquet_file(str(old_path), str(new_path), backup=backup)
            
            if n_scenarios is None:
                n_scenarios = count
            elif n_scenarios != count:
                print(f"  WARNING: Scenario count mismatch in {old_name}")
                print(f"    Expected: {n_scenarios}, Got: {count}")
        
        except Exception as e:
            print(f"  ERROR converting {old_name}: {e}")
            return False
    
    if n_scenarios is None:
        print("ERROR: No parquet files found to convert")
        return False
    
    # Write metadata file
    n_scenarios_file = data_dir / "n_scenarios.txt"
    print(f"\nWriting metadata: {n_scenarios_file}")
    with open(n_scenarios_file, "w") as f:
        f.write(str(n_scenarios))
    print(f"  Total scenarios: {n_scenarios}")
    
    print("\n✓ Conversion complete!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert old non-partitioned parquet files to new partitioned format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert data directory
  python scripts/convert_to_partitioned_parquet.py /path/to/raw/data
  
  # Convert without backup
  python scripts/convert_to_partitioned_parquet.py /path/to/raw/data --no-backup
        """,
    )
    
    parser.add_argument(
        "data_dir",
        type=str,
        help="Path to data directory containing parquet files",
    )
    
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not backup original files (default: backup)",
    )
    
    args = parser.parse_args()
    
    print(f"Converting data directory: {args.data_dir}\n")
    
    success = convert_data_directory(args.data_dir, backup=not args.no_backup)
    
    if not success:
        print("\n✗ Conversion failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

