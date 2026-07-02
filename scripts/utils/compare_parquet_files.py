#!/usr/bin/env python3
"""
Script to compare parquet files from two directories.

Loads parquet files from two directories and compares columns for each matching pair.
Useful for testing new dataset generation vs old versions.

config file to use


network:
  name: "case24_ieee_rts" # Name of the power grid network (without extension)
  source: "pglib" # Data source for the grid; options: pglib, file
  # WARNING: the following parameter is only used if source is "file"
  network_dir: "scripts/grids" # if using source "file", this is the directory containing the network file (relative to the project root)

load:
  generator: "powergraph" # Name of the load generator; options: agg_load_profile, powergraph
  agg_profile: "default" # Name of the aggregated load profile
  scenarios: 10 # Number of different load scenarios to generate
  # WARNING: the following parameters are only used if generator is "agg_load_profile"
  # if using generator "powergraph", these parameters are ignored
  sigma: 0.2 # max local noise
  change_reactive_power: true # If true, changes reactive power of loads. If False, keeps the ones from the case file
  global_range: 0.4 # Range of the global scaling factor. used to set the lower bound of the scaling factor
  max_scaling_factor: 4.0 # Max upper bound of the global scaling factor
  step_size: 0.1 # Step size when finding the upper bound of the global scaling factor
  start_scaling_factor: 1.0 # Initial value of the global scaling factor

topology_perturbation:
  type: "none" # Type of topology generator; options: n_minus_k, random, none
  # WARNING: the following parameters are only used if type is not "none"
  k: 1 # Maximum number of components to drop in each perturbation
  n_topology_variants: 2 # Number of unique perturbed topologies per scenario
  elements: [branch, gen] # elements to perturb. options: branch, gen

generation_perturbation:
  type: "none" # Type of generation perturbation; options: cost_permutation, cost_perturbation, none
  # WARNING: the following parameter is only used if type is "cost_permutation"
  sigma: 1.0 # Size of range used for sampling scaling factor

admittance_perturbation:
  type: "none" # Type of admittance perturbation; options: random_perturbation, none
  # WARNING: the following parameter is only used if type is "random_perturbation"
  sigma: 0.2 # Size of range used for sampling scaling factor

settings:
  num_processes: 5 # Number of parallel processes to use
  data_dir: "./testdelll" # Directory to save generated data relative to the project root
  large_chunk_size: 1000 # Number of load scenarios processed before saving
  overwrite: true # If true, overwrites existing files, if false, appends to files
  mode: "pf" # Mode of the script; options: pf, opf. pf: power flow data where one or more operating limits – the inequality constraints defined in OPF, e.g., voltage magnitude or branch limits – may be violated. opf:  generates datapoints for training OPF solvers, with cost-optimal dispatches that satisfy all operating limits (OPF-feasible)
  include_dc_res: true # If true, also stores the results of dc power flow or dc optimal power flow
  enable_solver_logs: true # If true, write OPF/PF logs to {data_dir}/solver_log; PF fast and DCPF fast do not log.
  pf_fast: true # Whether to use fast PF solver by default (compute_ac_pf from powermodels.jl); if false, uses Ipopt-based PF. Some networks (typically large ones e.g. case10000_goc) do not work with pf_fast: true. pf_fast is faster and more accurate than the Ipopt-based PF.
  dcpf_fast: true # Whether to use fast DCPF solver by default (compute_dc_pf from PowerModels.jl)
  max_iter: 200 # Max iterations for Ipopt-based solvers
  seed: null # Seed for random number generation. If null, a random seed is generated (RECOMMENDED). To get the same data across runs, set the seed and note that ALL OTHER PARAMETERS IN THE CONFIG FILE MUST BE THE SAME.



"""

import argparse
import pandas as pd
from pathlib import Path


def get_parquet_files(directory: str) -> dict:
    """Get all parquet files organized by filename.

    Args:
        directory: Path to directory containing parquet files

    Returns:
        Dictionary mapping filenames to their full paths
    """
    dir_path = Path(directory)
    parquet_files = {}

    for file in dir_path.glob("**/*.parquet"):
        parquet_files[file.name] = file

    return parquet_files


def compare_column_content(df1: pd.DataFrame, df2: pd.DataFrame, col: str) -> dict:
    """Compare content of a specific column between two dataframes.

    Args:
        df1: First dataframe
        df2: Second dataframe
        col: Column name to compare

    Returns:
        Dictionary with comparison results
    """
    try:
        # Check if lengths match
        if len(df1) != len(df2):
            return {
                "status": "LENGTH_MISMATCH",
                "len_dir1": len(df1),
                "len_dir2": len(df2),
            }

        # Compare values
        if df1[col].dtype != df2[col].dtype:
            return {
                "status": "TYPE_MISMATCH",
                "type_dir1": str(df1[col].dtype),
                "type_dir2": str(df2[col].dtype),
            }

        # Check if values are equal
        are_equal = df1[col].equals(df2[col])

        if are_equal:
            return {"status": "IDENTICAL"}
        else:
            # Find differences
            differences = (df1[col] != df2[col]).sum()
            return {
                "status": "VALUES_DIFFER",
                "num_differences": differences,
                "percent_different": (differences / len(df1)) * 100,
            }
    except Exception as e:
        return {
            "status": "ERROR",
            "error": str(e),
        }


def compare_parquet_files(dir1: str, dir2: str, verbose: bool = False) -> None:
    """Compare parquet files from two directories, including column content.

    Args:
        dir1: Path to first directory
        dir2: Path to second directory
        verbose: If True, print detailed information about each file
    """
    files_dir1 = get_parquet_files(dir1)
    files_dir2 = get_parquet_files(dir2)

    print(f"\n{'=' * 80}")
    print("Comparing parquet files from two directories (including content)")
    print(f"{'=' * 80}")
    print(f"Directory 1: {dir1}")
    print(f"Directory 2: {dir2}")
    print(f"{'=' * 80}\n")

    # Find matching files
    common_files = set(files_dir1.keys()) & set(files_dir2.keys())
    only_in_dir1 = set(files_dir1.keys()) - set(files_dir2.keys())
    only_in_dir2 = set(files_dir2.keys()) - set(files_dir1.keys())

    # Report missing files
    if only_in_dir1:
        print(f"Files only in directory 1: {sorted(only_in_dir1)}\n")

    if only_in_dir2:
        print(f"Files only in directory 2: {sorted(only_in_dir2)}\n")

    if not common_files:
        print("No matching parquet files found in both directories!")
        return

    print(f"Found {len(common_files)} matching parquet file(s) to compare:\n")

    comparison_results = {}

    for filename in sorted(common_files):
        print(f"{'-' * 80}")
        print(f"File: {filename}")
        print(f"{'-' * 80}")

        try:
            # Load parquet files
            df1 = pd.read_parquet(files_dir1[filename])
            df2 = pd.read_parquet(files_dir2[filename])

            # sort by load_scenario_idx if exists
            if "load_scenario_idx" in df1.columns:
                print(f"Sorting {filename} by load_scenario_idx")
                df1 = df1.sort_values(
                    by="load_scenario_idx",
                    kind="stable",
                ).reset_index(drop=True)
                df2 = df2.sort_values(
                    by="load_scenario_idx",
                    kind="stable",
                ).reset_index(drop=True)

            else:
                print(
                    f"No load_scenario_idx column found in {filename}, using sort by scenario",
                )
                df1 = df1.sort_values(by="scenario", kind="stable").reset_index(
                    drop=True,
                )
                df2 = df2.sort_values(by="scenario", kind="stable").reset_index(
                    drop=True,
                )

            cols1 = set(df1.columns)
            cols2 = set(df2.columns)

            # Find differences
            only_in_df1 = cols1 - cols2
            only_in_df2 = cols2 - cols1
            common_cols = sorted(cols1 & cols2)

            if verbose:
                print(f"  Shape dir1: {df1.shape}")
                print(f"  Shape dir2: {df2.shape}")
                print(f"  Columns dir1: {sorted(df1.columns)}")
                print(f"  Columns dir2: {sorted(df2.columns)}\n")

            # Report column structure differences
            if only_in_df1:
                print(f"  ✗ Columns ONLY in dir1: {sorted(only_in_df1)}")

            if only_in_df2:
                print(f"  ✗ Columns ONLY in dir2: {sorted(only_in_df2)}")

            if only_in_df1 or only_in_df2:
                print(f"  ✓ Common columns: {len(common_cols)}\n")
            else:
                print(f"  ✓ Column structure: IDENTICAL ({len(common_cols)} columns)\n")

            # Compare content of common columns
            content_issues = []
            identical_cols = []

            for col in common_cols:
                comp_result = compare_column_content(df1, df2, col)

                if comp_result["status"] == "IDENTICAL":
                    identical_cols.append(col)
                else:
                    content_issues.append((col, comp_result))

            # Report content differences
            if content_issues:
                print(
                    f"  ✗ Content differences found in {len(content_issues)} column(s):",
                )
                for col, result in content_issues:
                    if result["status"] == "LENGTH_MISMATCH":
                        print(
                            f"      - {col}: LENGTH MISMATCH (dir1: {result['len_dir1']} rows, dir2: {result['len_dir2']} rows)",
                        )
                    elif result["status"] == "TYPE_MISMATCH":
                        print(
                            f"      - {col}: TYPE MISMATCH (dir1: {result['type_dir1']}, dir2: {result['type_dir2']})",
                        )
                    elif result["status"] == "VALUES_DIFFER":
                        print(
                            f"      - {col}: {result['num_differences']} rows differ ({result['percent_different']:.2f}%)",
                        )
                    elif result["status"] == "ERROR":
                        print(f"      - {col}: ERROR - {result['error']}")
                print()
            else:
                print(
                    f"  ✓ All common column content: IDENTICAL ({len(identical_cols)} columns)\n",
                )

            # Determine overall status
            if only_in_df1 or only_in_df2 or content_issues:
                status = "DIFFERENT"
            else:
                status = "IDENTICAL"

            comparison_results[filename] = {
                "only_in_dir1": sorted(only_in_df1),
                "only_in_dir2": sorted(only_in_df2),
                "common_cols": len(common_cols),
                "content_issues": len(content_issues),
                "identical_cols": len(identical_cols),
                "status": status,
            }

        except Exception as e:
            print(f"  ✗ Error reading files: {e}\n")
            comparison_results[filename] = {"status": "ERROR", "error": str(e)}

    # Summary
    print(f"{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}\n")

    identical_count = sum(
        1 for r in comparison_results.values() if r.get("status") == "IDENTICAL"
    )
    different_count = sum(
        1 for r in comparison_results.values() if r.get("status") == "DIFFERENT"
    )
    error_count = sum(
        1 for r in comparison_results.values() if r.get("status") == "ERROR"
    )

    print(f"Total files compared: {len(comparison_results)}")
    print(f"  - Identical (structure & content): {identical_count}")
    print(f"  - Different: {different_count}")
    print(f"  - Errors: {error_count}\n")

    if different_count > 0:
        print("Files with differences:")
        for filename, result in sorted(comparison_results.items()):
            if result.get("status") == "DIFFERENT":
                print(f"\n  {filename}:")
                if result["only_in_dir1"]:
                    print(f"    Only in dir1: {result['only_in_dir1']}")
                if result["only_in_dir2"]:
                    print(f"    Only in dir2: {result['only_in_dir2']}")
                if result.get("content_issues", 0) > 0:
                    print(f"    Content issues: {result['content_issues']} column(s)")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare parquet files from two directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two directories
  python scripts/compare_parquet_files.py /path/to/old/data /path/to/new/data

  # Compare with verbose output
  python compare_parquet_files.py /path/to/old/data /path/to/new/data -v
        """,
    )

    parser.add_argument(
        "dir1",
        type=str,
        help="Path to first directory containing parquet files",
    )

    parser.add_argument(
        "dir2",
        type=str,
        help="Path to second directory containing parquet files",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed information including full column lists and shapes",
    )

    args = parser.parse_args()

    # Check if directories exist
    if not Path(args.dir1).exists():
        print(f"Error: Directory '{args.dir1}' does not exist")
        return

    if not Path(args.dir2).exists():
        print(f"Error: Directory '{args.dir2}' does not exist")
        return

    compare_parquet_files(args.dir1, args.dir2, verbose=args.verbose)


if __name__ == "__main__":
    main()
