"""
Sample and save the per-dataset parquet used by the comparison plots.

Finds the smallest scenario count across all datasets and downsamples every
dataset to it, so each contributes equally to the diversity metrics.

NOTE ON REPRODUCIBILITY: the scenario subset is drawn with np.random.choice.
Re-running this script produces a *different* subset and therefore does not
reproduce the published figures, even with --seed (the original run recorded no
seed). The committed snapshot in dataset_sampled/ is the authoritative input for
reproducing the paper -- see README.md. --seed only makes future runs
self-consistent.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Sample raw generated datasets down to a common scenario count.",
)
parser.add_argument(
    "--base-path",
    required=True,
    help="Directory holding the raw per-library datasets (see `datasets` below).",
)
parser.add_argument(
    "--output-dir",
    default=str(Path(__file__).parent / "dataset_sampled"),
    help="Where to write the sampled parquet.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Seed for the scenario sampler. Makes a run repeatable, but cannot "
         "recover the published snapshot.",
)
args = parser.parse_args()

if args.seed is not None:
    np.random.seed(args.seed)

base_path = Path(args.base_path)
output_folder = Path(args.output_dir)
output_folder.mkdir(parents=True, exist_ok=True)

# Define datasets with their paths and output names
datasets = [
    {
        "name": "gridfm_datakit_opf",
        "bus_data_path": base_path / "opf_baseline_perturbations_nov_12/case118_ieee/raw/bus_data.parquet",
        "gen_data_path": base_path / "opf_baseline_perturbations_nov_12/case118_ieee/raw/gen_data.parquet",
    },
    {
        "name": "gridfm_datakit_pf",
        "bus_data_path": base_path / "baseline_perturbations_nov_12/case118_ieee/raw/bus_data.parquet",
        "gen_data_path": base_path / "baseline_perturbations_nov_12/case118_ieee/raw/gen_data.parquet",
        "branch_data_path": base_path / "baseline_perturbations_nov_12/case118_ieee/raw/branch_data.parquet",
    },
    {
        "name": "opfdata",
        "bus_data_path": base_path / "opfdata/converted/bus_data.parquet",
        "gen_data_path": base_path / "opfdata/converted/gen_data.parquet",
    },
    {
        "name": "pglearn",
        "bus_data_path": base_path / "pglearn/PGLearn-Small-118_ieee-nminus1/converted/bus_data.parquet",
        "gen_data_path": base_path / "pglearn/PGLearn-Small-118_ieee-nminus1/converted/gen_data.parquet",
    },
    {
        "name": "opflearn",
        "bus_data_path": base_path / "opflearn/converted/bus_data.parquet",
        "gen_data_path": base_path / "opflearn/converted/gen_data.parquet",
    },
    {
        "name": "pfdelta",
        "bus_data_path": base_path / "pfdelta/case118_ieee_n_minus_one/converted/bus_data.parquet",
        "gen_data_path": base_path / "pfdelta/case118_ieee_n_minus_one/converted/gen_data.parquet",
        "branch_data_path": base_path / "pfdelta/case118_ieee_n_minus_one/converted/branch_data.parquet",
    },
]

# Step 1: Find min number of scenarios across all datasets
print("Finding min number of scenarios across datasets...")
min_scenarios = float('inf')
for dataset in datasets:
    df = pd.read_parquet(dataset["bus_data_path"])
    n_scenarios = df["scenario"].max() + 1
    print(f"  {dataset['name']}: {n_scenarios} scenarios")
    min_scenarios = min(min_scenarios, n_scenarios)

print(f"\nMin scenarios to sample: {min_scenarios}")

# Step 2: Sample and save each dataset
print(f"\nSampling datasets to {output_folder}...")
for dataset in datasets:
    bus_path = dataset["bus_data_path"]
    gen_path = dataset["gen_data_path"]
    branch_path = dataset.get("branch_data_path")
    name = dataset["name"]
    
    print(f"  Loading {name}...")
    
    # Load bus data
    df_bus = pd.read_parquet(bus_path)
    n_scenarios = df_bus["scenario"].max() + 1
    
    # Sample scenarios if needed
    if n_scenarios > min_scenarios:
        scenarios = np.random.choice(n_scenarios, size=int(min_scenarios), replace=False)
        df_bus = df_bus[df_bus["scenario"].isin(scenarios)]
    
    # Save bus data
    output_bus_path = output_folder / f"{name}_bus_data.parquet"
    df_bus.to_parquet(output_bus_path)
    print(f"    Saved bus_data: {df_bus['scenario'].nunique()} scenarios")
    
    # Load and save gen data
    df_gen = pd.read_parquet(gen_path)
    
    # Apply same sampling to gen data
    if n_scenarios > min_scenarios:
        df_gen = df_gen[df_gen["scenario"].isin(scenarios)]
    
    output_gen_path = output_folder / f"{name}_gen_data.parquet"
    df_gen.to_parquet(output_gen_path)
    print(f"    Saved gen_data: {df_gen['scenario'].nunique()} scenarios")
    
    # Load and save branch data (only for datasets that have it)
    if branch_path:
        df_branch = pd.read_parquet(branch_path)
        
        # Apply same sampling to branch data
        if n_scenarios > min_scenarios:
            df_branch = df_branch[df_branch["scenario"].isin(scenarios)]
        
        output_branch_path = output_folder / f"{name}_branch_data.parquet"
        df_branch.to_parquet(output_branch_path)
        print(f"    Saved branch_data: {df_branch['scenario'].nunique()} scenarios")

print(f"\nDone! Saved datasets to {output_folder}")
