"""
Generate branch-loading histograms (paper Figure 5) and the overload statistics
quoted in Section 5.

Branch loading is max(S_from, S_to) / rate_a, with
S = sqrt(P^2 + Q^2) at each end of the branch.

Two statistics are reported per dataset, both averaged over scenarios:
  - the share of branches overloaded (loading > 1) within a scenario
  - the share of scenarios having at least one overloaded branch

Paper values: gridfm-datakit 1.2% / 79%; PFDelta 8% / all scenarios.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib

# Force a non-interactive backend before pyplot is imported. On macOS the default
# "macosx" backend applies Retina scaling to the saved vector output, which changes
# the rendered figure; Agg keeps it identical to the published PDFs.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from pathlib import Path  # noqa: E402

# Datasets carrying branch data, and the figure each one is saved as.
DATASETS = {
    "gridfm_datakit_pf": "branch_loading_datakit",
    "pfdelta": "branch_loading_pfdelta",
}


def compute_loading(branch_data):
    """Branch loading as a fraction of rate_a."""
    s_from = np.sqrt(
        branch_data["pf"].to_numpy() ** 2 + branch_data["qf"].to_numpy() ** 2,
    )
    s_to = np.sqrt(
        branch_data["pt"].to_numpy() ** 2 + branch_data["qt"].to_numpy() ** 2,
    )
    rate_a = branch_data["rate_a"].to_numpy()

    return np.maximum(s_from, s_to) / rate_a


def overload_stats(branch_data):
    """Return (mean % overloaded branches, % scenarios with >=1 overload)."""
    overloads_per_scenario = branch_data.groupby("scenario")["loading"].apply(
        lambda x: (x > 1).mean(),
    )

    return (
        overloads_per_scenario.mean() * 100,
        (overloads_per_scenario > 0).mean() * 100,
    )


def plot_branch_loading(name, output_name, data_dir, output_dir):
    branch_file = Path(data_dir) / f"{name}_branch_data.parquet"
    if not branch_file.exists():
        print(f"  Warning: {branch_file} not found, skipping {name}")
        return

    print(f"Loading {name}...")
    branch_data = pd.read_parquet(
        branch_file,
        columns=["scenario", "pf", "qf", "pt", "qt", "rate_a"],
    )
    branch_data["loading"] = compute_loading(branch_data)

    pct_branches, pct_scenarios = overload_stats(branch_data)
    print(f"  percentage of overloaded branches per scenario: {pct_branches}")
    print(f"  percentage of scenarios with overloaded branches: {pct_scenarios}")

    plt.figure(figsize=(7, 4))
    sns.histplot(branch_data["loading"], bins=60, kde=False)
    plt.xlabel("Branch Loading", fontsize=16)
    plt.ylabel("Count", fontsize=16)
    plt.yscale("log")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{output_name}.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {output_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate branch-loading histograms (paper Figure 5).",
    )
    parser.add_argument("--datasets", nargs="*", help="Subset of datasets to plot.")
    parser.add_argument("--output-dir", default="comparison_plots", help="Output directory.")
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).parent / "dataset_sampled"),
        help="Directory containing sampled datasets.",
    )
    args = parser.parse_args()

    selected = (
        {k: v for k, v in DATASETS.items() if k in args.datasets}
        if args.datasets
        else DATASETS
    )

    for name, output_name in selected.items():
        plot_branch_loading(name, output_name, args.data_dir, args.output_dir)

    print("\nDone!")
