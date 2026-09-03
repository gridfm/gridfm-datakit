"""
Generate bar plots comparing branch flow data (Pf, Qf) for PF datasets.

BOUNDS STRATEGY:
|- Pf, Qf (per-branch): Data bounds (min/max across all datasets for each branch)

ENTROPY COMPUTATION:
|- All features are branch-level: Computed per branch, then averaged across all branches
|- Shannon entropy with 100 fixed bins, normalized by log2(100)
|- Std metric: Direct standard deviation of sampled values
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from plot_utils import entropy_from_samples_fixed, IBM_COLORS


def load_branch_datasets(dataset_names=None, data_dir=None):
    """Load branch data for selected datasets."""
    data_path = Path(data_dir) if data_dir else Path(__file__).parent / "dataset_sampled"
    
    # Only datasets that have branch data
    available_datasets = {
        "gridfm_datakit_pf": "GridFM-DataKit (PF)",
        "pfdelta": "PFDelta",
    }
    
    if dataset_names:
        selected = {k: v for k, v in available_datasets.items() if k in dataset_names}
    else:
        selected = available_datasets
    
    branch_data = {}
    description = {}
    
    for name, desc in selected.items():
        branch_file = data_path / f"{name}_branch_data.parquet"
        if branch_file.exists():
            branch_data[name] = pd.read_parquet(branch_file)
            description[name] = desc
            print(f"  Loaded {name}: {len(branch_data[name])} rows, {branch_data[name]['scenario'].nunique()} scenarios")
        else:
            print(f"  Warning: {branch_file} not found, skipping {name}")
    
    return branch_data, description


def compute_bounds(branch_data):
    """Compute bounds for spider plot using data bounds."""
    bounds = {}
    
    # Pf, Qf: use data bounds (per-branch)
    for feature in ["pf", "qf"]:
        per_branch = {}
        for df in branch_data.values():
            grouped = df.groupby("idx")[feature].agg(["min", "max"])
            for branch_idx, row in grouped.iterrows():
                if branch_idx in per_branch:
                    per_branch[branch_idx] = (
                        min(per_branch[branch_idx][0], row["min"]),
                        max(per_branch[branch_idx][1], row["max"]),
                    )
                else:
                    per_branch[branch_idx] = (row["min"], row["max"])
        bounds[feature] = per_branch
    
    return bounds


def plot_spider(selected_versions, branch_data, description, *, metric="entropy", output_dir="comparison_plots"):
    """Generate bar plots for branch flow data."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    features = ["pf", "qf"]
    bounds = compute_bounds(branch_data) if metric == "entropy" else {}
    
    stats = {}
    for label in selected_versions:
        feature_entropies = []
        
        for f in features:
            dfv = branch_data[label]
            branch_groups = dfv.groupby("idx")
            branch_entropies = []
            
            for branch_idx, g in branch_groups:
                x = g[f].values
                
                if metric == "entropy":
                    rng = bounds[f][branch_idx]
                    assert np.all(x >= rng[0]), f"{f} at branch {branch_idx}: values below min bound {rng[0]}"
                    assert np.all(x <= rng[1]), f"{f} at branch {branch_idx}: values above max bound {rng[1]}"
                    if rng[0] == rng[1]:
                        H = 0.0
                    else:
                        H, _ = entropy_from_samples_fixed(x, bins=100, rng=rng)
                else:
                    H = np.std(x)
                branch_entropies.append(H)
            
            feature_entropies.append(np.mean(branch_entropies))
        
        stats[label] = feature_entropies

    df_stats = pd.DataFrame(stats, index=features).T
    if metric == "entropy":
        df_stats = df_stats.div(np.log2(100), axis=1)
    else:
        df_stats = df_stats.div(df_stats.max(axis=0), axis=1)

    # Create bar plot
    feature_labels = ["Pf", "Qf"]  # Display labels (uppercase)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(features))
    width = 0.35  # width of bars
    
    for i, (label, row) in enumerate(df_stats.iterrows()):
        offset = width * (i - (len(selected_versions) - 1) / 2)
        color = IBM_COLORS[i % len(IBM_COLORS)]
        ax.bar(x + offset, row.values, width, label=description[label], color=color, alpha=0.8)

    ax.set_xlabel("Branch Flow Feature", fontsize=12)
    metric_label = "Normalized Entropy" if metric == "entropy" else "Normalized Std Dev"
    ax.set_ylabel(metric_label, fontsize=12)
    # ax.set_title(f"Branch Flow {metric_label} Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_file = output_path / f"barplot_branch_{metric}_pf.pdf"
    print(f"  Saving bar plot to {output_file.name}...")
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {output_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate bar plots for branch flow data.")
    parser.add_argument("--metric", choices=["entropy", "std"], default=None, help="Metric to use.")
    parser.add_argument("--datasets", nargs="*", help="Subset of datasets to compare (gridfm_datakit_pf, pfdelta).")
    parser.add_argument("--output-dir", default="comparison_plots", help="Output directory.")
    parser.add_argument("--data-dir", default=str(Path(__file__).parent / "dataset_sampled"), help="Directory containing sampled datasets.")
    args = parser.parse_args()

    print("Loading branch datasets...")
    branch_data, description = load_branch_datasets(args.datasets, data_dir=args.data_dir)
    selected_versions = list(branch_data.keys())
    
    if len(selected_versions) == 0:
        print("No datasets with branch data found!")
        exit(1)
    
    print(f"\nComparing {len(selected_versions)} datasets: {', '.join(selected_versions)}")
    print("Generating bar plots...")
    
    if args.metric is None:
        plot_spider(selected_versions, branch_data, description, metric='entropy', output_dir=args.output_dir)
        plot_spider(selected_versions, branch_data, description, metric='std', output_dir=args.output_dir)
    else:
        plot_spider(selected_versions, branch_data, description, metric=args.metric, output_dir=args.output_dir)
    
    print("\nDone!")

