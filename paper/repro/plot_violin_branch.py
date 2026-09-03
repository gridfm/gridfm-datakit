"""
Generate violin plots comparing branch flow data (Pf, Qf) for PF datasets.
Uses branch-level data for pfdelta and gridfm_datakit_pf.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from plot_utils import IBM_COLORS


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


def plot_branch_violins(selected_versions, branch_data, description, n_selected=10, output_dir="comparison_plots"):
    """Plot violin plots with branch-level data for Pf and Qf."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"Saving branch violin plots to: {output_path.absolute()}")
    
    # Base MVA for normalization (case118_ieee)
    base_mva = 100.0
    
    # Get all unique branches from the datasets (using 'idx' column)
    all_branches = set()
    for lab in selected_versions:
        all_branches.update(branch_data[lab]["idx"].unique())
    
    all_branches = sorted(list(all_branches))
    
    # Pre-filter datasets by branch
    versions_by_branch = {
        lab: {br: branch_data[lab][branch_data[lab]["idx"] == br] for br in all_branches}
        for lab in selected_versions
    }
    
    features = ["pf", "qf"]
    
    for f in features:
        # Find branches with variation (max != mean) for this feature
        all_data = pd.concat([branch_data[lab][["idx", f]] for lab in selected_versions])
        branches_with_variation = all_data.groupby("idx")[f].apply(lambda x: x.max() != x.mean())
        branches_with_variation = branches_with_variation[branches_with_variation].index.tolist()
        
        if len(branches_with_variation) == 0:
            print(f"  Warning: No branches with variation for {f}, skipping...")
            continue
        
        # Sample branches with variation for this feature
        rand_branches = np.random.choice(
            branches_with_variation, 
            size=min(n_selected, len(branches_with_variation)), 
            replace=False
        )
        
        data_for_plot = []
        
        # Branch-level data
        for branch in rand_branches:
            for lab in selected_versions:
                branch_df = versions_by_branch[lab][branch]
                vals = branch_df[f].values
                # Normalize to per-unit
                vals = vals / base_mva
                
                # Get from_bus and to_bus for label (take first row since same for all scenarios)
                from_bus = branch_df["from_bus"].iloc[0]
                to_bus = branch_df["to_bus"].iloc[0]
                
                for val in vals:
                    data_for_plot.append({
                        "branch": branch,
                        "dataset": description[lab],
                        "value": val,
                        "branch_label": f"{int(from_bus)}-{int(to_bus)}"
                    })
        
        df_plot = pd.DataFrame(data_for_plot)
        fig, ax = plt.subplots(figsize=(14, 6))
        
        sns.violinplot(
            data=df_plot, x="branch_label", y="value", hue="dataset",
            palette=IBM_COLORS[:len(selected_versions)], ax=ax,
            inner="box", scale="width", cut=0, linewidth=0.8, bw=0.3,
            split=False, width=1.0, alpha=0.75,
        )
        
        y_labels = {
            "pf": "Real Power Flow (p.u.)",
            "qf": "Reactive Power Flow (p.u.)",
        }
        
        y_min = df_plot["value"].min() * 1.1 if df_plot["value"].min() < 0 else df_plot["value"].min() * 0.9
        y_max = df_plot["value"].max() * 1.1
        ax.set_ylim(y_min, y_max)
        
        ax.set_xlabel("Branch", fontsize=12)
        ax.set_ylabel(y_labels[f], fontsize=12)
        ax.set_title(f"{f.upper()} Distribution Across Datasets (Branch Flows)", fontsize=14)
        ax.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
        
        plt.tight_layout()
        
        output_file = output_path / f"{f.upper()}_branch_violin_pf.pdf"
        print(f"  Saving {f.upper()} plot to {output_file.name}...")
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved {output_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate violin plots for branch flow data.")
    parser.add_argument("--datasets", nargs="*", help="Subset of dataset names to compare (gridfm_datakit_pf, pfdelta).")
    parser.add_argument("--output-dir", default="comparison_plots", help="Directory to save plots.")
    parser.add_argument("--data-dir", default=str(Path(__file__).parent / "dataset_sampled"), help="Directory containing sampled datasets.")
    args = parser.parse_args()
    
    print("Loading branch datasets...")
    branch_data, description = load_branch_datasets(args.datasets, data_dir=args.data_dir)
    selected_versions = list(branch_data.keys())
    
    if len(selected_versions) == 0:
        print("No datasets with branch data found!")
        exit(1)
    
    print(f"\nComparing {len(selected_versions)} datasets: {', '.join(selected_versions)}")
    print("Generating branch flow violin plots...")
    plot_branch_violins(selected_versions, branch_data, description, output_dir=args.output_dir)
    
    print("\nDone!")

