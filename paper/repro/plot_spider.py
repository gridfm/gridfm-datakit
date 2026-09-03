"""
Generate spider plots comparing datasets in PF or OPF mode.

BOUNDS STRATEGY:
|- Vm (per-bus): Network bounds from net.buses[:, VMIN/VMAX]
|- Va: Fixed bounds (-π, π)
|- Pd, Qd, Pg, Qg (per-bus): Data bounds (min/max across all datasets for each bus)

ENTROPY COMPUTATION:
|- All features are bus-level: Computed per bus, then averaged across all buses
|- Shannon entropy with 100 fixed bins, normalized by log2(100)
|- Std metric: Direct standard deviation of sampled values
|- Va (angle): Circular entropy using normalized angle representation
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from gridfm_datakit.network import load_net_from_pglib
from gridfm_datakit.utils.idx_bus import VMIN, VMAX, BUS_TYPE, REF
from plot_utils import set_datasets_folder, load_selected_datasets, entropy_from_samples_fixed, entropy_circular_from_deg_fixed, IBM_COLORS


def compute_bounds(bus_data):
    """Compute bounds for spider plot using network constraints and data."""
    net = load_net_from_pglib("case118_ieee")
    
    bounds = {}
    
    # Va: fixed bounds (-π, π)
    bounds["Va"] = (-np.pi, np.pi)
    
    # Pd, Qd, Pg, Qg: use data bounds (per-bus)
    for feature in ["Pd", "Qd", "Pg", "Qg", "Vm"]:
        per_bus = {}
        for df in bus_data.values():
            grouped = df.groupby("bus")[feature].agg(["min", "max"])
            for bus, row in grouped.iterrows():
                if bus in per_bus:
                    per_bus[bus] = (
                        min(per_bus[bus][0], row["min"]),
                        max(per_bus[bus][1], row["max"]),
                    )
                else:
                    per_bus[bus] = (row["min"], row["max"])
        bounds[feature] = per_bus
    
    return bounds


def plot_spider(selected_versions, bus_data, description, *, metric="entropy", mode="pf", output_dir="comparison_plots"):
    """Generate spider plots."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    features = ["Vm", "Va", "Pd", "Qd", "Pg", "Qg"]
    bounds = compute_bounds(bus_data) if metric == "entropy" else {}
    
    stats = {}
    for label in selected_versions:
        feature_entropies = []
        
        for f in features:
            dfv = bus_data[label]
            bus_groups = dfv.groupby("bus")
            bus_entropies = []
            
            for bus, g in bus_groups:
                x = g[f].values
                
                if metric == "entropy":
                    if f == "Va":
                        H = entropy_circular_from_deg_fixed(x, bins=100)
                    else:
                        rng = bounds[f][bus]
                        assert np.all(x >= rng[0]), f"{f} at bus {bus}: values below min bound {rng[0]}"
                        assert np.all(x <= rng[1]), f"{f} at bus {bus}: values above max bound {rng[1]}"
                        if rng[0] == rng[1]:
                            H = 0.0
                        else:
                            H, _ = entropy_from_samples_fixed(x, bins=100, rng=rng)
                else:
                    if f == "Va":
                        theta = (np.deg2rad(x) + np.pi) % (2 * np.pi) - np.pi
                        H = np.std(theta)
                    else:
                        H = np.std(x)
                bus_entropies.append(H)
            
            feature_entropies.append(np.mean(bus_entropies))
        
        stats[label] = feature_entropies

    df_stats = pd.DataFrame(stats, index=features).T
    if metric == "entropy":
        df_stats = df_stats.div(np.log2(100), axis=1)
    else:
        df_stats = df_stats.div(df_stats.max(axis=0), axis=1)
        

    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    
    for i, (label, row) in enumerate(df_stats.iterrows()):
        values = row.tolist() + [row.iloc[0]]
        color = IBM_COLORS[i % len(IBM_COLORS)]
        ax.plot(angles, values, color=color, linewidth=2, label=description[label])
        ax.fill(angles, values, color=color, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=16)
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.3, 1.2), fontsize=16)
    # set x and y ticks fontsize to 16
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=14)
    plt.tight_layout()
    
    output_file = output_path / f"spider_plot_{metric}_{mode}.pdf"
    print(f"  Saving spider plot to {output_file.name}...")
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {output_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate spider plots.")
    parser.add_argument("--mode", choices=["pf", "opf"], default="pf", help="Mode: pf or opf (default: opf).")
    parser.add_argument("--metric", choices=["entropy", "std"], default=None, help="Metric to use.")
    parser.add_argument("--datasets", nargs="*", help="Subset of datasets to compare.")
    parser.add_argument("--output-dir", default="comparison_plots", help="Output directory.")
    parser.add_argument("--data-dir", default=None,
                        help="Directory containing the sampled parquet snapshot (default: dataset_sampled/ next to this script).")
    args = parser.parse_args()

    if args.data_dir:
        set_datasets_folder(args.data_dir)

    bus_data, gen_data, description = load_selected_datasets(args.datasets, mode=args.mode)
    selected_versions = list(bus_data.keys())
    
    print(f"\nComparing {len(selected_versions)} datasets ({args.mode.upper()}): {', '.join(selected_versions)}")
    print("Generating spider plots...")
    
    if args.metric is None:
        plot_spider(selected_versions, bus_data, description, metric='entropy', mode=args.mode, output_dir=args.output_dir)
        plot_spider(selected_versions, bus_data, description, metric='std', mode=args.mode, output_dir=args.output_dir)
    else:
        plot_spider(selected_versions, bus_data, description, metric=args.metric, mode=args.mode, output_dir=args.output_dir)
    
    print("\nDone!")
