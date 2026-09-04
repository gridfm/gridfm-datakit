"""
Generate violin plots comparing datasets in PF or OPF mode.
Uses bus-level data for all features.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from gridfm_datakit.network import load_net_from_pglib
from gridfm_datakit.utils.idx_bus import VMIN, VMAX, BUS_TYPE, PQ, PV, REF
from plot_utils import set_datasets_folder, load_selected_datasets, IBM_COLORS


# Bus selections used for the figures published in the paper.
#
# The upstream code picks these buses with an unseeded np.random.choice, so the
# figures were not reproducible run-to-run. These lists were recovered from the
# figures published with the paper and reproduce all 12 violin panels exactly.
# List order is the x-axis order. See verify_reproducibility.md.
PINNED_BUSES = {
    ("pf", "Pd"): [53, 112, 42, 111, 61, 23, 3, 116, 82, 103],
    ("pf", "Qd"): [114, 74, 10, 82, 105, 101, 108, 93, 58, 0],
    ("pf", "Pg"): [64, 30, 53, 68, 11, 65, 99, 79, 102, 60],
    ("pf", "Qg"): [35, 23, 106, 75, 72, 109, 11, 90, 69, 84],
    ("pf", "Vm"): [95, 3, 99, 9, 40, 113, 90, 36, 65, 111],
    ("pf", "Va"): [62, 20, 97, 29, 106, 36, 108, 93, 27, 32],
    ("opf", "Pd"): [40, 16, 115, 75, 5, 41, 91, 27, 117, 31],
    ("opf", "Qd"): [95, 98, 51, 3, 38, 7, 5, 35, 17, 41],
    ("opf", "Pg"): [45, 11, 53, 68, 60, 86, 25, 24, 58, 9],
    ("opf", "Qg"): [71, 33, 88, 26, 7, 54, 106, 90, 69, 84],
    ("opf", "Vm"): [6, 33, 91, 76, 99, 83, 52, 44, 24, 56],
    ("opf", "Va"): [43, 90, 116, 24, 3, 1, 104, 98, 30, 81],
}


def plot_violins(selected_versions, bus_data, description, n_buses=118, n_selected=10, mode="pf", output_dir="comparison_plots", pin=True):
    """Plot violin plots with bus-level data for all features."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"Saving violin plots to: {output_path.absolute()}")
    
    net = load_net_from_pglib("case118_ieee")
    base_mva = net.baseMVA
    all_buses = np.arange(n_buses)
    
    bus_types = net.buses[:, BUS_TYPE].astype(int)
    bus_type_names = {PQ: "PQ", PV: "PV", REF: "REF"}
    
    def get_bus_type_label(bus_idx):
        return bus_type_names.get(bus_types[bus_idx], "Unknown")
    
    # Pre-filter datasets by bus
    versions_by_bus = {
        lab: {bus: bus_data[lab][bus_data[lab]["bus"] == bus] for bus in all_buses}
        for lab in selected_versions
    }

    vm_min = net.buses[:, VMIN]
    vm_max = net.buses[:, VMAX]

    features = ["Vm", "Va", "Pd", "Qd", "Pg", "Qg"]
    
    for f in features:
        # Find buses with variation (max != mean) for this feature
        all_data = pd.concat([bus_data[lab][["bus", f]] for lab in selected_versions])
        buses_with_variation = all_data.groupby("bus")[f].apply(lambda x: x.max() != x.mean())
        buses_with_variation = buses_with_variation[buses_with_variation].index.tolist()

        # Sample buses with variation for this feature
        rand_buses = np.random.choice(buses_with_variation, size=min(n_selected, len(buses_with_variation)), replace=False)

        # For Qg, ensure buses 69 and 84 are included, remove up to 2 others if needed
        if f == "Qg":
            for must_bus in [69, 84]:
                if must_bus not in rand_buses:
                    # Remove one bus and add must_bus, only if not already present
                    if len(rand_buses) > 0:
                        rand_buses = np.delete(rand_buses, 0)
                    rand_buses = np.append(rand_buses, must_bus)

        if pin and (mode, f) in PINNED_BUSES:
            rand_buses = np.array(PINNED_BUSES[(mode, f)])

        data_for_plot = []
        
        # Bus-level data for all features
        for bus in rand_buses:
            for lab in selected_versions:
                vals = versions_by_bus[lab][bus][f].values
                if f in ["Pg", "Qg", "Pd", "Qd"]:
                    vals = vals / base_mva
                
                for val in vals:
                    data_for_plot.append({
                        "bus": bus,
                        "bus_type": get_bus_type_label(bus),
                        "dataset": description[lab],
                        "value": val,
                        "bus_label": f"{bus} ({get_bus_type_label(bus)})"
                    })
        
        df_plot = pd.DataFrame(data_for_plot)
        fig, ax = plt.subplots(figsize=(14, 6))
        
        sns.violinplot(
            data=df_plot, x="bus_label", y="value", hue="dataset",
            palette=IBM_COLORS[:len(selected_versions)], ax=ax,
            inner="box", scale="width", cut=0, linewidth=0.8, bw=0.3,
            split=False, width=1.0, alpha=0.75,
        )
        
        # Add constraint bounds for bus features
        if f == "Vm":
            for bus in rand_buses:
                ax.axhline(vm_min[bus], color="gray", linestyle=":", linewidth=1, alpha=0.5)
                ax.axhline(vm_max[bus], color="gray", linestyle=":", linewidth=1, alpha=0.5)
        
        y_labels = {
            "Vm": "Voltage Magnitude (p.u.)",
            "Va": "Voltage Angle (degrees)",
            "Pd": "Real Power Demand (p.u.)",
            "Qd": "Reactive Power Demand (p.u.)",
            "Pg": "Real Power Generation (p.u.)",
            "Qg": "Reactive Power Generation (p.u.)",
        }
        
        if f in ["Pg", "Qd", "Pd", "Qg"]:
            y_min = df_plot["value"].min() * 0.9
            y_max = df_plot["value"].max() * 1.1
            ax.set_ylim(y_min, y_max)

        ax.set_xlabel("Bus (Type)", fontsize=20)
        ax.set_ylabel(y_labels[f], fontsize=20)
        ax.set_title(f"{f}", fontsize=20)
        ax.legend( loc="best", fontsize=20)
        
        # set x and y ticks fontsize to 20
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        
        plt.tight_layout()
        
        output_file = output_path / f"{f}_violin_{mode}.pdf"
        print(f"  Saving {f} plot to {output_file.name}...")
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved {output_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate violin plots.")
    parser.add_argument("--mode", choices=["pf", "opf"], default="pf", help="Mode: pf or opf (default: opf).")
    parser.add_argument("--datasets", nargs="*", help="Subset of dataset names to compare.")
    parser.add_argument("--output-dir", default="comparison_plots", help="Directory to save plots.")
    parser.add_argument("--no-pin", action="store_true",
                        help="Re-sample buses randomly instead of using the paper's pinned selection.")
    parser.add_argument("--data-dir", default=None,
                        help="Directory containing the sampled parquet snapshot (default: dataset_sampled/ next to this script).")
    args = parser.parse_args()

    if args.data_dir:
        set_datasets_folder(args.data_dir)

    bus_data, gen_data, description = load_selected_datasets(args.datasets, mode=args.mode)
    selected_versions = list(bus_data.keys())
    
    print(f"\nComparing {len(selected_versions)} datasets ({args.mode.upper()}): {', '.join(selected_versions)}")
    print("Generating violin plots...")
    plot_violins(selected_versions, bus_data, description, mode=args.mode, output_dir=args.output_dir, pin=not args.no_pin)
    
    print("\nDone!")
