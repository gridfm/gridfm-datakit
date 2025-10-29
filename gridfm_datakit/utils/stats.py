"""Statistics computation and visualization for power flow data."""

from __future__ import annotations
import pandas as pd
import numpy as np
import os
from typing import Dict, List
import matplotlib.pyplot as plt


def compute_stats_from_data(data_dir: str) -> Dict[str, np.ndarray]:
    """Compute statistics from parquet data files (vectorized).

    Computes aggregated statistics from generated power flow data. Processes all scenarios
    in the parquet files and returns per-scenario metrics as well as global statistics.

    Args:
        data_dir: Directory containing bus_data.parquet, branch_data.parquet, gen_data.parquet

    Returns:
        Dictionary with the following keys and corresponding numpy arrays:
        - **n_generators**: Array of active generator counts per scenario (int array)
        - **n_branches**: Array of active branch counts per scenario (int array)
        - **n_overloads**: Array of overloaded branch counts per scenario (loading > 1.0, int array)
        - **max_loading**: Array of maximum branch loading per scenario (float array)
        - **branch_loadings**: Vector of all branch loading values across all scenarios (float array)
        - **p_balance_error**: Array of mean absolute active power balance error per scenario,
          normalized by number of buses (float array)
        - **q_balance_error**: Array of mean absolute reactive power balance error per scenario,
          normalized by number of buses (float array)

        Branch loading is computed as max(||S_from||/rate_a, ||S_to||/rate_a) where ||S|| = sqrt(P² + Q²).
        Power balance errors are computed as the mean absolute difference between net injections
        (including shunt contributions) and aggregated branch flows at each bus.
    """
    # --- Load ---
    bus_data = pd.read_parquet(
        os.path.join(data_dir, "bus_data.parquet"),
        engine="fastparquet",
    )
    branch_data = pd.read_parquet(
        os.path.join(data_dir, "branch_data.parquet"),
        engine="fastparquet",
    )
    gen_data = pd.read_parquet(
        os.path.join(data_dir, "gen_data.parquet"),
        engine="fastparquet",
    )

    # The canonical scenario ordering (to match the original function's behavior)
    scenarios = bus_data["scenario"].unique()

    # --- 1) Counts: generators and branches (active by status only) ---
    n_generators_s = (
        gen_data.loc[gen_data["in_service"] == 1]
        .groupby("scenario", sort=False)
        .size()
        .reindex(scenarios, fill_value=0)
    )

    n_branches_s = (
        branch_data.loc[branch_data["br_status"] == 1]
        .groupby("scenario", sort=False)
        .size()
        .reindex(scenarios, fill_value=0)
    )

    # --- 2) Branch loadings & overloads (only active and with finite rating) ---
    active_br = branch_data[
        (branch_data["br_status"] == 1) & (branch_data["rate_a"] > 0)
    ]

    # loading_f = ||S_f||/rate_a, loading_t = ||S_t||/rate_a; loading = max(loading_f, loading_t)
    # Avoid division-by-zero because we've already filtered rate_a > 0
    s_f = np.sqrt(active_br["pf"].to_numpy() ** 2 + active_br["qf"].to_numpy() ** 2)
    s_t = np.sqrt(active_br["pt"].to_numpy() ** 2 + active_br["qt"].to_numpy() ** 2)
    rate = active_br["rate_a"].to_numpy()
    loading_f = s_f / rate
    loading_t = s_t / rate
    loading = np.maximum(loading_f, loading_t)

    # Attach loading to frame for groupby aggregations
    active_br = active_br.assign(_loading=loading)

    n_overloads_s = (
        (active_br["_loading"] > 1.0)
        .groupby(active_br["scenario"], sort=False)
        .sum()
        .reindex(scenarios, fill_value=0)
    )

    max_loading_s = (
        active_br.groupby("scenario", sort=False)["_loading"]
        .max()
        .reindex(scenarios, fill_value=0.0)
    )

    # Global vector of per-branch loadings (matches prior behavior of extending a list)
    branch_loadings_vec = active_br["_loading"].to_numpy(copy=False)

    # --- 3) Power balance errors (per bus, then mean over buses) ---
    # Per-bus net injections (including shunts)
    bus = bus_data.copy()

    # p_shunt = -GS * 100 * Vm^2 ; q_shunt =  BS * 100 * Vm^2
    vm2 = (bus["Vm"].to_numpy()) ** 2
    p_shunt = -bus["GS"].to_numpy() * 100.0 * vm2
    q_shunt = bus["BS"].to_numpy() * 100.0 * vm2

    net_p = bus["Pg"].to_numpy() - bus["Pd"].to_numpy() + p_shunt
    net_q = bus["Qg"].to_numpy() - bus["Qd"].to_numpy() + q_shunt

    bus["__net_p"] = net_p
    bus["__net_q"] = net_q

    # Aggregate branch flows to each bus (consider only active_br)
    # From-side contributions
    from_sum = (
        active_br.groupby(["scenario", "from_bus"], sort=False)[["pf", "qf"]]
        .sum()
        .rename(columns={"pf": "__p_from", "qf": "__q_from"})
    )
    # To-side contributions
    to_sum = (
        active_br.groupby(["scenario", "to_bus"], sort=False)[["pt", "qt"]]
        .sum()
        .rename(columns={"pt": "__p_to", "qt": "__q_to"})
    )

    # Prepare a bus-level key for merges
    key = ["scenario", "bus"]
    bus_keyed = bus[key + ["__net_p", "__net_q"]].copy()

    # Merge in flow sums; missing => 0
    # Left join on (scenario,bus) == (scenario, from_bus) and similarly for to_bus
    bus_flows = (
        bus_keyed.merge(from_sum, left_on=key, right_index=True, how="left")
        .merge(to_sum, left_on=key, right_index=True, how="left")
        .fillna({"__p_from": 0.0, "__q_from": 0.0, "__p_to": 0.0, "__q_to": 0.0})
    )

    total_p_flow = bus_flows["__p_from"].to_numpy() + bus_flows["__p_to"].to_numpy()
    total_q_flow = bus_flows["__q_from"].to_numpy() + bus_flows["__q_to"].to_numpy()

    p_abs_err = np.abs(bus_flows["__net_p"].to_numpy() - total_p_flow)
    q_abs_err = np.abs(bus_flows["__net_q"].to_numpy() - total_q_flow)

    bus_flows["__p_abs_err"] = p_abs_err
    bus_flows["__q_abs_err"] = q_abs_err

    # Mean over buses per scenario (equivalent to sum / n_buses)
    p_balance_s = (
        bus_flows.groupby("scenario", sort=False)["__p_abs_err"]
        .mean()
        .reindex(scenarios, fill_value=0.0)
    )
    q_balance_s = (
        bus_flows.groupby("scenario", sort=False)["__q_abs_err"]
        .mean()
        .reindex(scenarios, fill_value=0.0)
    )

    # --- Pack results (preserve original array shapes/order) ---
    return {
        "n_generators": n_generators_s.to_numpy(dtype=int),
        "n_branches": n_branches_s.to_numpy(dtype=int),
        "n_overloads": n_overloads_s.to_numpy(dtype=int),
        "max_loading": max_loading_s.to_numpy(dtype=float),
        "branch_loadings": branch_loadings_vec.astype(float, copy=False),
        "p_balance_error": p_balance_s.to_numpy(dtype=float),
        "q_balance_error": q_balance_s.to_numpy(dtype=float),
    }


def plot_stats(data_dir: str) -> None:
    """Generate and save statistics plots using matplotlib.

    Creates a multi-panel histogram plot showing distributions of key metrics across all scenarios.
    The plot is saved as `stats_plot.png` in the specified directory with 300 DPI resolution.

    Args:
        data_dir: Directory containing data files (bus_data.parquet, branch_data.parquet, gen_data.parquet)
                  and where the plot will be saved

    The generated plot contains histograms (with log scale on y-axis) for:
    - Number of generators per scenario
    - Number of branches per scenario
    - Number of overloads per scenario
    - Maximum loading per scenario
    - Branch loading (all branches across all scenarios)
    - Active power balance error (mean absolute error per scenario, normalized)
    - Reactive power balance error (mean absolute error per scenario, normalized)
    """
    stats = compute_stats_from_data(data_dir)
    filename = os.path.join(data_dir, "stats_plot.png")

    # Define figure and subplots
    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    axes = axes.ravel()

    # Titles and data pairs
    plots = [
        ("Number of Generators", stats["n_generators"]),
        ("Number of Branches", stats["n_branches"]),
        ("Number of Overloads", stats["n_overloads"]),
        ("Max Loading", stats["max_loading"]),
        ("Branch Loading", stats["branch_loadings"]),
        ("Active Power Balance Error (normalized)", stats["p_balance_error"]),
        ("Reactive Power Balance Error (normalized)", stats["q_balance_error"]),
    ]

    # Plot histograms
    for ax, (title, data) in zip(axes, plots):
        ax.hist(data, bins=100, color="steelblue", edgecolor="black", alpha=0.7)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel(title, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.4)

    # Remove any unused subplot (if any)
    for i in range(len(plots), len(axes)):
        fig.delaxes(axes[i])

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Statistics plots saved to {filename}")


def plot_feature_distributions(
    node_file: str,
    output_dir: str,
    sn_mva: float,
    dcpf: bool,
    buses: List[int] = None,
) -> None:
    """Create and save violin plots showing the distribution of each feature across all buses.

    Generates violin plots for each feature column defined in `BUS_COLUMNS` (and `DC_BUS_COLUMNS` if
    `dcpf=True`). Each plot shows the probability distribution of feature values across selected buses,
    with overlaid box plots showing quartiles.

    Args:
        node_file: Parquet file containing node data with a 'bus' column (typically bus_data.parquet).
        output_dir: Directory where plots will be saved as `distribution_{feature_name}.png`.
        sn_mva: Base MVA used to normalize power-related columns (Pd, Qd, Pg, Qg) by dividing by this value.
        dcpf: If True, includes DC_BUS_COLUMNS features (e.g., Va_dc) in addition to BUS_COLUMNS.
        buses: List of bus indices to plot. If None, randomly samples 30 buses (or all buses if fewer than 30).

    Each generated plot displays:
    - Violin plots showing the probability density of feature values per bus
    - Box plots overlaid on violins showing quartiles, median, and min/max
    - Power-related features (Pd, Qd, Pg, Qg) are normalized by dividing by `sn_mva`
    - Features are plotted for columns defined in `gridfm_datakit.utils.column_names.BUS_COLUMNS`
      and optionally `DC_BUS_COLUMNS` if `dcpf=True`
    """
    import matplotlib.pyplot as plt
    from gridfm_datakit.utils.column_names import BUS_COLUMNS, DC_BUS_COLUMNS

    node_data = pd.read_parquet(node_file, engine="fastparquet")
    os.makedirs(output_dir, exist_ok=True)

    if not buses:
        # sample 30 buses randomly
        buses = np.random.choice(
            node_data["bus"].unique(),
            size=min(30, len(node_data["bus"].unique())),
            replace=False,
        )

    node_data = node_data[node_data["bus"].isin(buses)]

    # normalize by sn_mva
    for col in ["Pd", "Qd", "Pg", "Qg"]:
        node_data[col] = node_data[col] / sn_mva

    # Group data by bus
    bus_groups = node_data.groupby("bus")
    sorted_buses = sorted(bus_groups.groups.keys())

    if dcpf:
        feature_cols = BUS_COLUMNS + DC_BUS_COLUMNS
    else:
        feature_cols = BUS_COLUMNS

    assert node_data.shape[1] == len(feature_cols) + 1, (
        "Node data has the wrong number of columns"
    )

    for feature_name in feature_cols:
        fig, ax = plt.subplots(figsize=(15, 6))

        bus_data = [
            bus_groups.get_group(bus)[feature_name].values for bus in sorted_buses
        ]

        parts = ax.violinplot(bus_data, showmeans=True)

        for pc in parts["bodies"]:
            pc.set_facecolor("#D43F3A")
            pc.set_alpha(0.7)

        ax.boxplot(
            bus_data,
            widths=0.15,
            showfliers=False,
            showcaps=True,
            medianprops=dict(color="black", linewidth=1.5),
        )

        ax.set_title(f"{feature_name} Distribution Across Buses")
        ax.set_xlabel("Bus Index")
        ax.set_ylabel(feature_name)
        ax.set_xticks(range(1, len(sorted_buses) + 1))
        ax.set_xticklabels(
            [f"Bus {bus}" for bus in sorted_buses],
            rotation=45,
            ha="right",
        )

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(
            output_dir,
            f"distribution_{feature_name}.png",
        )
        plt.savefig(out_path)
        plt.close()
