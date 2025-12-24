"""Statistics computation and visualization for power flow data."""

from __future__ import annotations
import pandas as pd
import numpy as np
import os
from typing import Dict, List
import matplotlib.pyplot as plt
from gridfm_datakit.utils.power_balance import (
    compute_branch_powers_vectorized,
    compute_bus_balance,
)
from gridfm_datakit.utils.utils import get_num_scenarios
from gridfm_datakit.utils.utils import read_partitions, n_scenario_per_partition


def compute_stats_from_data(
    data_dir: str,
    sn_mva: float,
    n_partitions: int = 0,
) -> Dict[str, np.ndarray]:
    """Compute statistics from parquet data files (vectorized).

    Computes aggregated statistics from generated power flow data. Processes all scenarios
    in the parquet files and returns per-scenario metrics as well as global statistics.

    Args:
        data_dir: Directory containing bus_data.parquet, branch_data.parquet, gen_data.parquet, and optionally runtime_data.parquet
        sn_mva: Base MVA used to scale power quantities
        n_partitions: Number of partitions to compute stats for (0 for all partitions)

    Returns:
        Dictionary with the following keys and corresponding numpy arrays:
        - **scenario_ids**: Array of scenario IDs
        - **n_generators**: Array of active generator counts per scenario (int array)
        - **n_branches**: Array of active branch counts per scenario (int array)
        - **n_overloads**: Array of overloaded branch counts per scenario (loading > 1.01, int array)
        - **max_loading**: Array of maximum branch loading per scenario (float array)
        - **branch_loadings**: Vector of all branch loading values across all scenarios (float array)
        - **p_balance_error_ac_max**: Array of maximum active power balance error per scenario (float array)
        - **p_balance_error_ac_mean**: Array of mean active power balance error per scenario (float array)
        - **q_balance_error_ac_max**: Array of maximum reactive power balance error per scenario (float array)
        - **q_balance_error_ac_mean**: Array of mean reactive power balance error per scenario (float array)
        - **runtime_data_ac_ms**: (if runtime data available) Array of AC solver runtime per scenario in milliseconds (float array)
        - **p_balance_error_dc_max**: (if DC data available) Array of maximum DC active power balance error per scenario (float array)
        - **p_balance_error_dc_mean**: (if DC data available) Array of mean DC active power balance error per scenario (float array)
        - **bus_idx_max_p_balance_error_dc_per_scenario**: (if DC data available) Array of bus index with max DC PBE per scenario (int array)
        - **runtime_data_dc_ms**: (if DC and runtime data available) Array of DC solver runtime per scenario in milliseconds (float array)

        Branch loading is computed as max(||S_from||/rate_a, ||S_to||/rate_a) where ||S|| = sqrt(P² + Q²).
        Power balance errors are computed as the mean absolute difference between net injections
        (including shunt contributions) and aggregated branch flows at each bus.
    """
    # --- Load from partitioned parquet ---
    # Get total number of scenarios efficiently
    total_scenarios = get_num_scenarios(data_dir)

    total_partitions = (
        total_scenarios + n_scenario_per_partition - 1
    ) // n_scenario_per_partition

    if n_partitions > 0:
        sampled_partitions = sorted(
            np.random.choice(
                total_partitions,
                size=min(n_partitions, total_partitions),
                replace=False,
            ),
        )
        print(
            f"Computing stats for {len(sampled_partitions)} partitions out of {total_partitions}",
        )
    else:
        sampled_partitions = list(range(total_partitions))

    # Read filtered data from partitioned parquet using partition filter
    bus_file = os.path.join(data_dir, "bus_data.parquet")
    branch_file = os.path.join(data_dir, "branch_data.parquet")
    gen_file = os.path.join(data_dir, "gen_data.parquet")
    runtime_file = os.path.join(data_dir, "runtime_data.parquet")

    bus_data = read_partitions(bus_file, sampled_partitions)
    branch_data = read_partitions(branch_file, sampled_partitions)
    gen_data = read_partitions(gen_file, sampled_partitions)

    # Runtime data is optional
    has_runtime = os.path.exists(runtime_file)
    if has_runtime:
        runtime_data = read_partitions(runtime_file, sampled_partitions)
    else:
        print("Runtime data not found, skipping runtime statistics")

    dc = True if "p_mw_dc" in gen_data.columns else False
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
        (active_br["_loading"] > 1.01)
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

    # --- 3) Power balance errors ---
    balance_ac = compute_bus_balance(
        bus_data,
        branch_data,
        branch_data[["pf", "qf", "pt", "qt"]],
        False,
        sn_mva=sn_mva,
    )
    group_by_scenario = balance_ac.groupby("scenario")
    p_balance_ac_max = group_by_scenario["P_mis_ac"].max().reindex(scenarios)
    p_balance_ac_mean = group_by_scenario["P_mis_ac"].mean().reindex(scenarios)
    q_balance_ac_max = group_by_scenario["Q_mis_ac"].max().reindex(scenarios)
    q_balance_ac_mean = group_by_scenario["Q_mis_ac"].mean().reindex(scenarios)
    if dc:
        pf_dc, _, pt_dc, _ = compute_branch_powers_vectorized(
            branch_data,
            bus_data,
            True,
            sn_mva=sn_mva,
        )
        balance_dc = compute_bus_balance(
            bus_data,
            branch_data,
            pd.DataFrame({"pf_dc": pf_dc, "pt_dc": pt_dc}, index=branch_data.index),
            True,
            sn_mva=sn_mva,
        )
        group_by_scenario = balance_dc.groupby("scenario")
        p_balance_dc_max = group_by_scenario["P_mis_dc"].max().reindex(scenarios)
        p_balance_dc_mean = group_by_scenario["P_mis_dc"].mean().reindex(scenarios)
        # bus index of the bus with the largest DC P-mismatch per scenario
        idxmax = group_by_scenario["P_mis_dc"].idxmax().dropna()
        idx_bus_max_p_balance_error_dc_per_scenario = (
            balance_dc.loc[idxmax.dropna(), ["scenario", "bus"]]
            .set_index("scenario")["bus"]
            .reindex(scenarios)
        )

    # ---4) Runtime data (optional) ---
    if has_runtime:
        runtime_data_ac = (
            runtime_data.set_index("scenario")["ac"].reindex(scenarios) * 1000.0
        )
        if dc:
            runtime_data_dc = (
                runtime_data.set_index("scenario")["dc"].reindex(scenarios) * 1000.0
            )

    # --- Pack results (preserve original array shapes/order) ---
    result = {
        "scenario_ids": scenarios,
        "n_generators": n_generators_s.to_numpy(dtype=int),
        "n_branches": n_branches_s.to_numpy(dtype=int),
        "n_overloads": n_overloads_s.to_numpy(dtype=int),
        "max_loading": max_loading_s.to_numpy(dtype=float),
        "branch_loadings": branch_loadings_vec.astype(float, copy=False),
        "p_balance_error_ac_max": p_balance_ac_max.to_numpy(dtype=float),
        "p_balance_error_ac_mean": p_balance_ac_mean.to_numpy(dtype=float),
        "q_balance_error_ac_max": q_balance_ac_max.to_numpy(dtype=float),
        "q_balance_error_ac_mean": q_balance_ac_mean.to_numpy(dtype=float),
    }
    if has_runtime:
        result["runtime_data_ac_ms"] = runtime_data_ac.to_numpy(dtype=float)

    if dc:
        result["p_balance_error_dc_max"] = p_balance_dc_max.to_numpy(dtype=float)
        result["p_balance_error_dc_mean"] = p_balance_dc_mean.to_numpy(dtype=float)
        result["bus_idx_max_p_balance_error_dc_per_scenario"] = (
            idx_bus_max_p_balance_error_dc_per_scenario.to_numpy(dtype=float)
        )
        if has_runtime:
            result["runtime_data_dc_ms"] = runtime_data_dc.to_numpy(dtype=float)

    return result


def plot_stats(data_dir: str, sn_mva: float, n_partitions: int = 0) -> None:
    """Generate and save statistics plots using matplotlib.

    Creates a multi-panel histogram plot showing distributions of key metrics across all scenarios.
    The plot is saved as `stats_plot.png` in the specified directory with 300 DPI resolution.

    Args:
        data_dir: Directory containing data files (bus_data.parquet, branch_data.parquet, gen_data.parquet, and optionally runtime_data.parquet)
                  and where the plot will be saved
        sn_mva: Base MVA used to scale power quantities
        n_partitions: Number of partitions to compute stats for (0 for all partitions)

    The generated plot contains histograms (with log scale on y-axis) for:
    - Number of generators per scenario
    - Number of branches per scenario
    - Number of overloads per scenario
    - Maximum loading per scenario
    - Branch loading (all branches across all scenarios)
    - Active power balance error (mean absolute error per scenario, normalized)
    - Reactive power balance error (mean absolute error per scenario, normalized)
    - (If runtime data available) Runtime (AC solver execution time in milliseconds)
    - (If DC data available) DC active power balance error, bus index with max DC PBE
    - (If DC and runtime data available) DC runtime
    """
    stats = compute_stats_from_data(data_dir, sn_mva=sn_mva, n_partitions=n_partitions)
    filename = os.path.join(data_dir, "stats_plot.png")

    # Save per-scenario statistics to a parquet file with one row per scenario

    per_scenario = {
        "scenario": stats["scenario_ids"],
        "n_generators": stats["n_generators"],
        "n_branches": stats["n_branches"],
        "n_overloads": stats["n_overloads"],
        "max_loading": stats["max_loading"],
        "p_balance_error_ac_max": stats["p_balance_error_ac_max"],
        "p_balance_error_ac_mean": stats["p_balance_error_ac_mean"],
        "q_balance_error_ac_max": stats["q_balance_error_ac_max"],
        "q_balance_error_ac_mean": stats["q_balance_error_ac_mean"],
    }
    if "runtime_data_ac_ms" in stats:
        per_scenario["runtime_data_ac_ms"] = stats["runtime_data_ac_ms"]

    mean_mean_p_balance_error_ac = np.nanmean(stats["p_balance_error_ac_mean"])
    mean_mean_q_balance_error_ac = np.nanmean(stats["q_balance_error_ac_mean"])

    if "p_balance_error_dc_max" in stats:
        per_scenario["p_balance_error_dc_max"] = stats["p_balance_error_dc_max"]
        per_scenario["p_balance_error_dc_mean"] = stats["p_balance_error_dc_mean"]
        per_scenario["bus_idx_max_p_balance_error_dc_per_scenario"] = stats[
            "bus_idx_max_p_balance_error_dc_per_scenario"
        ]
        if "runtime_data_dc_ms" in stats:
            per_scenario["runtime_data_dc_ms"] = stats["runtime_data_dc_ms"]
        mean_mean_p_balance_error_dc = np.nanmean(stats["p_balance_error_dc_mean"])

    df_stats = pd.DataFrame(per_scenario)
    # Add partition column for scenario-based partitioning (n_scenario_per_partition scenarios per partition)
    df_stats["scenario_partition"] = (
        df_stats["scenario"] // n_scenario_per_partition
    ).astype("int64")
    df_stats.to_parquet(
        os.path.join(data_dir, "stats.parquet"),
        partition_cols=["scenario_partition"],
        engine="pyarrow",
        index=False,
    )

    # Titles and data pairs
    plots = [
        ("Number of Generators", stats["n_generators"]),
        ("Number of Branches", stats["n_branches"]),
        ("Number of Overloads", stats["n_overloads"]),
        ("Max Loading", stats["max_loading"]),
        ("Branch Loading", stats["branch_loadings"]),
        # ("Max Active PBE (AC, normalized)", stats["p_balance_error_ac_max"]),
        (
            f"Mean Active PBE (AC, normalized). Mean={np.format_float_scientific(mean_mean_p_balance_error_ac, precision=2)}",
            stats["p_balance_error_ac_mean"],
        ),
        # ("Max Reactive PBE (AC, normalized)", stats["q_balance_error_ac_max"]),
        (
            f"Mean Reactive PBE (AC, normalized). Mean={np.format_float_scientific(mean_mean_q_balance_error_ac, precision=2)}",
            stats["q_balance_error_ac_mean"],
        ),
    ]

    # Add runtime plot if runtime data exists
    if "runtime_data_ac_ms" in stats:
        plots.append(
            (
                "Runtime (AC, ms). Mean={:.2f}".format(
                    stats["runtime_data_ac_ms"].mean(),
                ),
                stats["runtime_data_ac_ms"],
            ),
        )

    # Optionally add DC power balance if available
    if "p_balance_error_dc_max" in stats:
        plots.append(
            (
                "Max Active PBE (DC in AC model, normalized)",
                stats["p_balance_error_dc_max"],
            ),
        )
        plots.append(
            (
                f"Mean Active PBE (DC in AC model, normalized). Mean={mean_mean_p_balance_error_dc:.2f}",
                stats["p_balance_error_dc_mean"],
            ),
        )
        plots.append(
            (
                "Bus Index with Max Active PBE (DC in AC model, normalized)",
                stats["bus_idx_max_p_balance_error_dc_per_scenario"],
            ),
        )
        if "runtime_data_dc_ms" in stats:
            plots.append(
                (
                    "Runtime (DC, ms). Mean={:.2f}".format(
                        np.nanmean(stats["runtime_data_dc_ms"]),
                    ),
                    stats["runtime_data_dc_ms"],
                ),
            )

    # sort plots by title
    plots.sort(key=lambda x: x[0])
    # Define figure and subplots
    n_plots = len(plots)
    import math

    fig, axes = plt.subplots(math.ceil(n_plots / 2), 2, figsize=(12, 14))
    axes = axes.ravel()

    # Plot histograms
    for ax, (title, data) in zip(axes, plots):
        print(title)
        # For DC-related metrics, exclude NaNs from plot but show count in legend
        if "DC" in title:
            valid = data[~np.isnan(data)]
            nan_count = int(np.isnan(data).sum())
            ax.hist(
                valid,
                bins=100,
                color="steelblue",
                edgecolor="black",
                alpha=0.7,
                label=f"valid={len(valid)}, nan={nan_count}",
            )
            ax.legend()
        else:
            if data.size > 0:
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
    buses: List[int] = None,
    n_partitions: int = 0,
) -> None:
    """Create and save violin plots showing the distribution of each feature across all buses.

    Generates violin plots for each feature column defined in `BUS_COLUMNS` (and `DC_BUS_COLUMNS` if
    DC columns are present in the data). Each plot shows the probability distribution of feature values across selected buses,
    with overlaid box plots showing quartiles.

    Args:
        node_file: Parquet file containing node data with a 'bus' column (typically bus_data.parquet).
        output_dir: Directory where plots will be saved as `distribution_{feature_name}.png`.
        sn_mva: Base MVA used to normalize power-related columns (Pd, Qd, Pg, Qg) by dividing by this value.
        buses: List of bus indices to plot. If None, randomly samples 30 buses (or all buses if fewer than 30).
        n_partitions: Number of partitions to plot (0 for all partitions)

    Each generated plot displays:
    - Violin plots showing the probability density of feature values per bus
    - Box plots overlaid on violins showing quartiles, median, and min/max
    - Power-related features (Pd, Qd, Pg, Qg) are normalized by dividing by `sn_mva`
    - Features are plotted for columns defined in `gridfm_datakit.utils.column_names.BUS_COLUMNS`
      and optionally `DC_BUS_COLUMNS` if DC columns (e.g., Va_dc) are present in the data
    """
    import matplotlib.pyplot as plt
    from gridfm_datakit.utils.column_names import DC_BUS_COLUMNS

    # Get total number of scenarios and partitions
    data_dir = os.path.dirname(node_file)
    total_scenarios = get_num_scenarios(data_dir)
    total_partitions = (
        total_scenarios + n_scenario_per_partition - 1
    ) // n_scenario_per_partition

    if n_partitions > 0:
        sampled_partitions = sorted(
            np.random.choice(
                total_partitions,
                size=min(n_partitions, total_partitions),
                replace=False,
            ),
        )
        print(
            f"Plotting for {len(sampled_partitions)} partitions out of {total_partitions}",
        )
    else:
        sampled_partitions = list(range(total_partitions))

    # Read filtered data from partitioned parquet using partition filter
    node_data = read_partitions(node_file, sampled_partitions)
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

    feature_cols = [
        "Pd",
        "Qd",
        "Pg",
        "Qg",
        "Vm",
        "Va",
        "PQ",
        "PV",
        "REF",
    ]
    if "Va_dc" in node_data.columns:
        feature_cols = feature_cols + DC_BUS_COLUMNS
    else:
        feature_cols = feature_cols

    for feature_name in feature_cols:
        fig, ax = plt.subplots(figsize=(15, 6))

        bus_data = [
            bus_groups.get_group(bus)[feature_name].dropna().values
            for bus in sorted_buses
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
