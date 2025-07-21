import pandas as pd
import plotly.express as px
from pandapower.auxiliary import pandapowerNet
from gridfm_datakit.process.solvers import calculate_power_imbalance
from typing import List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from gridfm_datakit.utils.config import (
    IDX_P_NET,
    IDX_Q_NET,
    IDX_VM,
    IDX_VA_SIN,
    IDX_VA_COS,
    IDX_PQ,
    IDX_PV,
    IDX_REF,
    Sbase,
)


def plot_stats(base_path: str) -> None:
    """Generates and saves HTML plots of network statistics.

    Creates histograms for various network statistics including number of generators,
    lines, transformers, overloads, and maximum loading. Saves the plots to an HTML file.

    Args:
        base_path: Directory path where the stats CSV file is located and where
            the HTML plot will be saved.

    Raises:
        FileNotFoundError: If stats.csv is not found in the base_path directory.
    """
    stats_to_plot = Stats()
    stats_to_plot.load(base_path)
    filename = base_path + "/stats_plot.html"

    with open(filename, "w") as f:
        # Plot for n_generators
        fig_generators = px.histogram(stats_to_plot.n_generators)
        fig_generators.update_layout(xaxis_title="Number of Generators")
        f.write(fig_generators.to_html(full_html=False, include_plotlyjs="cdn"))

        # Plot for n_lines
        fig_lines = px.histogram(stats_to_plot.n_lines)
        fig_lines.update_layout(xaxis_title="Number of Lines")
        f.write(fig_lines.to_html(full_html=False, include_plotlyjs=False))

        # Plot for n_trafos
        fig_trafos = px.histogram(stats_to_plot.n_trafos)
        fig_trafos.update_layout(xaxis_title="Number of Transformers")
        f.write(fig_trafos.to_html(full_html=False, include_plotlyjs=False))

        # Plot for n_overloads
        fig_overloads = px.histogram(stats_to_plot.n_overloads)
        fig_overloads.update_layout(xaxis_title="Number of Overloads")
        f.write(fig_overloads.to_html(full_html=False, include_plotlyjs=False))

        # Plot for max_loading
        fig_max_loading = px.histogram(stats_to_plot.max_loading)
        fig_max_loading.update_layout(xaxis_title="Max Loading")
        f.write(fig_max_loading.to_html(full_html=False, include_plotlyjs=False))

        # Plot for total_p_diff
        fig_total_p_diff = px.histogram(stats_to_plot.total_p_diff)
        fig_total_p_diff.update_layout(xaxis_title="Total Active Power Imbalance")
        f.write(fig_total_p_diff.to_html(full_html=False, include_plotlyjs=False))

        # Plot for total_q_diff
        fig_total_q_diff = px.histogram(stats_to_plot.total_q_diff)
        fig_total_q_diff.update_layout(xaxis_title="Total Reactive Power Imbalance")
        f.write(fig_total_q_diff.to_html(full_html=False, include_plotlyjs=False))


class Stats:  # network stats
    """A class to track and analyze statistics related to power grid networks.

    This class maintains data lists of various network metrics including
    number of lines, transformers, generators, overloads, maximum loading, total active power imbalance, and total reactive power imbalance.

    Attributes:
        n_lines: List of number of in-service lines over time.
        n_trafos: List of number of in-service transformers over time.
        n_generators: List of total in-service generators (gen + sgen) over time.
        n_overloads: List of number of overloaded elements over time.
        max_loading: List of maximum loading percentages over time.
        total_p_diff: List of total active power imbalance over time.
        total_q_diff: List of total reactive power imbalance over time.
    """

    def __init__(self) -> None:
        """Initializes the Stats object with empty lists for all tracked metrics."""
        self.n_lines = []
        self.n_trafos = []
        self.n_generators = []
        self.n_overloads = []
        self.max_loading = []
        self.total_p_diff = []
        self.total_q_diff = []

    def update(self, net: pandapowerNet) -> None:
        """Adds the current state of the network to the data lists.

        Args:
            net: A pandapower network object containing the current state of the grid.
        """
        self.n_lines.append(net.line.in_service.sum())
        self.n_trafos.append(net.trafo.in_service.sum())
        self.n_generators.append(net.gen.in_service.sum() + net.sgen.in_service.sum())
        self.n_overloads.append(
            np.sum(
                [
                    (net.res_line["loading_percent"] > 100.01).sum(),
                    (net.res_trafo["loading_percent"] > 100.01).sum(),
                ],
            ),
        )

        self.max_loading.append(
            np.max(
                [
                    net.res_line["loading_percent"].max(),
                    net.res_trafo["loading_percent"].max(),
                ],
            ),
        )
        total_p_diff, total_q_diff = calculate_power_imbalance(net)
        self.total_p_diff.append(total_p_diff)
        self.total_q_diff.append(total_q_diff)

    def merge(self, other: "Stats") -> None:
        """Merges another Stats object into this one.

        Args:
            other: Another Stats object whose data will be merged into this one.
        """
        self.n_lines.extend(other.n_lines)
        self.n_trafos.extend(other.n_trafos)
        self.n_generators.extend(other.n_generators)
        self.n_overloads.extend(other.n_overloads)
        self.max_loading.extend(other.max_loading)
        self.total_p_diff.extend(other.total_p_diff)
        self.total_q_diff.extend(other.total_q_diff)

    def save(self, base_path: str) -> None:
        """Saves the tracked statistics to a CSV file.

        If the file already exists, appends the new data with a continuous index.
        If the file doesn't exist, creates a new file.

        Args:
            base_path: Directory path where the CSV file will be saved.
        """
        filename = os.path.join(base_path, "stats.csv")

        new_data = pd.DataFrame(
            {
                "n_lines": self.n_lines,
                "n_trafos": self.n_trafos,
                "n_generators": self.n_generators,
                "n_overloads": self.n_overloads,
                "max_loading": self.max_loading,
                "total_p_diff": self.total_p_diff,
                "total_q_diff": self.total_q_diff,
            },
        )

        if os.path.exists(filename):
            # Read existing file to determine the new index start
            existing_data = pd.read_csv(filename)
            start_index = existing_data.index[-1] + 1 if not existing_data.empty else 0
            new_data.index = range(start_index, start_index + len(new_data))

            new_data.to_csv(filename, mode="a", header=False)
        else:
            new_data.to_csv(filename, index=True)

    def load(self, base_path: str) -> None:
        """Loads the tracked statistics from a CSV file.

        Args:
            base_path: Directory path where the CSV file is saved.

        Raises:
            FileNotFoundError: If stats.csv is not found in the base_path directory.
        """
        filename = os.path.join(base_path, "stats.csv")
        df = pd.read_csv(filename)
        self.n_lines = df["n_lines"].values
        self.n_trafos = df["n_trafos"].values
        self.n_generators = df["n_generators"].values
        self.n_overloads = df["n_overloads"].values
        self.max_loading = df["max_loading"].values
        self.total_p_diff = df["total_p_diff"].values
        self.total_q_diff = df["total_q_diff"].values


# Plotting functions for bus-level features and distributions
def get_feature_data(data_list, bus_idx: int, feature_idx: int) -> np.ndarray:
    """
    Extract feature data for a specific bus from the dataset.

    Args:
        dataset: The dataset containing power system data
        bus_idx: Index of the bus to extract data for
        feature_idx: Index of the feature to extract

    Returns:
        numpy.ndarray: Array of feature values for the specified bus
    """
    data = []
    for sample in data_list:
        data.append(sample[bus_idx, feature_idx].item())
    return np.array(data)


def get_va_deg(data_list, bus_idx: int) -> np.ndarray:
    """
    Calculate voltage angle in degrees for a specific bus.

    Args:
        dataset: The dataset containing power system data
        bus_idx: Index of the bus to calculate voltage angle for

    Returns:
        numpy.ndarray: Array of voltage angles in degrees
    """
    va_sin = get_feature_data(data_list, bus_idx, IDX_VA_SIN)
    va_cos = get_feature_data(data_list, bus_idx, IDX_VA_COS)
    return np.degrees(np.arctan2(va_sin, va_cos))


def plot_bus_level_features(data_list, bus_idx: int, output_dir: str) -> None:
    """
    Create and save histograms, line plots, and violin plots for all features of a specific bus.

    Args:
        dataset: The dataset containing power system data
        bus_idx: Index of the bus to plot features for
        output_dir: Directory to save the plots
    """
    # Feature names and indices
    features: List[Tuple[str, int]] = [
        ("P_net", IDX_P_NET),
        ("Q_net", IDX_Q_NET),
        ("Vm", IDX_VM),
        ("Va_sin", IDX_VA_SIN),
        ("Va_cos", IDX_VA_COS),
        ("Va_deg", -1),  # Special case for voltage angle in degrees
        ("PQ", IDX_PQ),
        ("PV", IDX_PV),
        ("REF", IDX_REF),
    ]

    # Create a figure with subplots for each feature (3 rows per feature: histogram, line plot, and violin plot)
    fig, axes = plt.subplots(9, 3, figsize=(20, 45))
    fig.suptitle(
        f"Feature Distributions and Time Series for Bus {bus_idx}",
        fontsize=16,
    )

    # Plot histograms, line plots, and violin plots for each feature
    for row, (feature_name, feature_idx) in enumerate(features):
        # Get feature data
        if feature_idx == -1:  # Special case for voltage angle in degrees
            feature_data = get_va_deg(data_list, bus_idx)
        else:
            feature_data = get_feature_data(data_list, bus_idx, feature_idx)

        # Plot histogram
        ax_hist = axes[row, 0]
        ax_hist.hist(feature_data, bins=50, alpha=0.7)
        ax_hist.set_title(f"{feature_name} - Distribution")
        ax_hist.set_xlabel("Value")
        ax_hist.set_ylabel("Frequency")

        # Add statistics to histogram
        mean = np.mean(feature_data)
        std = np.std(feature_data)
        ax_hist.text(
            0.05,
            0.95,
            f"Mean: {mean:.4f}\nStd: {std:.4f}",
            transform=ax_hist.transAxes,
            verticalalignment="top",
        )

        # Plot line plot
        ax_line = axes[row, 1]
        ax_line.plot(feature_data, alpha=0.7)
        ax_line.set_title(f"{feature_name} - Time Series")
        ax_line.set_xlabel("Sample Index")
        ax_line.set_ylabel("Value")

        # Add statistics to line plot
        ax_line.text(
            0.05,
            0.95,
            f"Mean: {mean:.4f}\nStd: {std:.4f}",
            transform=ax_line.transAxes,
            verticalalignment="top",
        )

        # Add grid to line plot for better readability
        ax_line.grid(True, alpha=0.3)

        # Plot violin plot
        ax_violin = axes[row, 2]
        parts = ax_violin.violinplot(feature_data, vert=True, showmeans=True)

        # Customize violin plot appearance
        for pc in parts["bodies"]:
            pc.set_facecolor("#D43F3A")
            pc.set_alpha(0.7)

        # Add individual data points
        x = np.random.normal(1, 0.04, size=len(feature_data))
        ax_violin.scatter(x, feature_data, alpha=0.1, s=10, color="black")

        # Add box plot on top
        ax_violin.boxplot(
            feature_data,
            vert=True,
            widths=0.15,
            showfliers=False,
            showbox=True,
            showcaps=True,
            showmeans=False,
            medianprops=dict(color="black", linewidth=1.5),
        )

        ax_violin.set_title(f"{feature_name} - Distribution")
        ax_violin.set_ylabel("Value")
        ax_violin.set_xticks([1])
        ax_violin.set_xticklabels([f"Bus {bus_idx}"])

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"feature_plots_bus_{bus_idx}.png"))
    plt.close()


def plot_bus_features(data_list, bus_idx: int, output_dir: str) -> None:
    """
    Create and save all plots for a specific bus.

    Args:
        dataset: The dataset containing power system data
        bus_idx: Index of the bus to plot features for
        output_dir: Directory to save the plots
    """
    plot_bus_level_features(data_list, bus_idx, output_dir)


def plot_feature_distributions(node_file, output_dir: str) -> None:
    """
    Create and save violin plots showing the distribution of each feature across all buses.

    Args:
        dataset: The dataset containing power system data
        output_dir: Directory to save the plots
    """
    node_data = pd.read_csv(node_file)

    # Normalize power values
    node_data["Pd"] = node_data["Pd"] / Sbase
    node_data["Qd"] = node_data["Qd"] / Sbase
    node_data["Pg"] = node_data["Pg"] / Sbase
    node_data["Qg"] = node_data["Qg"] / Sbase

    # Compute net power injections
    node_data["P_net"] = node_data["Pg"] - node_data["Pd"]
    node_data["Q_net"] = node_data["Qg"] - node_data["Qd"]

    # Convert voltage angles to sin and cos
    node_data["Va_sin"] = np.sin(node_data["Va"] * np.pi / 180)
    node_data["Va_cos"] = np.cos(node_data["Va"] * np.pi / 180)

    # Process all data at once
    # Add node type column
    node_data["bus_type"] = np.argmax(
        node_data[["PQ", "PV", "REF"]],
        axis=1,
    )
    # Select features
    feature_cols = ["P_net", "Q_net", "Vm", "Va_sin", "Va_cos", "PQ", "PV", "REF"]

    # Group both nodes and edges by scenario
    node_groups = node_data.groupby("scenario")

    data_list = []

    for scenario, group in tqdm(node_groups):
        # Get node features
        x = group[feature_cols].values
        data_list.append(x)
    # data_df = pd.DataFrame(data=data_list, columns=["x"])

    # Feature names and indices
    features: List[Tuple[str, int]] = [
        ("P_net", IDX_P_NET),
        ("Q_net", IDX_Q_NET),
        ("Vm", IDX_VM),
        ("Va_sin", IDX_VA_SIN),
        ("Va_cos", IDX_VA_COS),
        ("Va_deg", -1),  # Special case for voltage angle in degrees
        ("PQ", IDX_PQ),
        ("PV", IDX_PV),
        ("REF", IDX_REF),
    ]

    n_buses = len(node_data["bus"].unique())

    for feature_name, feature_idx in features:
        fig, ax = plt.subplots(figsize=(15, 6))

        # Collect data for all buses
        all_bus_data = []
        for bus in range(n_buses):
            if feature_idx == -1:  # Special case for voltage angle in degrees
                bus_data = get_va_deg(data_list, bus)
            else:
                bus_data = get_feature_data(data_list, bus, feature_idx)
            all_bus_data.append(bus_data)

        # Create violin plot
        parts = ax.violinplot(all_bus_data, orientation="vertical", showmeans=True)

        # Customize violin plot appearance
        for pc in parts["bodies"]:
            pc.set_facecolor("#D43F3A")
            pc.set_alpha(0.7)

        # Add individual data points
        for i, data in enumerate(all_bus_data):
            # Add jittered scatter points
            x = np.random.normal(i + 1, 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.1, s=10, color="black")

        # Add box plot on top of violin plot
        ax.boxplot(
            all_bus_data,
            orientation="vertical",
            widths=0.15,
            showfliers=False,
            showbox=True,
            showcaps=True,
            showmeans=False,
            medianprops=dict(color="black", linewidth=1.5),
        )

        ax.set_title(f"{feature_name} - Distribution Across All Buses")
        ax.set_xlabel("Bus Index")
        ax.set_ylabel("Value")
        ax.set_xticks(range(1, n_buses + 1))
        ax.set_xticklabels([f"Bus {i}" for i in range(n_buses)])

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"distribution_{feature_name}_all_buses.png"),
        )
        plt.close()
