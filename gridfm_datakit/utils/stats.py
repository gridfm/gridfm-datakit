import pandas as pd
import plotly.express as px
from pandapower.auxiliary import pandapowerNet
import torch
from torch_geometric.data import InMemoryDataset, Data
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
from gridfm_datakit.utils.power_calculations import (
    get_flows,
    get_injections,
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
def get_edge_attr(matrix, edge_list):
    """
    Get the edge attributes from the adjacency matrix and edge list.
    """
    i = edge_list[0]
    j = edge_list[1]
    edge_attr = matrix[i, j]
    return edge_attr


def get_adj_matrix(edge_attr, edge_list, n):
    """
    Get the adjacency matrix from the edge attributes and edge list.
    """
    # TODO: optimize this
    A = np.zeros((n, n))
    for idx, (i, j) in enumerate(edge_list.T):
        A[i, j] = edge_attr[idx]
    return A


class PowerFlowDataset(InMemoryDataset):
    def __init__(
        self,
        node_file: str,
        edge_file: str,
    ):
        """
        Initialize the PowerFlowDataset.

        Args:
            node_file: Path to the node data CSV file
            edge_file: Path to the edge data CSV file
        """
        super().__init__()
        self.node_data = pd.read_csv(node_file)
        self.edge_data = pd.read_csv(edge_file)

        # Group data by scenario
        self.scenarios = self.node_data["scenario"].unique()

        # Normalize power values
        self.node_data["Pd"] = self.node_data["Pd"] / Sbase
        self.node_data["Qd"] = self.node_data["Qd"] / Sbase
        self.node_data["Pg"] = self.node_data["Pg"] / Sbase
        self.node_data["Qg"] = self.node_data["Qg"] / Sbase

        # Compute net power injections
        self.node_data["P_net"] = self.node_data["Pg"] - self.node_data["Pd"]
        self.node_data["Q_net"] = self.node_data["Qg"] - self.node_data["Qd"]

        # Convert voltage angles to sin and cos
        self.node_data["Va_sin"] = np.sin(self.node_data["Va"] * np.pi / 180)
        self.node_data["Va_cos"] = np.cos(self.node_data["Va"] * np.pi / 180)

        # Process all data at once
        # Add node type column
        self.node_data["bus_type"] = np.argmax(
            self.node_data[["PQ", "PV", "REF"]],
            axis=1,
        )

        # Select features
        feature_cols = ["P_net", "Q_net", "Vm", "Va_sin", "Va_cos", "PQ", "PV", "REF"]

        # Group both nodes and edges by scenario
        node_groups = self.node_data.groupby("scenario")
        edge_groups = self.edge_data.groupby("scenario")

        # Process all scenarios without explicit loops
        self.data_list = []
        for scenario, group in tqdm(node_groups):
            # Get node features
            x = group[feature_cols].values

            # Get edge information
            edges = edge_groups.get_group(scenario)
            edge_index = edges[["index1", "index2"]].to_numpy().T
            G = edges["G"].values
            B = edges["B"].values
            # build adjacency matrix G based on adjacency list edge_index
            G_mat = get_adj_matrix(G, edge_index, len(x))
            B_mat = get_adj_matrix(B, edge_index, len(x))

            # check sum is the same
            assert np.allclose(np.sum(G_mat.flatten()), np.sum(G))
            assert np.allclose(np.sum(B_mat.flatten()), np.sum(B))

            # Get voltage states
            vm = x[:, IDX_VM]
            va_sin = x[:, IDX_VA_SIN]
            va_cos = x[:, IDX_VA_COS]
            va = np.arctan2(va_sin, va_cos)

            # Pre-compute power flows
            P_flow, Q_flow = get_flows(vm, va, G_mat, B_mat, debug=False)
            P_inj, Q_inj = get_injections(vm, va, G_mat, B_mat, debug=False)

            # convert flows to edge_attr format
            P_flow_edge_attr = get_edge_attr(P_flow, edge_index)
            Q_flow_edge_attr = get_edge_attr(Q_flow, edge_index)
            flows = np.stack((P_flow_edge_attr, Q_flow_edge_attr), axis=1)

            # admittance matrix
            admittance_matrix = np.stack((G, B), axis=1)

            # standardize admittance matrix
            admittance_matrix = (
                admittance_matrix - np.mean(admittance_matrix, axis=0)
            ) / np.std(admittance_matrix, axis=0)

            # check if P and Q are close to x[:, IDX_P_NET] and x[:, IDX_Q_NET]
            assert np.allclose(P_inj, x[:, IDX_P_NET])
            assert np.allclose(Q_inj, x[:, IDX_Q_NET])

            # replace x[:, IDX_P_NET] and x[:, IDX_Q_NET] with P_inj and Q_inj for consistency
            x[:, IDX_P_NET] = P_inj
            x[:, IDX_Q_NET] = Q_inj

            # convert to tensors
            x = torch.tensor(x, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            G = torch.tensor(G, dtype=torch.float)
            B = torch.tensor(B, dtype=torch.float)
            admittance_matrix = torch.tensor(admittance_matrix, dtype=torch.float)
            flows = torch.tensor(flows, dtype=torch.float)

            # Create data object with pre-computed measurements
            data = Data(
                x=x,
                edge_index=edge_index,
                G=G,
                B=B,
                bus_type=torch.tensor(group["bus_type"].values, dtype=torch.long),
                flows=flows,
                admittance_matrix=admittance_matrix,
                scenario=scenario,
            )

            # Add noisy measurements
            # if self.add_noise_state_estimation:
            #    data = add_noisy_measurements(
            #        data,
            #        self.noise_std,
            #        self.percentage_branch_meas,
            #        self.percentage_buses_meas,
            #    )

            self.data_list.append(data)

        # Convert to tensors for faster processing
        self.data, self.slices = self.collate(self.data_list)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def get_feature_data(dataset, bus_idx: int, feature_idx: int) -> np.ndarray:
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
    for sample in dataset:
        data.append(sample.x[bus_idx, feature_idx].item())
    return np.array(data)


def get_va_deg(dataset, bus_idx: int) -> np.ndarray:
    """
    Calculate voltage angle in degrees for a specific bus.

    Args:
        dataset: The dataset containing power system data
        bus_idx: Index of the bus to calculate voltage angle for

    Returns:
        numpy.ndarray: Array of voltage angles in degrees
    """
    va_sin = get_feature_data(dataset, bus_idx, IDX_VA_SIN)
    va_cos = get_feature_data(dataset, bus_idx, IDX_VA_COS)
    return np.degrees(np.arctan2(va_sin, va_cos))


def plot_bus_level_features(dataset, bus_idx: int, output_dir: str) -> None:
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
            feature_data = get_va_deg(dataset, bus_idx)
        else:
            feature_data = get_feature_data(dataset, bus_idx, feature_idx)

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


def plot_bus_features(dataset, bus_idx: int, output_dir: str) -> None:
    """
    Create and save all plots for a specific bus.

    Args:
        dataset: The dataset containing power system data
        bus_idx: Index of the bus to plot features for
        output_dir: Directory to save the plots
    """
    plot_bus_level_features(dataset, bus_idx, output_dir)


def plot_feature_distributions(dataset, output_dir: str) -> None:
    """
    Create and save violin plots showing the distribution of each feature across all buses.

    Args:
        dataset: The dataset containing power system data
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

    n_buses = dataset[0].x.shape[0]

    for feature_name, feature_idx in features:
        fig, ax = plt.subplots(figsize=(15, 6))

        # Collect data for all buses
        all_bus_data = []
        for bus in range(n_buses):
            if feature_idx == -1:  # Special case for voltage angle in degrees
                bus_data = get_va_deg(dataset, bus)
            else:
                bus_data = get_feature_data(dataset, bus, feature_idx)
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

        # # Add statistics as text
        # stats_text = []
        # for i, data in enumerate(all_bus_data):
        #     mean = np.mean(data)
        #     std = np.std(data)
        #     stats_text.append(f"Bus {i}:\nMean: {mean:.3f}\nStd: {std:.3f}")

        # Add statistics text box
        # props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        # ax.text(
        #     0.02,
        #     0.98,
        #     "\n".join(stats_text[:5]),
        #     transform=ax.transAxes,
        #     verticalalignment="top",
        #     bbox=props,
        #     fontsize=8,
        # )

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"distribution_{feature_name}_all_buses.png"),
        )
        plt.close()
