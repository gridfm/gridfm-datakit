import pandas as pd
import plotly.express as px
import os
import numpy as np
from pandapower.auxiliary import pandapowerNet
from gridfm_datakit.process.solvers import calculate_power_imbalance


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


class Stats:
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
