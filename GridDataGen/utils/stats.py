import pandas as pd
import plotly.express as px
import os


def plot_stats(base_path):
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


class Stats:
    """
    A class to track and analyze statistics related to power grid networks.

    Attributes:
        n_lines (list): Tracks the number of lines in the network over time.
        n_trafos (list): Tracks the number of transformers in the network over time.
        n_generators (list): Tracks the number of in-service generators and static generators in the network over time.
    """

    def __init__(self):
        """
        Initializes the Stats object with empty lists for n_lines, n_trafos, and n_generators.
        """
        self.n_lines = []
        self.n_trafos = []
        self.n_generators = []

    def update(self, net):
        """
        Updates the statistics for the current state of the power grid network.

        Args:
            net: A power grid network object with attributes `line`, `trafo`, `gen`, and `sgen`.
        """
        self.n_lines.append(len(net.line))
        self.n_trafos.append(len(net.trafo))
        self.n_generators.append(net.gen.in_service.sum() + net.sgen.in_service.sum())

    def merge(self, other):
        """
        Merges another Stats object into this one.

        Args:
            other (Stats): Another Stats object to merge with this one.
        """
        self.n_lines.extend(other.n_lines)
        self.n_trafos.extend(other.n_trafos)
        self.n_generators.extend(other.n_generators)

    def save(self, base_path):
        """
        Saves the tracked statistics to a CSV file, appending if the file already exists.
        Ensures the index continues from the last entry instead of restarting from 0.

        Args:
            base_path (str): The directory path where the CSV file will be saved.
        """
        filename = os.path.join(base_path, "stats.csv")

        new_data = pd.DataFrame(
            {
                "n_lines": self.n_lines,
                "n_trafos": self.n_trafos,
                "n_generators": self.n_generators,
            }
        )

        if os.path.exists(filename):
            # Read existing file to determine the new index start
            existing_data = pd.read_csv(filename)
            start_index = existing_data.index[-1] + 1 if not existing_data.empty else 0
            new_data.index = range(start_index, start_index + len(new_data))

            new_data.to_csv(filename, mode="a", header=False)
        else:
            new_data.to_csv(filename, index=True)

    def load(self, base_path):
        """
        Loads the tracked statistics from a CSV file.

        Args:
            base_path (str): The directory path where the CSV file is saved.
        """
        filename = base_path + "/stats.csv"
        df = pd.read_csv(filename)
        self.n_lines = df["n_lines"].values
        self.n_trafos = df["n_trafos"].values
        self.n_generators = df["n_generators"].values
