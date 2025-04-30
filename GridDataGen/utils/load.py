import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from GridDataGen.utils.io import load_net_from_pglib
import os
from importlib import resources
import pandapower as pp
from abc import ABC, abstractmethod
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from copy import deepcopy


def load_scenarios_to_df(scenarios: np.ndarray) -> pd.DataFrame:
    """
    convert load scenarios to df.
    """

    n_loads = scenarios.shape[0]
    n_scenarios = scenarios.shape[1]

    # Flatten the array
    reshaped_array = scenarios.reshape((-1, 2), order="F")

    # Create a DataFrame
    df = pd.DataFrame(reshaped_array, columns=["p_mw", "q_mvar"])

    # Create load_scenario and bus columns
    load_idx = np.tile(np.arange(n_loads), n_scenarios)
    scenarios_idx = np.repeat(np.arange(n_scenarios), n_loads)

    df.insert(0, "load_scenario", scenarios_idx)
    df.insert(1, "load", load_idx)

    return df


def plot_load_scenarios_combined(df, output_file):
    """
    Generate a combined Plotly plot with p_mw and q_mvar in the same file, one line per bus.

    Parameters:
    - df: DataFrame containing the load scenarios
    - output_file: File name for the combined plot
    """
    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("p_mw", "q_mvar"),
    )

    # Add p_mw plot
    for load in df["load"].unique():
        df_load = df[df["load"] == load]
        fig.add_trace(
            go.Scatter(
                x=df_load["load_scenario"],
                y=df_load["p_mw"],
                mode="lines",
                name=f"Load {load} p_mw",
            ),
            row=1,
            col=1,
        )

    # Add q_mvar plot
    for load in df["load"].unique():
        df_load = df[df["load"] == load]
        fig.add_trace(
            go.Scatter(
                x=df_load["load_scenario"],
                y=df_load["q_mvar"],
                mode="lines",
                name=f"Load {load} q_mvar",
            ),
            row=2,
            col=1,
        )

    # Update layout
    fig.update_layout(height=800, width=1500, title_text="Load Scenarios")

    # Save the combined plot to an HTML file
    fig.write_html(output_file)


class LoadScenarioGeneratorBase(ABC):
    """Abstract base class for load scenario generators."""

    @abstractmethod
    def __call__(self, net, n_scenarios, scenario_log):
        pass

    @staticmethod
    def interpolate_row(row, data_points):
        """Interpolate a row of data to match the desired number of data points."""
        if np.all(row == 0):
            return np.zeros(data_points)
        x_original = np.linspace(1, len(row), len(row))
        x_target = np.linspace(1, len(row), data_points)
        return interp1d(x_original, row, kind="linear")(x_target)

    @staticmethod
    def find_largest_scaling_factor(
        net, max_scaling, step_size, start, change_reactive_power
    ):
        net = deepcopy(
            net
        )  ## Without this, we change the base load of the network, whioch impacts the entire data gen
        p_ref = net.load["p_mw"]
        q_ref = net.load["q_mvar"]
        u = start

        # find upper limit
        converged = True
        print("Finding upper limit u .", end="", flush=True)

        while (u <= max_scaling) and (converged == True):
            net.load["p_mw"] = p_ref * u
            if change_reactive_power:
                net.load["q_mvar"] = q_ref * u
            else:
                net.load["q_mvar"] = q_ref

            try:
                pp.runopp(net, numba=True)
                u += step_size
                print(".", end="", flush=True)
            except pp.OPFNotConverged as err:
                if u == start:
                    raise RuntimeError(
                        f"OPF did not converge for the starting value of u={u:.3f}"
                    )
                print(
                    f"\nOPF did not converge for u={u:.3f}. Using u={u-step_size:.3f} for upper limit",
                    flush=True,
                )
                u -= step_size
                converged = False

        return u

    @staticmethod
    def min_max_scale(series, new_min, new_max):
        """
        Scale a series of values to a new range using min-max normalization.
        """
        old_min, old_max = np.min(series), np.max(series)
        return new_min + (series - old_min) * (new_max - new_min) / (old_max - old_min)


class LoadScenariosFromAggProfile(LoadScenarioGeneratorBase):
    """Load scenario generator using aggregated load profiles."""

    def __init__(
        self,
        agg_load_name,
        sigma,
        change_reactive_power,
        global_range,
        max_scaling_factor,
        step_size,
        start_scaling_factor,
    ):
        self.agg_load_name = agg_load_name
        self.sigma = sigma
        self.change_reactive_power = change_reactive_power
        self.global_range = global_range
        self.max_scaling_factor = max_scaling_factor
        self.step_size = step_size
        self.start_scaling_factor = start_scaling_factor

    def __call__(self, net, n_scenarios, scenarios_log):
        """Generate load profiles for a power grid based on aggregated load data."""

        if self.start_scaling_factor - self.global_range < 0:
            raise ValueError(
                "The start scaling factor must be larger than the global range."
            )

        u = self.find_largest_scaling_factor(
            net,
            max_scaling=self.max_scaling_factor,
            step_size=self.step_size,
            start=self.start_scaling_factor,
            change_reactive_power=self.change_reactive_power,
        )
        l = (
            u - self.global_range * u
        )  ## The lower bound used to be set as e.g. u - 40%, while now it is set as u - 40% of u (otherwise the lower bound is too low when u is big)

        with open(scenarios_log, "a") as f:
            f.write("u=" + str(u) + "\n")
            f.write("l=" + str(l) + "\n")

        agg_load_path = resources.files(f"GridDataGen.load_profiles").joinpath(
            f"{self.agg_load_name}.csv"
        )
        agg_load = pd.read_csv(agg_load_path).to_numpy()
        agg_load = agg_load.reshape(agg_load.shape[0])
        ref_curve = self.min_max_scale(agg_load, l, u)
        print("min, max of ref_curve: {}, {}".format(ref_curve.min(), ref_curve.max()))
        print("l, u: {}, {}".format(l, u))

        p_mw_array = (
            net.load.p_mw.to_numpy()
        )  # we now store the active power at the load elements level, we don't aggregate it at the bus level (which was probably not changing anything since there is usually max one load per bus)

        q_mvar_array = net.load.q_mvar.to_numpy()

        # if the number of requested scenarios is smaller than the number of timesteps in the load profile, we cut the load profile
        if n_scenarios <= ref_curve.shape[0]:
            print(
                "cutting the load profile (original length: {}, requested length: {})".format(
                    ref_curve.shape[0], n_scenarios
                )
            )
            ref_curve = ref_curve[:n_scenarios]
        # if it is larger, we interpolate it
        else:
            print(
                "interpolating the load profile (original length: {}, requested length: {})".format(
                    ref_curve.shape[0], n_scenarios
                )
            )
            ref_curve = self.interpolate_row(ref_curve, data_points=n_scenarios)

        load_profile_pmw = p_mw_array[:, np.newaxis] * ref_curve
        noise = np.random.uniform(
            1 - self.sigma, 1 + self.sigma, size=load_profile_pmw.shape
        )  # Add uniform noise
        load_profile_pmw *= noise

        if self.change_reactive_power:
            load_profile_qmvar = q_mvar_array[:, np.newaxis] * ref_curve
            noise = np.random.uniform(
                1 - self.sigma, 1 + self.sigma, size=load_profile_qmvar.shape
            )  # Add uniform noise
            load_profile_qmvar *= noise
        else:
            load_profile_qmvar = q_mvar_array[:, np.newaxis] * np.ones_like(ref_curve)
            print("No change in reactive power across scenarios")

        # Stack profiles along the last dimension
        load_profiles = np.stack((load_profile_pmw, load_profile_qmvar), axis=-1)

        return load_profiles


class Powergraph(LoadScenarioGeneratorBase):
    """Load scenario generator as in powergraph."""

    def __init__(
        self,
        agg_load_name,
    ):
        self.agg_load_name = agg_load_name

    def __call__(self, net, n_scenarios, scenario_log):
        """Generate load profiles for a power grid based on aggregated load data."""

        agg_load_path = resources.files(f"GridDataGen.load_profiles").joinpath(
            f"{self.agg_load_name}.csv"
        )
        agg_load = pd.read_csv(agg_load_path).to_numpy()
        agg_load = agg_load.reshape(agg_load.shape[0])
        ref_curve = agg_load / agg_load.max()
        print("u={}, l={}".format(ref_curve.max(), ref_curve.min()))

        p_mw_array = (
            net.load.p_mw.to_numpy()
        )  # we now store the active power at the load elements level, we don't aggregate it at the bus level (which was probably not changing anything since there is usually max one load per bus)

        q_mvar_array = net.load.q_mvar.to_numpy()

        # if the number of requested scenarios is smaller than the number of timesteps in the load profile, we cut the load profile
        if n_scenarios <= ref_curve.shape[0]:
            print(
                "cutting the load profile (original length: {}, requested length: {})".format(
                    ref_curve.shape[0], n_scenarios
                )
            )
            ref_curve = ref_curve[:n_scenarios]
        # if it is larger, we interpolate it
        else:
            print(
                "interpolating the load profile (original length: {}, requested length: {})".format(
                    ref_curve.shape[0], n_scenarios
                )
            )
            ref_curve = self.interpolate_row(ref_curve, data_points=n_scenarios)

        load_profile_pmw = p_mw_array[:, np.newaxis] * ref_curve
        load_profile_qmvar = q_mvar_array[:, np.newaxis] * np.ones_like(ref_curve)
        print("No change in reactive power across scenarios")

        # Stack profiles along the last dimension
        load_profiles = np.stack((load_profile_pmw, load_profile_qmvar), axis=-1)

        return load_profiles
