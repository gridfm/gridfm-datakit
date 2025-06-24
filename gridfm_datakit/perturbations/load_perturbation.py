import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from importlib import resources
import pandapower as pp
from abc import ABC, abstractmethod
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from copy import deepcopy
from pandapower.auxiliary import pandapowerNet


def load_scenarios_to_df(scenarios: np.ndarray) -> pd.DataFrame:
    """Converts load scenarios array to a DataFrame.

    Args:
        scenarios: 3D numpy array of shape (n_loads, n_scenarios, 2) containing p_mw and q_mvar values.

    Returns:
        DataFrame with columns: load_scenario, load, p_mw, q_mvar.
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


def plot_load_scenarios_combined(df: pd.DataFrame, output_file: str) -> None:
    """Generates a combined plot of active and reactive power load scenarios.

    Creates a two-subplot figure with p_mw and q_mvar plots, one line per bus.

    Args:
        df: DataFrame containing load scenarios with columns: load_scenario, load, p_mw, q_mvar.
        output_file: Path where the HTML plot file should be saved.
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
    """Abstract base class for load scenario generators.

    This class defines the interface and common functionality for generating
    load scenarios for power grid networks.
    """

    @abstractmethod
    def __call__(
        self,
        net: pandapowerNet,
        n_scenarios: int,
        scenario_log: str,
    ) -> np.ndarray:
        """Generates load scenarios for a power network.

        Args:
            net: The power network.
            n_scenarios: Number of scenarios to generate.
            scenario_log: Path to log file for scenario generation details.

        Returns:
            numpy.ndarray: Array of shape (n_loads, n_scenarios, 2) containing p_mw and q_mvar values.
        """
        pass

    @staticmethod
    def interpolate_row(row: np.ndarray, data_points: int) -> np.ndarray:
        """Interpolates a row of data to match the desired number of data points.

        Args:
            row: Input data array to interpolate.
            data_points: Number of points in the output array.

        Returns:
            numpy.ndarray: Interpolated data array of length data_points.
        """
        if np.all(row == 0):
            return np.zeros(data_points)
        x_original = np.linspace(1, len(row), len(row))
        x_target = np.linspace(1, len(row), data_points)
        return interp1d(x_original, row, kind="linear")(x_target)

    @staticmethod
    def find_largest_scaling_factor(
        net: pandapowerNet,
        max_scaling: float,
        step_size: float,
        start: float,
        change_reactive_power: bool,
    ) -> float:
        """Finds the largest load scaling factor that maintains OPF convergence.

        Args:
            net: The power network.
            max_scaling: Maximum scaling factor to try.
            step_size: Increment for scaling factor search.
            start: Starting scaling factor.
            change_reactive_power: Whether to scale reactive power.

        Returns:
            float: Largest scaling factor that maintains OPF convergence.

        Raises:
            RuntimeError: If OPF does not converge for the starting value.
        """
        net = deepcopy(
            net,
        )  # Without this, we change the base load of the network, whioch impacts the entire data gen
        p_ref = net.load["p_mw"]
        q_ref = net.load["q_mvar"]
        u = start

        # find upper limit
        converged = True
        print("Finding upper limit u .", end="", flush=True)

        while (u <= max_scaling) and (converged is True):
            net.load["p_mw"] = p_ref * u  # we scale the active power
            if change_reactive_power:  # if we want to change the reactive power, we scale it by the same factor as the active power
                net.load["q_mvar"] = q_ref * u
            else:
                net.load["q_mvar"] = q_ref

            try:
                pp.runopp(net, numba=True)
                u += step_size  # we increment the scaling factor by the step size
                print(".", end="", flush=True)
            except pp.OPFNotConverged as err:
                if u == start:
                    raise RuntimeError(
                        f"OPF did not converge for the starting value of u={u:.3f}, {err}",
                    )
                print(
                    f"\nOPF did not converge for u={u:.3f}. Using u={u - step_size:.3f} for upper limit",
                    flush=True,
                )
                u -= step_size
                converged = False

        return u

    @staticmethod
    def min_max_scale(series: np.ndarray, new_min: float, new_max: float) -> np.ndarray:
        """Scales a series of values to a new range using min-max normalization.

        Args:
            series: Input data array to scale.
            new_min: Minimum value of the output range.
            new_max: Maximum value of the output range.

        Returns:
            numpy.ndarray: Scaled data array.
        """
        old_min, old_max = np.min(series), np.max(series)
        if old_max == old_min:
            return np.ones_like(series) * new_min
        else:
            return new_min + (series - old_min) * (new_max - new_min) / (
                old_max - old_min
            )


class LoadScenariosFromAggProfile(LoadScenarioGeneratorBase):
    r"""
    Generates load scenarios by scaling an aggregated load profile and adding local noise.

    **Overview**

    This generator uses an aggregated load profile (a time series of normalized demand values)
    to simulate realistic variations in load over time. The process includes:

    1. Determining an upper bound `u` for load scaling such that the network still
       supports a feasible optimal power flow (OPF) solution.
    2. Setting the lower bound $l = (1 - \text{global\textunderscore range}) \cdot u$.
    3. Min-max scaling the aggregate profile to the interval \([l, u]\).
    4. Applying this global scaling factor to each load's nominal value with additive uniform noise.

    **Mathematical Model**

    Let:

    - $n$: Number of loads ($i \in \{1, \dots, n\}$)

    - $K$: Number of scenarios ($k \in \{1, \dots, K\}$)

    - $(p, q) \in (\mathbb{R}_{\geq 0}^n)^2$: Nominal active/reactive loads

    - $\text{agg}^k$: Aggregated load profile value at time step $k$

    - $u$: Maximum feasible global scaling factor (from OPF)

    - $l = (1 - \text{global\textunderscore range}) \cdot u$: Minimum global scaling factor

    - $\text{ref}^k = \text{MinMaxScale}(\text{agg}^k, [l, u])$: Scaled aggregate profile

    - $\varepsilon_i^k \sim \mathcal{U}(1 - \sigma, 1 + \sigma)$: Active power noise

    - $\eta_i^k \sim \mathcal{U}(1 - \sigma, 1 + \sigma)$: Reactive power noise (if enabled)

    Then for each load $i$ and scenario $k$:

    For each load $i$ and scenario $k$:
    $$
    \tilde{p}_i^k = p_i \cdot \text{ref}^k \cdot \varepsilon_i^k
    $$

    $$
    \tilde{q}_i^k =
    \begin{cases}
    q_i \cdot \text{ref}^k \cdot \eta_i^k & \text{if } \texttt{change\textunderscore reactive\textunderscore power} = \texttt{True} \\
    q_i & \text{otherwise}
    \end{cases}
    $$

    **Notes**

    - The upper bound `u` is automatically determined by gradually increasing the base load and solving the OPF until it fails.

    - The lower bound `l` is computed as a relative percentage (1-`global_range`) of `u`.

    - Noise helps simulate local variability across loads within a global trend.
    """

    def __init__(
        self,
        agg_load_name: str,
        sigma: float,
        change_reactive_power: bool,
        global_range: float,
        max_scaling_factor: float,
        step_size: float,
        start_scaling_factor: float,
    ):
        """Initializes the load scenario generator.

        Args:
            agg_load_name: Name of the aggregated load profile file.
            sigma: Standard deviation for noise addition.
            change_reactive_power: Whether to scale reactive power.
            global_range: Range for scaling factor.
            max_scaling_factor: Maximum scaling factor to try.
            step_size: Increment for scaling factor search.
            start_scaling_factor: Starting scaling factor.
        """
        self.agg_load_name = agg_load_name
        self.sigma = sigma
        self.change_reactive_power = change_reactive_power
        self.global_range = global_range
        self.max_scaling_factor = max_scaling_factor
        self.step_size = step_size
        self.start_scaling_factor = start_scaling_factor

    def __call__(
        self,
        net: pandapowerNet,
        n_scenarios: int,
        scenarios_log: str,
    ) -> np.ndarray:
        """Generates load profiles based on aggregated load data.

        Args:
            net: The power network.
            n_scenarios: Number of scenarios to generate.
            scenarios_log: Path to log file for scenario generation details.

        Returns:
            numpy.ndarray: Array of shape (n_loads, n_scenarios, 2) containing p_mw and q_mvar values.

        Raises:
            ValueError: If start_scaling_factor is less than global_range.
        """
        if (
            self.start_scaling_factor - self.global_range * self.start_scaling_factor
            < 0
        ):
            raise ValueError(
                "The start scaling factor must be larger than the global range.",
            )

        u = self.find_largest_scaling_factor(
            net,
            max_scaling=self.max_scaling_factor,
            step_size=self.step_size,
            start=self.start_scaling_factor,
            change_reactive_power=self.change_reactive_power,
        )
        lower = (
            u - self.global_range * u
        )  # The lower bound used to be set as e.g. u - 40%, while now it is set as u - 40% of u

        with open(scenarios_log, "a") as f:
            f.write("u=" + str(u) + "\n")
            f.write("l=" + str(lower) + "\n")

        agg_load_path = resources.files("gridfm_datakit.load_profiles").joinpath(
            f"{self.agg_load_name}.csv",
        )
        agg_load = pd.read_csv(agg_load_path).to_numpy()
        agg_load = agg_load.reshape(agg_load.shape[0])
        ref_curve = self.min_max_scale(agg_load, lower, u)
        print("min, max of ref_curve: {}, {}".format(ref_curve.min(), ref_curve.max()))
        print("l, u: {}, {}".format(lower, u))

        p_mw_array = net.load.p_mw.to_numpy()  # we now store the active power at the load elements level, we don't aggregate it at the bus level (which was probably not changing anything since there is usually max one load per bus)

        q_mvar_array = net.load.q_mvar.to_numpy()

        # if the number of requested scenarios is smaller than the number of timesteps in the load profile, we cut the load profile
        if n_scenarios <= ref_curve.shape[0]:
            print(
                "cutting the load profile (original length: {}, requested length: {})".format(
                    ref_curve.shape[0],
                    n_scenarios,
                ),
            )
            ref_curve = ref_curve[:n_scenarios]
        # if it is larger, we interpolate it
        else:
            print(
                "interpolating the load profile (original length: {}, requested length: {})".format(
                    ref_curve.shape[0],
                    n_scenarios,
                ),
            )
            ref_curve = self.interpolate_row(ref_curve, data_points=n_scenarios)

        load_profile_pmw = p_mw_array[:, np.newaxis] * ref_curve
        noise = np.random.uniform(
            1 - self.sigma,
            1 + self.sigma,
            size=load_profile_pmw.shape,
        )  # Add uniform noise
        load_profile_pmw *= noise

        if self.change_reactive_power:
            load_profile_qmvar = q_mvar_array[:, np.newaxis] * ref_curve
            noise = np.random.uniform(
                1 - self.sigma,
                1 + self.sigma,
                size=load_profile_qmvar.shape,
            )  # Add uniform noise
            load_profile_qmvar *= noise
        else:
            load_profile_qmvar = q_mvar_array[:, np.newaxis] * np.ones_like(ref_curve)
            print("No change in reactive power across scenarios")

        # Stack profiles along the last dimension
        load_profiles = np.stack((load_profile_pmw, load_profile_qmvar), axis=-1)

        return load_profiles


class Powergraph(LoadScenarioGeneratorBase):
    r"""
    Load scenario generator using the PowerGraph method.

    Generates load scenarios by scaling the nominal active power profile
    with a normalized reference curve while keeping reactive power fixed.

    **Mathematical Model**

    Let:

    - $n$: Number of loads (indexed by $i \in \{1, \dots, n\}$)
    - $K$: Number of scenarios (indexed by $k \in \{1, \dots, K\}$)
    - $(p, q) \in (\mathbb{R}_{\geq 0}^n)^2$: Nominal active and reactive load vectors
    - $\text{ref}^k \in [0, 1]$: Normalized aggregate reference profile at scenario $k$
    - $(\tilde{p}_i^k, \tilde{q}_i^k) \in \mathbb{R}_{\geq 0}^2$: Active/reactive load at bus $i$ in scenario $k$

    The reference profile is computed by normalizing an aggregated profile:

    $$
    \text{ref}^k = \frac{\text{agg}^k}{\max_k \text{agg}^k}
    $$

    Then, for each bus $i$ and scenario $k$:

    $$
    \tilde{p}_i^k = p_i \cdot \text{ref}^k
    $$

    and reactive power is kept constant:

    $$
    \tilde{q}_i^k = q_i
    $$"""

    def __init__(
        self,
        agg_load_name: str,
    ):
        """Initializes the powergraph load scenario generator.

        Args:
            agg_load_name: Name of the aggregated load profile file.
        """
        self.agg_load_name = agg_load_name

    def __call__(
        self,
        net: pandapowerNet,
        n_scenarios: int,
        scenario_log: str,
    ) -> np.ndarray:
        """Generates load profiles based on aggregated load data.

        Args:
            net: The power network.
            n_scenarios: Number of scenarios to generate.
            scenario_log: Path to log file for scenario generation details.

        Returns:
            numpy.ndarray: Array of shape (n_loads, n_scenarios, 2) containing p_mw and q_mvar values.
        """
        agg_load_path = resources.files("gridfm_datakit.load_profiles").joinpath(
            f"{self.agg_load_name}.csv",
        )
        agg_load = pd.read_csv(agg_load_path).to_numpy()
        agg_load = agg_load.reshape(agg_load.shape[0])
        ref_curve = agg_load / agg_load.max()
        print("u={}, l={}".format(ref_curve.max(), ref_curve.min()))

        p_mw_array = net.load.p_mw.to_numpy()  # we now store the active power at the load elements level, we don't aggregate it at the bus level (which was probably not changing anything since there is usually max one load per bus)

        q_mvar_array = net.load.q_mvar.to_numpy()

        # if the number of requested scenarios is smaller than the number of timesteps in the load profile, we cut the load profile
        if n_scenarios <= ref_curve.shape[0]:
            print(
                "cutting the load profile (original length: {}, requested length: {})".format(
                    ref_curve.shape[0],
                    n_scenarios,
                ),
            )
            ref_curve = ref_curve[:n_scenarios]
        # if it is larger, we interpolate it
        else:
            print(
                "interpolating the load profile (original length: {}, requested length: {})".format(
                    ref_curve.shape[0],
                    n_scenarios,
                ),
            )
            ref_curve = self.interpolate_row(ref_curve, data_points=n_scenarios)

        load_profile_pmw = p_mw_array[:, np.newaxis] * ref_curve
        load_profile_qmvar = q_mvar_array[:, np.newaxis] * np.ones_like(ref_curve)
        print("No change in reactive power across scenarios")

        # Stack profiles along the last dimension
        load_profiles = np.stack((load_profile_pmw, load_profile_qmvar), axis=-1)

        return load_profiles
