import numpy as np
import pandas as pd
from gridfm_datakit.utils.config import (
    PQ,
    PV,
    REF,
    GEN_COLUMNS,
    BUS_COLUMNS,
    DC_BUS_COLUMNS,
)
from pandapower.auxiliary import pandapowerNet
from typing import Tuple, List, Union
from pandapower import makeYbus_pypower
import pandapower as pp
import copy
from gridfm_datakit.process.solvers import run_opf, run_pf
from pandapower.pypower.idx_bus import GS, BS
from pandapower.pypower.idx_brch import (
    F_BUS,
    T_BUS,
    RATE_A,
    BR_STATUS,
    PF,
    QF,
    PT,
    QT,
    TAP,
    ANGMIN,
    ANGMAX,
)
from pandapower.pypower.makeYbus import branch_vectors
from queue import Queue
from gridfm_datakit.utils.stats import Stats
from gridfm_datakit.perturbations.topology_perturbation import TopologyGenerator
from gridfm_datakit.perturbations.generator_perturbation import GenerationGenerator
from gridfm_datakit.perturbations.admittance_perturbation import AdmittanceGenerator
import traceback


def network_preprocessing(net: pandapowerNet) -> None:
    """Adds names to bus dataframe and bus types to load, bus, gen, sgen dataframes.

    This function performs several preprocessing steps:

    1. Assigns names to all network components
    2. Determines bus types (PQ, PV, REF)
    3. Assigns bus types to connected components
    4. Performs validation checks on the network structure

    Args:
        net: The power network to preprocess.

    Raises:
        AssertionError: If network structure violates expected constraints:
            - More than one load per bus
            - REF bus not matching ext_grid connection
            - PQ bus definition mismatch
    """
    # Clean-Up things in Data-Frame // give numbered item names
    for i, row in net.bus.iterrows():
        net.bus.at[i, "name"] = "Bus " + str(i)
    for i, row in net.load.iterrows():
        net.load.at[i, "name"] = "Load " + str(i)
    for i, row in net.sgen.iterrows():
        net.sgen.at[i, "name"] = "Sgen " + str(i)
    for i, row in net.gen.iterrows():
        net.gen.at[i, "name"] = "Gen " + str(i)
    for i, row in net.shunt.iterrows():
        net.shunt.at[i, "name"] = "Shunt " + str(i)
    for i, row in net.ext_grid.iterrows():
        net.ext_grid.at[i, "name"] = "Ext_Grid " + str(i)
    for i, row in net.line.iterrows():
        net.line.at[i, "name"] = "Line " + str(i)
    for i, row in net.trafo.iterrows():
        net.trafo.at[i, "name"] = "Trafo " + str(i)

    num_buses = len(net.bus)
    bus_types = np.zeros(num_buses, dtype=int)

    # assert one slack bus
    assert len(net.ext_grid) == 1
    indices_slack = np.unique(np.array(net.ext_grid["bus"]))

    indices_PV = np.union1d(
        np.unique(np.array(net.sgen["bus"])),
        np.unique(np.array(net.gen["bus"])),
    )
    indices_PV = np.setdiff1d(
        indices_PV,
        indices_slack,
    )  # Exclude slack indices from PV indices

    indices_PQ = np.setdiff1d(
        np.arange(num_buses),
        np.union1d(indices_PV, indices_slack),
    )

    bus_types[indices_PQ] = PQ  # Set PV bus types to 1
    bus_types[indices_PV] = PV  # Set PV bus types to 2
    bus_types[indices_slack] = REF  # Set Slack bus types to 3

    net.bus["type"] = bus_types

    # assign type of the bus connected to each load and generator
    net.load["type"] = net.bus.type[net.load.bus].to_list()
    net.gen["type"] = net.bus.type[net.gen.bus].to_list()
    net.sgen["type"] = net.bus.type[net.sgen.bus].to_list()

    # there is no more than one load per bus:
    assert net.load.bus.unique().shape[0] == net.load.bus.shape[0]

    # REF bus is bus with ext grid:
    assert (
        np.where(net.bus["type"] == REF)[0]  # REF bus indicated by case file
        == net.ext_grid.bus.values
    ).all()  # Buses connected to an ext grid

    # PQ buses are buses with no gen nor ext_grid, only load or nothing connected to them
    assert (
        (net.bus["type"] == PQ)  # PQ buses indicated by case file
        == ~np.isin(
            range(net.bus.shape[0]),
            np.concatenate(
                [net.ext_grid.bus.values, net.gen.bus.values, net.sgen.bus.values],
            ),
        )
    ).all()  # Buses which are NOT connected to a gen nor an ext grid


def pf_preprocessing(net: pandapowerNet) -> pandapowerNet:
    """Sets variables to the results of OPF.

    Updates the following network components with OPF results:

    - sgen.p_mw: active power generation for static generators
    - gen.p_mw, gen.vm_pu: active power and voltage magnitude for generators

    Args:
        net: The power network to preprocess.

    Returns:
        The updated power network with OPF results.
    """
    net.sgen[["p_mw"]] = net.res_sgen[
        ["p_mw"]
    ]  # sgens are not voltage controlled, so we set P only
    net.gen[["p_mw", "vm_pu"]] = net.res_gen[["p_mw", "vm_pu"]]
    return net


def pf_post_processing(net: pandapowerNet, dcpf: bool = False) -> dict:
    """Post-processes solved network results into numpy arrays for CSV export.

    This function extracts power flow results and builds four arrays matching
    the column schemas defined in `gridfm_datakit.utils.config`:

    - Bus data with BUS_COLUMNS (+ DC_BUS_COLUMNS if dcpf=True)
    - Generator data with GEN_COLUMNS
    - Branch data with BRANCH_COLUMNS
    - Y-bus nonzero entries with [index1, index2, G, B]

    Args:
        net: The power network to process (must have solved power flow results).
        dcpf: If True, include DC power flow voltage magnitude/angle (Vm_dc, Va_dc).

    Returns:
        dict: {
            "bus": np.ndarray with bus-level features,
            "gen": np.ndarray with generator features,
            "branch": np.ndarray with branch features and admittances,
            "Y_bus": np.ndarray with nonzero Y-bus entries
        }
    """

    n_buses = net.bus.shape[0]
    n_cols = len(BUS_COLUMNS + DC_BUS_COLUMNS if dcpf else BUS_COLUMNS)
    X_bus = np.zeros((n_buses, n_cols))

    # --- Loads ---
    all_loads = net.res_load.groupby("bus")[["p_mw", "q_mvar"]].sum()

    # --- Generators (gen + sgen + ext_grid) ---
    all_gens = (
        pd.concat([net.res_gen, net.res_sgen, net.res_ext_grid], axis=0)
        .groupby("bus")[["p_mw", "q_mvar"]]
        .sum()
    )

    # --- Assert bus indices are 0..n_buses-1 ---
    assert (net.bus.index.values == np.arange(n_buses)).all()

    # Fill basic bus info
    X_bus[:, 0] = net.bus.index.values  # bus
    X_bus[all_loads.index, 1:3] = all_loads.values  # Pd, Qd

    # Generator injections (PV + REF buses)
    pv_mask = net.bus.type == PV
    ref_mask = net.bus.type == REF
    X_bus[pv_mask, 3:5] = all_gens.reindex(net.bus.index).values[pv_mask]
    X_bus[ref_mask, 3:5] = all_gens.reindex(net.bus.index).values[ref_mask]

    # Voltage
    X_bus[:, 5] = net.res_bus.vm_pu.values
    X_bus[:, 6] = net.res_bus.va_degree.values

    # one-hot encoding of bus type
    X_bus[np.arange(n_buses), 7 + net.bus.type - 1] = (
        1  # because type is 1, 2, 3, not 0, 1, 2
    )

    # vn_kv, min_vm_pu, max_vm_pu
    X_bus[:, 10] = net.bus.vn_kv.values
    X_bus[:, 11] = net.bus.min_vm_pu.values
    X_bus[:, 12] = net.bus.max_vm_pu.values

    X_bus[:, 13] = net._ppc["bus"][:, GS] / net.sn_mva
    X_bus[:, 14] = net._ppc["bus"][:, BS] / net.sn_mva

    if dcpf:
        X_bus[:, 15] = net.bus["Vm_dc"].values
        X_bus[:, 16] = net.bus["Va_dc"].values

    # --- Generator data ---
    poly = net.poly_cost[
        ["element", "et", "cp0_eur", "cp1_eur_per_mw", "cp2_eur_per_mw2"]
    ]
    dataframes = {
        "gen": (net.gen, net.res_gen),
        "sgen": (net.sgen, net.res_sgen),
        "ext_grid": (net.ext_grid, net.res_ext_grid),
    }

    gen_dfs = []
    for et, (static_df, res_df) in dataframes.items():
        columns = [
            "bus",
            "min_p_mw",
            "max_p_mw",
            "min_q_mvar",
            "max_q_mvar",
            "in_service",
        ]
        columns = [col for col in columns if col in static_df.columns]
        df = poly.loc[
            poly.et == et
        ].merge(
            static_df[columns],
            left_on="element",
            right_index=True,
            how="right",  # Otherwise we don't add generators that don't have cost information
        )
        df = df.merge(
            res_df[["p_mw", "q_mvar"]],
            left_on="element",
            right_index=True,
            how="left",
        )
        df["is_gen"] = int(et == "gen")
        df["is_sgen"] = int(et == "sgen")
        df["is_ext_grid"] = int(et == "ext_grid")
        gen_dfs.append(df)

    all_gen_data = pd.concat(gen_dfs, axis=0, ignore_index=True)
    all_gen_data = all_gen_data.sort_values(
        by=["bus", "element"],
        kind="mergesort",
    ).reset_index(drop=True)

    X_gen = all_gen_data[GEN_COLUMNS].to_numpy()

    # --- Edge (branch) info ---
    ppc = net._ppc
    to_bus = np.real(ppc["branch"][:, T_BUS])
    from_bus = np.real(ppc["branch"][:, F_BUS])
    pf = np.real(ppc["branch"][:, PF])
    qf = np.real(ppc["branch"][:, QF])
    pt = np.real(ppc["branch"][:, PT])
    qt = np.real(ppc["branch"][:, QT])
    Ytt, Yff, Yft, Ytf = branch_vectors(ppc["branch"], ppc["branch"].shape[0])
    Ytt_r = np.real(Ytt)
    Ytt_i = np.imag(Ytt)
    Yff_r = np.real(Yff)
    Yff_i = np.imag(Yff)
    Yft_r = np.real(Yft)
    Yft_i = np.imag(Yft)
    Ytf_r = np.real(Ytf)
    Ytf_i = np.imag(Ytf)
    tap = np.real(ppc["branch"][:, TAP])
    ang_min = np.real(ppc["branch"][:, ANGMIN])
    ang_max = np.real(ppc["branch"][:, ANGMAX])
    rate_a = np.real(ppc["branch"][:, RATE_A])
    br_status = np.real(ppc["branch"][:, BR_STATUS])
    branch_params = np.column_stack(
        (
            from_bus,
            to_bus,
            pf,
            qf,
            pt,
            qt,
            Yff_r,
            Yff_i,
            Yft_r,
            Yft_i,
            Ytf_r,
            Ytf_i,
            Ytt_r,
            Ytt_i,
            tap,
            ang_min,
            ang_max,
            rate_a,
            br_status,
        ),
    )

    # Y_bus

    Y_bus, Yf, Yt = makeYbus_pypower(ppc["baseMVA"], ppc["bus"], ppc["branch"])

    i, j = np.nonzero(Y_bus)
    # note that Y_bus[i,j] can be != 0 even if a branch from i to j is not in service because there might be other branches connected to the same buses

    s = Y_bus[i, j]
    G = np.real(s)
    B = np.imag(s)

    edge_index = np.column_stack((i, j))
    edge_attr = np.stack((G, B)).T
    Y_bus = np.column_stack((edge_index, edge_attr))

    return {"bus": X_bus, "gen": X_gen, "branch": branch_params, "Y_bus": Y_bus}


def process_scenario_unsecure(
    net: pandapowerNet,
    scenarios: np.ndarray,
    scenario_index: int,
    topology_generator: TopologyGenerator,
    generation_generator: GenerationGenerator,
    admittance_generator: AdmittanceGenerator,
    no_stats: bool,
    local_processed_data: List[np.ndarray],
    local_stats: Union[Stats, None],
    error_log_file: str,
    dcpf: bool,
) -> Tuple[List[np.ndarray], Union[Stats, None]]:
    """Generates one or more unsecure operating points for the given load scenario.

    In unsecure mode, OPF is run first to get generator setpoints, then topology
    perturbations are applied. This can lead to constraint violations (overloads,
    voltage violations) since the setpoints are not re-optimized for the new topology.

    Args:
        net: The power network.
        scenarios: Array of load scenarios with shape (n_loads, n_scenarios, 2).
        scenario_index: Index of the current scenario to process.
        topology_generator: Generator for topology perturbations (line/transformer outages).
        generation_generator: Generator for generation cost perturbations.
        admittance_generator: Generator for line admittance perturbations.
        no_stats: Whether to skip statistics collection.
        local_processed_data: List to accumulate processed data tuples.
        local_stats: Statistics object for collecting network performance metrics.
        error_log_file: Path to error log file for recording failures.
        dcpf: Whether to include DC power flow results in output.

    Returns:
        Tuple containing:
            - Updated list of processed data (bus, gen, branch, Y_bus arrays)
            - Updated statistics object (or None if no_stats=True)
    """
    net = copy.deepcopy(net)

    # apply the load scenario to the network
    net.load.p_mw = scenarios[:, scenario_index, 0]
    net.load.q_mvar = scenarios[:, scenario_index, 1]

    # Apply generation perturbations before OPF.
    perturbations = generation_generator.generate((x for x in [net]))

    # Apply admittance perturbations
    perturbations = admittance_generator.generate(perturbations)

    net = next(perturbations)

    # first run OPF to get the gen set points
    try:
        run_opf(net)
    except Exception as e:
        with open(error_log_file, "a") as f:
            f.write(
                f"Caught an exception at scenario {scenario_index} in run_opf function: {e}\n",
            )
        return (
            local_processed_data,
            local_stats,
        )

    net_pf = copy.deepcopy(net)
    net_pf = pf_preprocessing(net_pf)

    # Generate perturbed topologies
    perturbations = topology_generator.generate(net_pf)

    # to get unsecure points, we apply the topology perturbation after OPF.
    # The setpoints are then no longer adapted to the new topology, and might lead to e.g. abranch overload or a voltage magnitude violation once we drop an element.
    for perturbation in perturbations:
        try:
            pp.rundcpp(perturbation)
            perturbation.bus["Vm_dc"] = perturbation.res_bus.vm_pu
            perturbation.bus["Va_dc"] = perturbation.res_bus.va_degree
            run_pf(perturbation)
        except Exception as e:
            with open(error_log_file, "a") as f:
                f.write(
                    f"Caught an exception at scenario {scenario_index} when solving dcpf or in run_pf function: {e}\n",
                )
            continue

        # Append processed power flow data
        pf_data = pf_post_processing(perturbation, dcpf=dcpf)
        local_processed_data.append(
            (pf_data["bus"], pf_data["gen"], pf_data["branch"], pf_data["Y_bus"]),
        )
        if not no_stats:
            local_stats.update(perturbation)

    return local_processed_data, local_stats


def process_scenario_chunk(
    mode: str,
    start_idx: int,
    end_idx: int,
    scenarios: np.ndarray,
    net: pandapowerNet,
    progress_queue: Queue,
    topology_generator: TopologyGenerator,
    generation_generator: GenerationGenerator,
    admittance_generator: AdmittanceGenerator,
    no_stats: bool,
    error_log_path: str,
    dcpf: bool,
) -> Tuple[
    Union[None, Exception],
    Union[None, str],
    List[np.ndarray],
    Union[Stats, None],
]:
    """Process a chunk of scenarios for distributed processing.

    This function processes multiple scenarios in a single worker process,
    accumulating results before returning them to the main process.

    Args:
        mode: Processing mode ("secure" or "unsecure").
        start_idx: Starting scenario index (inclusive).
        end_idx: Ending scenario index (exclusive).
        scenarios: Array of load scenarios with shape (n_loads, n_scenarios, 2).
        net: The power network.
        progress_queue: Queue for reporting progress to main process.
        topology_generator: Generator for topology perturbations.
        generation_generator: Generator for generation cost perturbations.
        admittance_generator: Generator for line admittance perturbations.
        no_stats: Whether to skip statistics collection.
        error_log_path: Path to error log file for recording failures.
        dcpf: Whether to include DC power flow results in output.

    Returns:
        Tuple containing:
            - Exception object (None if successful)
            - Traceback string (None if successful)
            - List of processed data tuples (bus, gen, branch, Y_bus arrays)
            - Statistics object (or None if no_stats=True)
    """
    try:
        local_stats = Stats() if not no_stats else None
        local_processed_data = []
        for scenario_index in range(start_idx, end_idx):
            if mode == "secure":
                (
                    local_processed_data,
                    local_stats,
                ) = process_scenario_secure(
                    net,
                    scenarios,
                    scenario_index,
                    topology_generator,
                    generation_generator,
                    admittance_generator,
                    no_stats,
                    local_processed_data,
                    local_stats,
                    error_log_path,
                    dcpf,
                )
            elif mode == "unsecure":
                (
                    local_processed_data,
                    local_stats,
                ) = process_scenario_unsecure(
                    net,
                    scenarios,
                    scenario_index,
                    topology_generator,
                    generation_generator,
                    admittance_generator,
                    no_stats,
                    local_processed_data,
                    local_stats,
                    error_log_path,
                    dcpf,
                )

            progress_queue.put(1)  # update queue

        return (
            None,
            None,
            local_processed_data,
            local_stats,
        )
    except Exception as e:
        with open(error_log_path, "a") as f:
            f.write(f"Caught an exception in process_scenario_chunk function: {e}\n")
            f.write(traceback.format_exc())
            f.write("\n")
        for _ in range(end_idx - start_idx):
            progress_queue.put(1)
        return e, traceback.format_exc(), None, None, None


def process_scenario_secure(
    net: pandapowerNet,
    scenarios: np.ndarray,
    scenario_index: int,
    topology_generator: TopologyGenerator,
    generation_generator: GenerationGenerator,
    admittance_generator: AdmittanceGenerator,
    no_stats: bool,
    local_processed_data: List[np.ndarray],
    local_stats: Union[Stats, None],
    error_log_file: str,
    dcpf: bool,
) -> Tuple[List[np.ndarray], Union[Stats, None]]:
    """Processes a load scenario in secure mode.

    In secure mode, perturbations are applied first, then OPF is run to get
    generator setpoints that account for the perturbed topology. This ensures
    all constraints are satisfied in the final operating point.

    Args:
        net: The power network.
        scenarios: Array of load scenarios with shape (n_loads, n_scenarios, 2).
        scenario_index: Index of the current scenario to process.
        topology_generator: Generator for topology perturbations (line/transformer outages).
        generation_generator: Generator for generation cost perturbations.
        admittance_generator: Generator for line admittance perturbations.
        no_stats: Whether to skip statistics collection.
        local_processed_data: List to accumulate processed data tuples.
        local_stats: Statistics object for collecting network performance metrics.
        error_log_file: Path to error log file for recording failures.
        dcpf: Whether to include DC power flow results in output.

    Returns:
        Tuple containing:
            - Updated list of processed data (bus, gen, branch, Y_bus arrays)
            - Updated statistics object (or None if no_stats=True)
    """
    # apply the load scenario to the network
    net.load.p_mw = scenarios[:, scenario_index, 0]
    net.load.q_mvar = scenarios[:, scenario_index, 1]

    # Generate perturbed topologies
    perturbations = topology_generator.generate(net)

    # Apply generation perturbations
    perturbations = generation_generator.generate(perturbations)

    # Apply admittance perturbations
    perturbations = admittance_generator.generate(perturbations)

    for perturbation in perturbations:
        try:
            # run DCPF to get the bus voltages
            pp.rundcopp(perturbation)
            perturbation.bus["Vm_dc"] = perturbation.res_bus.vm_pu
            perturbation.bus["Va_dc"] = perturbation.res_bus.va_degree
        except Exception as e:
            perturbation.bus["Vm_dc"] = np.nan
            perturbation.bus["Va_dc"] = np.nan
            with open(error_log_file, "a") as f:
                f.write(
                    f"Caught an exception at scenario {scenario_index} in rundcopp function: {e}\n",
                )
        try:
            # run OPF to get the gen set points. Here the set points account for the topology perturbation.
            run_opf(perturbation)
        except Exception as e:
            with open(error_log_file, "a") as f:
                f.write(
                    f"Caught an exception at scenario {scenario_index} in run_opf function: {e}\n",
                )
            continue
        assert (
            perturbation.res_gen.vm_pu[perturbation.res_gen.type == 2]
            - perturbation.res_gen.vm_pu[perturbation.res_gen.type == 2]
            < 1e-3
        ).all(), "Generator voltage at PV buses is not the same after PF"

        # Append processed power flow data
        pf_data = pf_post_processing(perturbation, dcpf=dcpf)
        local_processed_data.append(
            (pf_data["bus"], pf_data["gen"], pf_data["branch"], pf_data["Y_bus"]),
        )
        if not no_stats:
            local_stats.update(perturbation)

    return local_processed_data, local_stats
