import numpy as np
import pandas as pd
from gridfm_datakit.utils.config import PQ, PV, REF
from pandapower.auxiliary import pandapowerNet
from typing import Tuple, List, Union
from pandapower import makeYbus_pypower
import pandapower as pp
import copy
from gridfm_datakit.process.solvers import run_opf, run_pf
from pandapower.pypower.idx_brch import BR_STATUS
from queue import Queue
from gridfm_datakit.utils.stats import Stats
from gridfm_datakit.perturbations.topology_perturbation import TopologyGenerator
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


def pf_post_processing(net: pandapowerNet, dcpf: bool = False) -> np.ndarray:
    """Post-processes PF data to build the final data representation.

    Creates a matrix of shape (n_buses, 10) or (n_buses, 12) for DC power flow,
    with columns: (bus, Pd, Qd, Pg, Qg, Vm, Va, PQ, PV, REF) plus (Vm_dc, Va_dc)
    for DC power flow.

    Args:
        net: The power network to process.
        dcpf: Whether to include DC power flow results. Defaults to False.

    Returns:
        numpy.ndarray: Matrix containing the processed power flow data.
    """
    X = np.zeros((net.bus.shape[0], 12 if dcpf else 10))
    all_loads = (
        pd.concat([net.res_load])[["p_mw", "q_mvar", "bus"]].groupby("bus").sum()
    )

    all_gens = (
        pd.concat([net.res_gen, net.res_sgen, net.res_ext_grid])[
            ["p_mw", "q_mvar", "bus"]
        ]
        .groupby("bus")
        .sum()
    )

    assert (net.bus.index.values == list(range(X.shape[0]))).all()

    X[:, 0] = net.bus.index.values

    # Active and reactive power demand
    X[all_loads.index, 1] = all_loads.p_mw  # Pd
    X[all_loads.index, 2] = all_loads.q_mvar  # Qd

    # Active and reactive power generated
    X[net.bus.type == PV, 3] = all_gens.p_mw[
        net.res_bus.type == PV
    ]  # active Power generated
    X[net.bus.type == PV, 4] = all_gens.q_mvar[
        net.res_bus.type == PV
    ]  # reactive Power generated
    X[net.bus.type == REF, 3] = all_gens.p_mw[
        net.res_bus.type == REF
    ]  # active Power generated
    X[net.bus.type == REF, 4] = all_gens.q_mvar[
        net.res_bus.type == REF
    ]  # reactive Power generated

    # Voltage
    X[:, 5] = net.res_bus.vm_pu  # voltage magnitude
    X[:, 6] = net.res_bus.va_degree  # voltage angle
    X[:, 7:10] = pd.get_dummies(net.bus["type"]).values

    if dcpf:
        X[:, 10] = net.bus["Vm_dc"]
        X[:, 11] = net.bus["Va_dc"]
    return X


def get_adjacency_list(net: pandapowerNet) -> np.ndarray:
    """Gets adjacency list for network.

    Creates an adjacency list representation of the network's bus admittance matrix,
    including real and imaginary components of the admittance.

    Args:
        net: The power network.

    Returns:
        numpy.ndarray: Array containing edge indices and attributes (G, B).
    """
    ppc = net._ppc
    Y_bus, Yf, Yt = makeYbus_pypower(ppc["baseMVA"], ppc["bus"], ppc["branch"])

    i, j = np.nonzero(Y_bus)
    # note that Y_bus[i,j] can be != 0 even if a branch from i to j is not in service because there might be other branches connected to the same buses

    s = Y_bus[i, j]
    G = np.real(s)
    B = np.imag(s)

    edge_index = np.column_stack((i, j))
    edge_attr = np.stack((G, B)).T
    adjacency_lists = np.column_stack((edge_index, edge_attr))
    return adjacency_lists


def get_branch_idx_removed(branch: np.ndarray) -> List[int]:
    """Gets indices of removed branches in the network.

    Args:
        branch: Branch data array from the network.

    Returns:
        List of indices of branches that are out of service (= removed when applying topology perturbations)
    """
    in_service = branch[:, BR_STATUS]
    return np.where(in_service == 0)[0].tolist()


def process_scenario_contingency(
    net: pandapowerNet,
    scenarios: np.ndarray,
    scenario_index: int,
    generator: TopologyGenerator,
    no_stats: bool,
    local_csv_data: List[np.ndarray],
    local_adjacency_lists: List[np.ndarray],
    local_branch_idx_removed: List[List[int]],
    local_stats: Union[Stats, None],
    error_log_file: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[int]], Union[Stats, None]]:
    """Processes a load scenario for contingency analysis.

    Args:
        net: The power network.
        scenarios: Array of load scenarios.
        scenario_index: Index of the current scenario.
        generator: Topology perturbation generator.
        no_stats: Whether to skip statistics collection.
        local_csv_data: List to store processed CSV data.
        local_adjacency_lists: List to store adjacency lists.
        local_branch_idx_removed: List to store removed branch indices.
        local_stats: Statistics object for collecting network statistics.
        error_log_file: Path to error log file.

    Returns:
        Tuple containing:
            - List of processed CSV data
            - List of adjacency lists
            - List of removed branch indices
            - Statistics object
    """
    net = copy.deepcopy(net)

    # apply the load scenario to the network
    net.load.p_mw = scenarios[:, scenario_index, 0]
    net.load.q_mvar = scenarios[:, scenario_index, 1]

    # first run OPF to get the gen set points
    try:
        run_opf(net)
    except Exception as e:
        with open(error_log_file, "a") as f:
            f.write(
                f"Caught an exception at scenario {scenario_index} in run_opf function: {e}\n",
            )
        return (
            local_csv_data,
            local_adjacency_lists,
            local_branch_idx_removed,
            local_stats,
        )

    net_pf = copy.deepcopy(net)
    net_pf = pf_preprocessing(net_pf)

    # Generate perturbed topologies
    perturbed_topologies = generator.generate(net_pf)

    # to simulate contingency, we apply the topology perturbation after OPF
    for perturbed_topology in perturbed_topologies:
        try:
            # run DCPF for benchmarking purposes
            pp.rundcpp(perturbed_topology)
            perturbed_topology.bus["Vm_dc"] = perturbed_topology.res_bus.vm_pu
            perturbed_topology.bus["Va_dc"] = perturbed_topology.res_bus.va_degree
            # run AC-PF to get the new state of the network after contingency (we don't model any remedial actions)
            run_pf(perturbed_topology)
        except Exception as e:
            with open(error_log_file, "a") as f:
                f.write(
                    f"Caught an exception at scenario {scenario_index} when solving dcpf or in in run_pf function: {e}\n",
                )

                continue

                # TODO: What to do when the network does not converge for AC-PF? -> we dont have targets for regression!!

        # Append processed power flow data
        local_csv_data.extend(pf_post_processing(perturbed_topology, dcpf=True))
        local_adjacency_lists.append(get_adjacency_list(perturbed_topology))
        local_branch_idx_removed.append(
            get_branch_idx_removed(perturbed_topology._ppc["branch"]),
        )
        if not no_stats:
            local_stats.update(perturbed_topology)

    return local_csv_data, local_adjacency_lists, local_branch_idx_removed, local_stats


def process_scenario_chunk(
    mode,
    start_idx: int,
    end_idx: int,
    scenarios: np.ndarray,
    net: pandapowerNet,
    progress_queue: Queue,
    generator,
    no_stats: bool,
    error_log_path,
) -> Tuple[
    Union[None, Exception],
    Union[None, str],
    List[np.ndarray],
    List[np.ndarray],
    List[List[int]],
    Union[Stats, None],
]:
    """
    Create data for all scenarios in scenario indexed between start_idx and end_idx
    """
    try:
        local_stats = Stats() if not no_stats else None
        local_csv_data = []
        local_adjacency_lists = []
        local_branch_idx_removed = []
        for scenario_index in range(start_idx, end_idx):
            if mode == "pf":
                (
                    local_csv_data,
                    local_adjacency_lists,
                    local_branch_idx_removed,
                    local_stats,
                ) = process_scenario(
                    net,
                    scenarios,
                    scenario_index,
                    generator,
                    no_stats,
                    local_csv_data,
                    local_adjacency_lists,
                    local_branch_idx_removed,
                    local_stats,
                    error_log_path,
                )
            elif mode == "contingency":
                (
                    local_csv_data,
                    local_adjacency_lists,
                    local_branch_idx_removed,
                    local_stats,
                ) = process_scenario_contingency(
                    net,
                    scenarios,
                    scenario_index,
                    generator,
                    no_stats,
                    local_csv_data,
                    local_adjacency_lists,
                    local_branch_idx_removed,
                    local_stats,
                    error_log_path,
                )

            progress_queue.put(1)  # update queue

        return (
            None,
            None,
            local_csv_data,
            local_adjacency_lists,
            local_branch_idx_removed,
            local_stats,
        )
    except Exception as e:
        with open(error_log_path, "a") as f:
            f.write(f"Caught an exception in process_scenario_chunk function: {e}\n")
            f.write(traceback.format_exc())
            f.write("\n")
        for _ in range(end_idx - start_idx):
            progress_queue.put(1)
        return e, traceback.format_exc(), None, None, None, None


def process_scenario(
    net: pandapowerNet,
    scenarios: np.ndarray,
    scenario_index: int,
    generator: TopologyGenerator,
    no_stats: bool,
    local_csv_data: List[np.ndarray],
    local_adjacency_lists: List[np.ndarray],
    local_branch_idx_removed: List[List[int]],
    local_stats: Union[Stats, None],
    error_log_file: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[int]], Union[Stats, None]]:
    """Processes a load scenario.

    Args:
        net: The power network.
        scenarios: Array of load scenarios.
        scenario_index: Index of the current scenario.
        generator: Topology perturbation generator.
        no_stats: Whether to skip statistics collection.
        local_csv_data: List to store processed CSV data.
        local_adjacency_lists: List to store adjacency lists.
        local_branch_idx_removed: List to store removed branch indices.
        local_stats: Statistics object for collecting network statistics.
        error_log_file: Path to error log file.

    Returns:
        Tuple containing:
            - List of processed CSV data
            - List of adjacency lists
            - List of removed branch indices
            - Statistics object
    """
    # apply the load scenario to the network
    net.load.p_mw = scenarios[:, scenario_index, 0]
    net.load.q_mvar = scenarios[:, scenario_index, 1]
    # Generate perturbed topologies
    perturbed_topologies = generator.generate(net)

    for perturbed_topology in perturbed_topologies:
        try:
            # run OPF to get the gen set points. Here the set points account for the topology perturbation.
            run_opf(perturbed_topology)
        except Exception as e:
            with open(error_log_file, "a") as f:
                f.write(
                    f"Caught an exception at scenario {scenario_index} in run_opf function: {e}\n",
                )
            continue

        net_pf = copy.deepcopy(perturbed_topology)

        net_pf = pf_preprocessing(net_pf)

        try:
            # This is not striclty necessary as we havent't changed the setpoints nor the load since we solved OPF, but it gives more accurate PF results
            run_pf(net_pf)

        except Exception as e:
            with open(error_log_file, "a") as f:
                f.write(
                    f"Caught an exception at scenario {scenario_index} in run_pf function: {e}\n",
                )
            continue

        assert (
            net.res_line.loc[net.line.in_service == 0, "loading_percent"] == 0
        ).all(), "Line loading percent is not 0 where the line is not in service"
        assert (
            net.res_trafo.loc[net.trafo.in_service == 0, "loading_percent"] == 0
        ).all(), "Trafo loading percent is not 0 where the trafo is not in service"

        assert (
            net_pf.res_gen.vm_pu[net_pf.res_gen.type == 2]
            - perturbed_topology.res_gen.vm_pu[perturbed_topology.res_gen.type == 2]
            < 1e-3
        ).all(), "Generator voltage at PV buses is not the same after PF"

        # Append processed power flow data
        local_csv_data.extend(pf_post_processing(net_pf))
        local_adjacency_lists.append(get_adjacency_list(net_pf))
        local_branch_idx_removed.append(get_branch_idx_removed(net_pf._ppc["branch"]))
        if not no_stats:
            local_stats.update(net_pf)

    return local_csv_data, local_adjacency_lists, local_branch_idx_removed, local_stats
