"""
Power system network processing and scenario generation.

This module provides functionality for processing power system networks,
running power flow calculations, and generating perturbed scenarios
for data generation purposes.
"""

import numpy as np
from gridfm_datakit.utils.column_names import (
    GEN_COLUMNS,
    BUS_COLUMNS,
    DC_BUS_COLUMNS,
    BRANCH_COLUMNS,
)
from typing import Tuple, List, Union, Dict, Any, Optional
from gridfm_datakit.network import makeYbus, branch_vectors
import copy
from gridfm_datakit.process.solvers import run_opf, run_pf, run_dcpf
from gridfm_datakit.utils.idx_bus import (
    GS,
    BS,
    BUS_TYPE,
    BASE_KV,
    VMIN,
    VMAX,
    PQ,
    PV,
    REF,
)
from gridfm_datakit.utils.idx_brch import SHIFT
from gridfm_datakit.utils.idx_gen import GEN_BUS, PMIN, PMAX, QMIN, QMAX
from gridfm_datakit.utils.idx_cost import NCOST, COST
from gridfm_datakit.utils.idx_brch import (
    F_BUS,
    T_BUS,
    RATE_A,
    BR_STATUS,
    TAP,
    ANGMIN,
    ANGMAX,
)
from queue import Queue
from gridfm_datakit.perturbations.topology_perturbation import TopologyGenerator
from gridfm_datakit.perturbations.generator_perturbation import GenerationGenerator
from gridfm_datakit.perturbations.admittance_perturbation import AdmittanceGenerator
import traceback
from gridfm_datakit.network import Network
from gridfm_datakit.utils.idx_bus import BUS_I, PD, QD


def init_julia() -> Any:
    """Initialize Julia interface with PowerModels.jl.

    Sets up Julia environment with required packages and defines
    power flow and optimal power flow functions.

    Returns:
        Julia interface object for running power flow calculations.

    Raises:
        RuntimeError: If Julia initialization fails.
    """
    # TODO: check if juliacall already initialized
    from juliacall import Main as jl

    try:
        jl.seval("""
        using PowerModels
        using Ipopt
        using Memento
        Memento.config!("not_set")


        function run_opf(case_file)
            result = solve_ac_opf(case_file, optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0, "tol" => 1e-6))
            result["solution"]["pf"] = false
            return result
        end

        function run_pf(case_file)
            # Solve AC power flow
            network = PowerModels.parse_file(case_file)
            result = compute_ac_pf(network)

            # Return immediately if the solver did not converge
            if result["termination_status"] == false
                return result
            end

            # Update network data
            update_data!(network, result["solution"])

            # Compute branch flows
            flows = calc_branch_flow_ac(network)

            result["solution"]["branch"] = flows["branch"]
            result["solution"]["pf"] = true
            return result
        end


        function run_dcpf(case_file)
            result = compute_dc_pf(case_file)
            result["solution"]["pf"] = true
            return result
        end
        """)
    except Exception as e:
        raise RuntimeError(f"Error initializing Julia: {e}")
    return jl


def pf_preprocessing(net: Network, res: Dict[str, Any]) -> Network:
    """Set variables to the results of OPF.

    Updates the following network components with OPF results:

    - sgen.p_mw: active power generation for static generators
    - gen.p_mw, gen.vm_pu: active power and voltage magnitude for generators

    Args:
        net: The power network to preprocess.
        res: OPF result dictionary containing solution data.

    Returns:
        Updated network with OPF results applied.
    """
    pg = [
        res["solution"]["gen"][str(i + 1)]["pg"] * net.baseMVA
        for i in net.idx_gens_in_service
    ]
    vm = [
        res["solution"]["bus"][str(net.reverse_bus_index_mapping[i])]["vm"]
        for i in range(net.buses.shape[0])
    ]

    initial_pg = net.Pg_gen.copy()
    net.Pg_gen = pg
    net.Vm = vm
    new_pg = net.Pg_gen
    assert not np.allclose(initial_pg, new_pg), "Pg has not changed"

    return net


def pf_post_processing(
    net: Network,
    res: Dict[str, Any],
    dcpf: bool = False,
) -> Dict[str, np.ndarray]:
    """Post-process solved network results into numpy arrays for CSV export.

    This function extracts power flow results and builds four arrays matching
    the column schemas defined in `gridfm_datakit.utils.column_names`:

    - Bus data with BUS_COLUMNS (+ DC_BUS_COLUMNS if dcpf=True)
    - Generator data with GEN_COLUMNS
    - Branch data with BRANCH_COLUMNS
    - Y-bus nonzero entries with [index1, index2, G, B]

    Args:
        net: The power network to process (must have solved power flow results).
        res: Power flow result dictionary containing solution data.
        dcpf: If True, include DC power flow voltage magnitude/angle (Vm_dc, Va_dc).

    Returns:
        Dictionary containing:
        - "bus": np.ndarray with bus-level features
        - "gen": np.ndarray with generator features
        - "branch": np.ndarray with branch features and admittances
        - "Y_bus": np.ndarray with nonzero Y-bus entries
    """

    # --- Bus data ---
    n_buses = net.buses.shape[0]
    n_cols = len(BUS_COLUMNS) + len(DC_BUS_COLUMNS) if dcpf else len(BUS_COLUMNS)
    X_bus = np.zeros((n_buses, n_cols))

    # --- Loads ---
    X_bus[:, 0] = net.buses[:, BUS_I]  # bus
    X_bus[:, 1] = net.buses[:, PD]
    X_bus[:, 2] = net.buses[:, QD]

    # --- Generator injections
    assert len(res["solution"]["gen"]) == len(net.idx_gens_in_service), (
        "Number of generators in solution should match number of generators in network"
    )
    pg_gen = np.array(
        [
            res["solution"]["gen"][str(i + 1)]["pg"] * net.baseMVA
            for i in net.idx_gens_in_service
        ],
    )
    qg_gen = np.array(
        [
            res["solution"]["gen"][str(i + 1)]["qg"] * net.baseMVA
            for i in net.idx_gens_in_service
        ],
    )
    gen_bus = net.gens[net.idx_gens_in_service, GEN_BUS].astype(int)
    Pg_bus = np.bincount(gen_bus, weights=pg_gen, minlength=n_buses)
    Qg_bus = np.bincount(gen_bus, weights=qg_gen, minlength=n_buses)

    assert np.all(Pg_bus[net.buses[:, BUS_TYPE] == PQ] == 0)
    assert np.all(Qg_bus[net.buses[:, BUS_TYPE] == PQ] == 0)

    X_bus[:, 3] = Pg_bus
    X_bus[:, 4] = Qg_bus

    # Voltage
    assert set([int(k) for k in res["solution"]["bus"].keys()]) == set(
        net.reverse_bus_index_mapping.values(),
    ), "Buses in solution should match buses in network"

    X_bus[:, 5] = [
        res["solution"]["bus"][str(net.reverse_bus_index_mapping[i])]["vm"]
        for i in range(n_buses)
    ]
    X_bus[:, 6] = np.rad2deg(
        [
            res["solution"]["bus"][str(net.reverse_bus_index_mapping[i])]["va"]
            for i in range(n_buses)
        ],
    )

    # one-hot encoding of bus type
    assert np.all(np.isin(net.buses[:, BUS_TYPE], [PQ, PV, REF])), (
        "Bus type should be PQ, PV, or REF, no disconnected buses (4)"
    )
    X_bus[np.arange(n_buses), 7 + net.buses[:, BUS_TYPE].astype(int) - 1] = (
        1  # because type is 1, 2, 3, not 0, 1, 2
    )

    # base_kv, min_vm_pu, max_vm_pu
    X_bus[:, 10] = net.buses[:, BASE_KV]
    X_bus[:, 11] = net.buses[:, VMIN]
    X_bus[:, 12] = net.buses[:, VMAX]

    X_bus[:, 13] = net.buses[:, GS] / net.baseMVA
    X_bus[:, 14] = net.buses[:, BS] / net.baseMVA

    if dcpf:
        X_bus[:, 15] = np.rad2deg(net.res_dc_va_rad)

    # --- Generator data ---
    assert np.all(net.gencosts[:, NCOST] == 3), "NCOST should be 3"
    n_gens = net.gens.shape[0]
    n_cols = len(GEN_COLUMNS)
    X_gen = np.zeros((n_gens, n_cols))

    X_gen[:, 0] = list(range(n_gens))
    X_gen[:, 1] = net.gens[:, GEN_BUS]
    X_gen[net.idx_gens_in_service, 2] = pg_gen  # 0 if not in service
    X_gen[net.idx_gens_in_service, 3] = qg_gen  # 0 if not in service
    X_gen[:, 4] = net.gens[:, PMIN]
    X_gen[:, 5] = net.gens[:, PMAX]
    X_gen[:, 6] = net.gens[:, QMIN]
    X_gen[:, 7] = net.gens[:, QMAX]
    X_gen[:, 8] = net.gencosts[:, COST]
    X_gen[:, 9] = net.gencosts[:, COST + 1]
    X_gen[:, 10] = net.gencosts[:, COST + 2]
    X_gen[net.idx_gens_in_service, 11] = 1

    # slack gen (can be any generator connected to the ref node)
    slack_gen_idx = np.where(net.gens[:, GEN_BUS] == net.ref_bus_idx)[0]
    X_gen[slack_gen_idx, 12] = 1

    # --- Edge (branch) info ---
    n_branches = net.branches.shape[0]
    X_branch = np.zeros((n_branches, len(BRANCH_COLUMNS)))
    X_branch[:, 0] = list(range(n_branches))
    X_branch[:, 1] = np.real(net.branches[:, F_BUS])
    X_branch[:, 2] = np.real(net.branches[:, T_BUS])

    # pf, qf, pt, qt
    if res["solution"]["pf"]:
        # when solving pf, the flow of all branches is computed, so the number of branches in solution should match the number of branches in network
        assert len(res["solution"]["branch"]) == n_branches, (
            "Number of branches in solution should match number of branches in network"
        )
    else:
        # when solving opf, the flow of only the in-service branches is computed, so the number of branches in solution should match the number of in-service branches in network
        assert len(res["solution"]["branch"]) == len(net.idx_branches_in_service), (
            "Number of branches in solution should match number of branches in network"
        )

    X_branch[net.idx_branches_in_service, 3] = np.array(
        [
            res["solution"]["branch"][str(i + 1)]["pf"] * net.baseMVA
            for i in net.idx_branches_in_service
        ],
    )
    X_branch[net.idx_branches_in_service, 4] = np.array(
        [
            res["solution"]["branch"][str(i + 1)]["qf"] * net.baseMVA
            for i in net.idx_branches_in_service
        ],
    )
    X_branch[net.idx_branches_in_service, 5] = np.array(
        [
            res["solution"]["branch"][str(i + 1)]["pt"] * net.baseMVA
            for i in net.idx_branches_in_service
        ],
    )
    X_branch[net.idx_branches_in_service, 6] = np.array(
        [
            res["solution"]["branch"][str(i + 1)]["qt"] * net.baseMVA
            for i in net.idx_branches_in_service
        ],
    )

    # admittances
    Ytt, Yff, Yft, Ytf = branch_vectors(net.branches, net.branches.shape[0])
    X_branch[:, 7] = np.real(Yff)
    X_branch[:, 8] = np.imag(Yff)
    X_branch[:, 9] = np.real(Yft)
    X_branch[:, 10] = np.imag(Yft)
    X_branch[:, 11] = np.real(Ytf)
    X_branch[:, 12] = np.imag(Ytf)
    X_branch[:, 13] = np.real(Ytt)
    X_branch[:, 14] = np.imag(Ytt)

    X_branch[:, 15] = net.branches[:, TAP]
    X_branch[:, 16] = net.branches[:, SHIFT]
    X_branch[:, 17] = net.branches[:, ANGMIN]
    X_branch[:, 18] = net.branches[:, ANGMAX]
    X_branch[:, 19] = net.branches[:, RATE_A]
    X_branch[:, 20] = net.branches[:, BR_STATUS]

    # --- Y-bus ---
    Y_bus, Yf, Yt = makeYbus(net.baseMVA, net.buses, net.branches)

    i, j = np.nonzero(Y_bus)
    # note that Y_bus[i,j] can be != 0 even if a branch from i to j is not in service because there might be other branches connected to the same buses

    s = Y_bus[i, j]
    G = np.real(s)
    B = np.imag(s)

    edge_index = np.column_stack((i, j))
    edge_attr = np.stack((G, B)).T
    Y_bus = np.column_stack((edge_index, edge_attr))

    return {"bus": X_bus, "gen": X_gen, "branch": X_branch, "Y_bus": Y_bus}


def process_scenario_pf_mode(
    net: Network,
    scenarios: np.ndarray,
    scenario_index: int,
    topology_generator: TopologyGenerator,
    generation_generator: GenerationGenerator,
    admittance_generator: AdmittanceGenerator,
    local_processed_data: List[np.ndarray],
    error_log_file: str,
    dcpf: bool,
) -> List[np.ndarray]:
    """Processes a load scenario in PF mode

    In PF mode, OPF is run first to get generator setpoints, then topology
    perturbations are applied. This can lead to constraint violations (overloads,
    voltage violations) since the setpoints are not re-optimized for the new topology.

    Args:
        net: The power network.
        scenarios: Array of load scenarios with shape (n_loads, n_scenarios, 2).
        scenario_index: Index of the current scenario to process.
        topology_generator: Generator for topology perturbations (line/transformer outages).
        generation_generator: Generator for generation cost perturbations.
        admittance_generator: Generator for line admittance perturbations.
        local_processed_data: List to accumulate processed data tuples.
        error_log_file: Path to error log file for recording failures.
        dcpf: Whether to include DC power flow results in output.

    Returns:
        Updated list of processed data (bus, gen, branch, Y_bus arrays)
    """
    jl = init_julia()
    net = copy.deepcopy(net)

    # apply the load scenario to the network
    net.Pd = scenarios[:, scenario_index, 0]
    net.Qd = scenarios[:, scenario_index, 1]

    # Apply generation perturbations before OPF.
    perturbations = generation_generator.generate((x for x in [net]))

    # Apply admittance perturbations
    perturbations = admittance_generator.generate(perturbations)

    net = next(perturbations)

    # first run OPF to get the gen set points
    try:
        res = run_opf(net, jl)
    except Exception as e:
        with open(error_log_file, "a") as f:
            f.write(
                f"Caught an exception at scenario {scenario_index} in run_opf function: {e}\n",
            )
        return local_processed_data

    net_pf = copy.deepcopy(net)
    net_pf = pf_preprocessing(net_pf, res)

    # Generate perturbed topologies
    perturbations = topology_generator.generate(net_pf)

    # to get PF points that can violate some OPF inequality constraints (to train PF solvers that can handle points outside of normal operating limits), we apply the topology perturbation after OPF.
    # The setpoints are then no longer adapted to the new topology, and might lead to e.g. abranch overload or a voltage magnitude violation once we drop an element.
    for perturbation in perturbations:
        try:
            res = run_dcpf(perturbation, jl)
            perturbation.res_dc_va_rad = [
                res["solution"]["bus"][str(perturbation.reverse_bus_index_mapping[i])][
                    "va"
                ]
                for i in range(perturbation.buses.shape[0])
            ]
        except Exception as e:
            with open(error_log_file, "a") as f:
                f.write(
                    f"Caught an exception at scenario {scenario_index} when solving dcpf function: {e}\n",
                )
            perturbation.res_dc_va_rad = np.array(
                [np.nan for i in range(perturbation.buses.shape[0])],
            )
        try:
            res = run_pf(perturbation, jl)
        except Exception as e:
            with open(error_log_file, "a") as f:
                f.write(
                    f"Caught an exception at scenario {scenario_index} when solving in run_pf function: {e}\n",
                )
            continue

        # Append processed power flow data
        pf_data = pf_post_processing(perturbation, res, dcpf)
        local_processed_data.append(
            (pf_data["bus"], pf_data["gen"], pf_data["branch"], pf_data["Y_bus"]),
        )
    return local_processed_data


def process_scenario_chunk(
    mode: str,
    start_idx: int,
    end_idx: int,
    scenarios: np.ndarray,
    net: Network,
    progress_queue: Queue,
    topology_generator: TopologyGenerator,
    generation_generator: GenerationGenerator,
    admittance_generator: AdmittanceGenerator,
    error_log_path: str,
    dcpf: bool,
) -> Tuple[
    Union[None, Exception],
    Union[None, str],
    Optional[List[np.ndarray]],
]:
    """Process a chunk of scenarios for distributed processing.

    This function processes multiple scenarios in a single worker process,
    accumulating results before returning them to the main process.

    Args:
        mode: Processing mode ("opf" or "pf").
        start_idx: Starting scenario index (inclusive).
        end_idx: Ending scenario index (exclusive).
        scenarios: Array of load scenarios with shape (n_loads, n_scenarios, 2).
        net: The power network.
        progress_queue: Queue for reporting progress to main process.
        topology_generator: Generator for topology perturbations.
        generation_generator: Generator for generation cost perturbations.
        admittance_generator: Generator for line admittance perturbations.
        error_log_path: Path to error log file for recording failures.
        dcpf: Whether to include DC power flow results in output.

    Returns:
        Tuple containing:
            - Exception object (None if successful)
            - Traceback string (None if successful)
            - List of processed data tuples (bus, gen, branch, Y_bus arrays)
    """
    try:
        local_processed_data = []
        for scenario_index in range(start_idx, end_idx):
            if mode == "opf":
                local_processed_data = process_scenario_opf_mode(
                    net,
                    scenarios,
                    scenario_index,
                    topology_generator,
                    generation_generator,
                    admittance_generator,
                    local_processed_data,
                    error_log_path,
                    dcpf,
                )
            elif mode == "pf":
                local_processed_data = process_scenario_pf_mode(
                    net,
                    scenarios,
                    scenario_index,
                    topology_generator,
                    generation_generator,
                    admittance_generator,
                    local_processed_data,
                    error_log_path,
                    dcpf,
                )

            progress_queue.put(1)  # update queue

        return (
            None,
            None,
            local_processed_data,
        )
    except Exception as e:
        with open(error_log_path, "a") as f:
            f.write(f"Caught an exception in process_scenario_chunk function: {e}\n")
            f.write(traceback.format_exc())
            f.write("\n")
        for _ in range(end_idx - start_idx):
            progress_queue.put(1)
        return e, traceback.format_exc(), None


def process_scenario_opf_mode(
    net: Network,
    scenarios: np.ndarray,
    scenario_index: int,
    topology_generator: TopologyGenerator,
    generation_generator: GenerationGenerator,
    admittance_generator: AdmittanceGenerator,
    local_processed_data: List[np.ndarray],
    error_log_file: str,
    dcpf: bool,
) -> List[np.ndarray]:
    """Processes a load scenario in OPF mode

    In OPF mode, perturbations are applied first, then OPF is run to get
    generator setpoints that account for the perturbed topology. This ensures
    all constraints are satisfied in the final operating point.

    Args:
        net: The power network.
        scenarios: Array of load scenarios with shape (n_loads, n_scenarios, 2).
        scenario_index: Index of the current scenario to process.
        topology_generator: Generator for topology perturbations (line/transformer outages).
        generation_generator: Generator for generation cost perturbations.
        admittance_generator: Generator for line admittance perturbations.
        local_processed_data: List to accumulate processed data tuples.
        error_log_file: Path to error log file for recording failures.
        dcpf: Whether to include DC power flow results in output.

    Returns:
        Updated list of processed data (bus, gen, branch, Y_bus arrays)
    """
    jl = init_julia()

    # apply the load scenario to the network
    net.Pd = scenarios[:, scenario_index, 0]
    net.Qd = scenarios[:, scenario_index, 1]

    # Generate perturbed topologies
    perturbations = topology_generator.generate(net)

    # Apply generation perturbations
    perturbations = generation_generator.generate(perturbations)

    # Apply admittance perturbations
    perturbations = admittance_generator.generate(perturbations)

    for perturbation in (
        perturbations
    ):  # (that returns copies of the network with the topology perturbation applied)
        try:
            # run DCPF to get the bus voltages (just for benchmarking purposes)
            res = run_dcpf(perturbation, jl)
            perturbation.res_dc_va_rad = [
                res["solution"]["bus"][str(i + 1)]["va"]
                for i in range(perturbation.buses.shape[0])
            ]
        except Exception as e:
            perturbation.res_dc_va_rad = np.array(
                [np.nan for i in range(perturbation.buses.shape[0])],
            )
            with open(error_log_file, "a") as f:
                f.write(
                    f"Caught an exception at scenario {scenario_index} in rundcopp function: {e}\n",
                )
        try:
            # run OPF to get the gen set points. Here the set points account for the topology perturbation.
            res = run_opf(perturbation, jl)
        except Exception as e:
            with open(error_log_file, "a") as f:
                f.write(
                    f"Caught an exception at scenario {scenario_index} in run_opf function: {e}\n",
                )
            continue

        # Append processed power flow data
        pf_data = pf_post_processing(perturbation, res, dcpf)
        local_processed_data.append(
            (pf_data["bus"], pf_data["gen"], pf_data["branch"], pf_data["Y_bus"]),
        )
    return local_processed_data
