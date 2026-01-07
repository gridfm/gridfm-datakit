"""
Power system network processing and scenario generation.

This module provides functionality for processing power system networks,
running power flow calculations, and generating perturbed scenarios
for data generation purposes.
"""

import numpy as np
from importlib import resources
from gridfm_datakit.utils.column_names import (
    GEN_COLUMNS,
    DC_GEN_COLUMNS,
    BUS_COLUMNS,
    DC_BUS_COLUMNS,
    BRANCH_COLUMNS,
    DC_BRANCH_COLUMNS,
    RUNTIME_COLUMNS,
    DC_RUNTIME_COLUMNS,
)
import os
from typing import Tuple, List, Union, Dict, Any, Optional
from gridfm_datakit.network import makeYbus, branch_vectors
import copy
from gridfm_datakit.process.solvers import run_opf, run_pf, run_dcpf, run_dcopf
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
    BR_R,
    BR_X,
    BR_B,
)
from queue import Queue
from gridfm_datakit.perturbations.topology_perturbation import TopologyGenerator
from gridfm_datakit.perturbations.generator_perturbation import GenerationGenerator
from gridfm_datakit.perturbations.admittance_perturbation import AdmittanceGenerator
import traceback
from gridfm_datakit.network import Network
from gridfm_datakit.utils.idx_bus import BUS_I, PD, QD
import multiprocessing


def init_julia(
    max_iter: int,
    solver_log_dir: str = None,
    dc_max_iter: Optional[int] = None,
    print_level: Optional[int] = None,
) -> Any:
    """Initialize Julia interface with PowerModels.jl.

    Sets up Julia environment and defines AC OPF/PF/DCPF entrypoints.

    Args:
        max_iter: Maximum number of iterations for AC OPF solver.
        solver_log_dir: If provided, enable OPF/PF logging to files under this
            directory using per-process names (opf_<proc>.log, pf_<proc>.log).
            If None, logging is disabled.
        dc_max_iter: Maximum number of iterations for DC OPF solver. If None, uses 1000.
        print_level: Ipopt print level. If None, uses 0 when solver_log_dir is None, else 5.

    Returns:
        Julia interface object for running power flow calculations.

    Raises:
        RuntimeError: If Julia initialization fails.
    """
    from juliacall import Main as jl

    # Decide log paths and Ipopt print levels
    proc = multiprocessing.current_process().name

    opf_solver_log_file = (
        ""
        if solver_log_dir is None
        else os.path.join(solver_log_dir, "opf_" + str(proc) + ".log").replace(
            "\\",
            "/",
        )
    )
    pf_solver_log_file = (
        ""
        if solver_log_dir is None
        else os.path.join(solver_log_dir, "pf_" + str(proc) + ".log").replace("\\", "/")
    )
    dcpf_solver_log_file = (
        ""
        if solver_log_dir is None
        else os.path.join(solver_log_dir, "dcpf_" + str(proc) + ".log").replace(
            "\\",
            "/",
        )
    )
    dcopf_solver_log_file = (
        ""
        if solver_log_dir is None
        else os.path.join(solver_log_dir, "dcopf_" + str(proc) + ".log").replace(
            "\\",
            "/",
        )
    )

    if print_level is None:
        print_level = 0 if solver_log_dir is None else 5
    else:
        print_level = print_level

    try:
        # If dc_max_iter not provided, use 1000
        dc_iter = 1000 if dc_max_iter is None else dc_max_iter
        # Base imports and logging config in Julia
        jl.seval("""
        using PowerModels
        using Ipopt
        using Memento
        Memento.config!("not_set")
        """)

        # ----- AC-OPF core -----
        jl.seval(
            """
        function _run_opf_core(case_file)
            start_time = time()  # record start (seconds since epoch)
            result = solve_ac_opf(
                case_file,
                optimizer_with_attributes(
                    Ipopt.Optimizer,
                    "tol" => 1e-6,
                    "print_level" => {},
                    "max_iter" => {},
                ),
            )
            end_time = time()  # record end time
            result["runtime"] = end_time - start_time  # elapsed seconds
            result["solution"]["pf"] = false
            return result
        end
        """.format(print_level, max_iter),
        )

        # run_opf: either alias to core (no logging) or wrap with redirection (logging)
        if opf_solver_log_file == "":
            jl.seval("""
            const run_opf = _run_opf_core
            """)
        else:
            jl.seval(
                """
            function run_opf(case_file)
                open("{}", "a") do io

                    redirect_stdout(io) do
                        redirect_stderr(io) do
                            return _run_opf_core(case_file)
                        end
                    end
                end
            end
            """.format(opf_solver_log_file),
            )

        # ----- DC-OPF core -----
        jl.seval(
            """
        function _run_dcopf_core(case_file)
            start_time = time()  # record start (seconds since epoch)
            result = solve_dc_opf(
                case_file,
                optimizer_with_attributes(
                    Ipopt.Optimizer,
                    "tol" => 1e-6,
                    "print_level" => {},
                    "max_iter" => {},
                ),
            )
            end_time = time()  # record end time
            result["runtime"] = end_time - start_time  # elapsed seconds
            result["solution"]["pf"] = false
            return result
        end
        """.format(print_level, dc_iter),
        )

        # run_dcopf: either alias to core (no logging) or wrap with redirection (logging)
        if dcopf_solver_log_file == "":
            jl.seval("""
            const run_dcopf = _run_dcopf_core
            """)
        else:
            jl.seval(
                """
            function run_dcopf(case_file)
                open("{}", "a") do io

                    redirect_stdout(io) do
                        redirect_stderr(io) do
                            return _run_dcopf_core(case_file)
                        end
                    end
                end
            end
            """.format(dcopf_solver_log_file),
            )

        # ----- Fast PF (direct computation) -----
        jl.seval("""
        function run_pf_fast(case_file)
            network = PowerModels.parse_file(case_file)
            result = compute_ac_pf(network)

            if result["termination_status"] == false
                return result
            end

            update_data!(network, result["solution"])
            flows = calc_branch_flow_ac(network)

            result["solution"]["branch"] = flows["branch"]
            result["solution"]["pf"] = true
            return result
        end
        """)

        # ----- Fast DC-PF (direct computation) -----
        jl.seval("""
        function run_dcpf_fast(case_file)
            network = PowerModels.parse_file(case_file)
            result = compute_dc_pf(network)

            if result["termination_status"] == false
                return result
            end

            update_data!(network, result["solution"])
            flows = calc_branch_flow_dc(network)

            result["solution"]["branch"] = flows["branch"]
            result["solution"]["pf"] = true
            return result
        end
        """)

        # ----- AC-PF core -----
        jl.seval(
            """
        function _run_pf_core(case_file)
            network = PowerModels.parse_file(case_file)
            result = solve_ac_pf(
                network,
                optimizer_with_attributes(
                    Ipopt.Optimizer,
                    "tol" => 1e-6,
                    "print_level" => {},
                    "max_iter" => {},
                ),
            )

            if string(result["termination_status"]) != "LOCALLY_SOLVED"
                return result
            end

            update_data!(network, result["solution"])
            flows = calc_branch_flow_ac(network)

            result["solution"]["branch"] = flows["branch"]
            result["solution"]["pf"] = true
            return result
        end
        """.format(print_level, max_iter),
        )

        # run_pf: either alias to core (no logging) or wrap with redirection (logging)
        if pf_solver_log_file == "":
            jl.seval("""
            const run_pf = _run_pf_core
            """)
        else:
            jl.seval(
                """
            function run_pf(case_file)
                open("{}", "a") do io

                    redirect_stdout(io) do
                        redirect_stderr(io) do
                            return _run_pf_core(case_file)
                        end
                    end
                end
            end
            """.format(pf_solver_log_file),
            )

        # ----- DC-PF core -----
        jl.seval(
            """
        function _run_dcpf_core(case_file)
            network = PowerModels.parse_file(case_file)
            result = solve_dc_pf(
                network,
                optimizer_with_attributes(
                    Ipopt.Optimizer,
                    "tol" => 1e-6,
                    "print_level" => {},
                    "max_iter" => {},
                ),
            )

            if string(result["termination_status"]) != "LOCALLY_SOLVED"
                return result
            end

            update_data!(network, result["solution"])
            flows = calc_branch_flow_dc(network)

            result["solution"]["branch"] = flows["branch"]
            result["solution"]["pf"] = true
            return result
        end
        """.format(print_level, dc_iter),
        )

        # run_dcpf: either alias to core (no logging) or wrap with redirection (logging)
        if dcpf_solver_log_file == "":
            jl.seval("""
            const run_dcpf = _run_dcpf_core
            """)
        else:
            jl.seval(
                """
            function run_dcpf(case_file)
                open("{}", "a") do io

                    redirect_stdout(io) do
                        redirect_stderr(io) do
                            return _run_dcpf_core(case_file)
                        end
                    end
                end
            end
            """.format(dcpf_solver_log_file),
            )

        # warm start all functions by running a dummy case
        dummy_case_file = str(
            resources.files("gridfm_datakit.process").joinpath("dummy.m"),
        )
        if print_level > 0 and solver_log_dir is None:
            print("\n ======= warm starting Julia interface =======\n", flush=True)
        if opf_solver_log_file:
            with open(opf_solver_log_file, "a") as f:
                f.write(" ======= warm starting Julia interface opf function =======\n")
        jl.run_opf(dummy_case_file)

        if dcopf_solver_log_file:
            with open(dcopf_solver_log_file, "a") as f:
                f.write(
                    " ======= warm starting Julia interface dcopf function =======\n",
                )
        jl.run_dcopf(dummy_case_file)

        # run_pf_fast has no log file
        jl.run_pf_fast(dummy_case_file)

        if pf_solver_log_file:
            with open(pf_solver_log_file, "a") as f:
                f.write(" ======= warm starting Julia interface pf function =======\n")
        jl.run_pf(dummy_case_file)

        if dcpf_solver_log_file:
            with open(dcpf_solver_log_file, "a") as f:
                f.write(
                    " ======= warm starting Julia interface dcpf function =======\n",
                )
        jl.run_dcpf(dummy_case_file)

        # run_dcpf_fast has no log file
        jl.run_dcpf_fast(dummy_case_file)

        if print_level > 0 and solver_log_dir is None:
            print(
                "\n ======= warm starting Julia interface completed =======\n",
                flush=True,
            )

    except Exception as e:
        raise RuntimeError("Error initializing Julia: {}".format(e))

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

    net.Pg_gen = pg
    net.Vm = vm

    return net


def apply_slack_single_gen(
    net: Network,
    pg_gen: np.ndarray,
    Pg_bus: np.ndarray,
    pf_dcpf: np.ndarray,
    pt_dcpf: np.ndarray,
) -> np.ndarray:
    """
    Put the entire slack-bus power imbalance on the first generator
    connected to the slack (reference) bus.

    Parameters
    ----------
    net : Network
    pg_gen : np.ndarray
        Generator outputs (current), aligned with net.gens[net.idx_gens_in_service, :].
    Pg_bus : np.ndarray
        Total generation per bus.
    pf_dcpf, pt_dcpf : np.ndarray
        Line flows (from, to) from the DC power flow.

    Returns
    -------
    np.ndarray
        Updated generator outputs, with the first slack-bus generator adjusted.
    """

    pd_slack = net.Pd[net.ref_bus_idx]
    pg_slack = Pg_bus[net.ref_bus_idx]

    # branches with slack as from/to bus
    branches_from = net.branches[net.idx_branches_in_service, F_BUS] == net.ref_bus_idx
    branches_to = net.branches[net.idx_branches_in_service, T_BUS] == net.ref_bus_idx

    sum_flows_from = pf_dcpf[branches_from].sum()
    sum_flows_to = pt_dcpf[branches_to].sum()

    # power balance at slack
    balance = pg_slack - pd_slack - (sum_flows_from + sum_flows_to)

    # find generators at slack bus
    slack_gen = np.where(net.gens[net.idx_gens_in_service, GEN_BUS] == net.ref_bus_idx)[
        0
    ]

    # copy current setpoints
    pg_gen_dc = pg_gen.copy()

    # assign entire balance to first generator at slack
    first_slack_gen = slack_gen[0]
    pg_gen_dc[first_slack_gen] -= balance

    return pg_gen_dc


def pf_post_processing(
    scenario_index: int,
    net: Network,
    res: Dict[str, Any],
    res_dc: Dict[str, Any],
    include_dc_res: bool,
) -> Dict[str, np.ndarray]:
    """Post-process solved network results into numpy arrays for CSV export.

    This function extracts power flow results and builds four arrays matching
    the column schemas defined in `gridfm_datakit.utils.column_names`:

    - Bus data with BUS_COLUMNS (+ DC_BUS_COLUMNS if include_dc_res=True)
    - Generator data with GEN_COLUMNS
    - Branch data with BRANCH_COLUMNS
    - Y-bus nonzero entries with [index1, index2, G, B]

    Args:
        net: The power network to process (must have solved power flow results).
        res: Power flow result dictionary containing solution data.
        include_dc_res: If True, include DC power flow voltage magnitude/angle (Vm_dc, Va_dc).

    Returns:
        Dictionary containing:
        - "bus": np.ndarray with bus-level features
        - "gen": np.ndarray with generator features
        - "branch": np.ndarray with branch features and admittances
        - "Y_bus": np.ndarray with nonzero Y-bus entries
    """

    # --- Edge (branch) info ---
    n_branches = net.branches.shape[0]
    n_cols = (
        len(BRANCH_COLUMNS) + len(DC_BRANCH_COLUMNS)
        if include_dc_res
        else len(BRANCH_COLUMNS)
    )
    X_branch = np.zeros((n_branches, n_cols))
    X_branch[:, 0] = scenario_index
    X_branch[:, 1] = list(range(n_branches))
    X_branch[:, 2] = np.real(net.branches[:, F_BUS])
    X_branch[:, 3] = np.real(net.branches[:, T_BUS])

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

    X_branch[net.idx_branches_in_service, 4] = np.array(
        [
            res["solution"]["branch"][str(i + 1)]["pf"] * net.baseMVA
            for i in net.idx_branches_in_service
        ],
    )
    X_branch[net.idx_branches_in_service, 5] = np.array(
        [
            res["solution"]["branch"][str(i + 1)]["qf"] * net.baseMVA
            for i in net.idx_branches_in_service
        ],
    )
    X_branch[net.idx_branches_in_service, 6] = np.array(
        [
            res["solution"]["branch"][str(i + 1)]["pt"] * net.baseMVA
            for i in net.idx_branches_in_service
        ],
    )
    X_branch[net.idx_branches_in_service, 7] = np.array(
        [
            res["solution"]["branch"][str(i + 1)]["qt"] * net.baseMVA
            for i in net.idx_branches_in_service
        ],
    )

    X_branch[:, 8] = net.branches[:, BR_R]
    X_branch[:, 9] = net.branches[:, BR_X]
    X_branch[:, 10] = net.branches[:, BR_B]

    # admittances
    Ytt, Yff, Yft, Ytf = branch_vectors(net.branches, net.branches.shape[0])
    X_branch[:, 11] = np.real(Yff)
    X_branch[:, 12] = np.imag(Yff)
    X_branch[:, 13] = np.real(Yft)
    X_branch[:, 14] = np.imag(Yft)
    X_branch[:, 15] = np.real(Ytf)
    X_branch[:, 16] = np.imag(Ytf)
    X_branch[:, 17] = np.real(Ytt)
    X_branch[:, 18] = np.imag(Ytt)

    X_branch[:, 19] = net.branches[:, TAP]
    # assign 1 to tap = 0
    X_branch[net.branches[:, TAP] == 0, 19] = 1

    X_branch[:, 20] = net.branches[:, SHIFT]
    X_branch[:, 21] = net.branches[:, ANGMIN]
    X_branch[:, 22] = net.branches[:, ANGMAX]
    X_branch[:, 23] = net.branches[:, RATE_A]
    X_branch[:, 24] = net.branches[:, BR_STATUS]

    if include_dc_res:
        if res_dc is not None:
            pf_dc = np.array(
                [
                    res_dc["solution"]["branch"][str(i + 1)]["pf"] * net.baseMVA
                    for i in net.idx_branches_in_service
                ],
            )
            pt_dc = np.array(
                [
                    res_dc["solution"]["branch"][str(i + 1)]["pt"] * net.baseMVA
                    for i in net.idx_branches_in_service
                ],
            )
            X_branch[net.idx_branches_in_service, 25] = pf_dc
            X_branch[net.idx_branches_in_service, 26] = pt_dc
        else:
            X_branch[net.idx_branches_in_service, 25] = np.nan
            X_branch[net.idx_branches_in_service, 26] = np.nan

    # --- Bus data ---
    n_buses = net.buses.shape[0]
    n_cols = (
        len(BUS_COLUMNS) + len(DC_BUS_COLUMNS) if include_dc_res else len(BUS_COLUMNS)
    )
    X_bus = np.zeros((n_buses, n_cols))

    # --- Loads ---
    X_bus[:, 0] = scenario_index
    X_bus[:, 1] = net.buses[:, BUS_I]  # bus
    X_bus[:, 2] = net.buses[:, PD]
    X_bus[:, 3] = net.buses[:, QD]

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

    if include_dc_res:
        if res_dc is not None:
            # check if "gen" key is in res_dc["solution"]
            if "gen" in res_dc["solution"]:
                pg_gen_dc = np.array(
                    [
                        res_dc["solution"]["gen"][str(i + 1)]["pg"] * net.baseMVA
                        for i in net.idx_gens_in_service
                    ],
                )
            else:
                pg_gen_dc = apply_slack_single_gen(net, pg_gen, Pg_bus, pf_dc, pt_dc)
            Pg_bus_dc = np.bincount(gen_bus, weights=pg_gen_dc, minlength=n_buses)
            assert np.all(Pg_bus_dc[net.buses[:, BUS_TYPE] == PQ] == 0)

    X_bus[:, 4] = Pg_bus
    X_bus[:, 5] = Qg_bus

    # Voltage
    assert set([int(k) for k in res["solution"]["bus"].keys()]) == set(
        net.reverse_bus_index_mapping.values(),
    ), "Buses in solution should match buses in network"

    X_bus[:, 6] = [
        res["solution"]["bus"][str(net.reverse_bus_index_mapping[i])]["vm"]
        for i in range(n_buses)
    ]
    va = np.rad2deg(
        [
            res["solution"]["bus"][str(net.reverse_bus_index_mapping[i])]["va"]
            for i in range(n_buses)
        ],
    )

    # convert to range [-180, 180]
    va = (va + 180) % 360 - 180
    X_bus[:, 7] = va

    # one-hot encoding of bus type
    assert np.all(np.isin(net.buses[:, BUS_TYPE], [PQ, PV, REF])), (
        "Bus type should be PQ, PV, or REF, no disconnected buses (4)"
    )
    X_bus[np.arange(n_buses), 8 + net.buses[:, BUS_TYPE].astype(int) - 1] = (
        1  # because type is 1, 2, 3, not 0, 1, 2
    )

    # base_kv, min_vm_pu, max_vm_pu
    X_bus[:, 11] = net.buses[:, BASE_KV]
    X_bus[:, 12] = net.buses[:, VMIN]
    X_bus[:, 13] = net.buses[:, VMAX]

    X_bus[:, 14] = net.buses[:, GS] / net.baseMVA
    X_bus[:, 15] = net.buses[:, BS] / net.baseMVA

    if include_dc_res:
        if res_dc is not None:
            va = np.rad2deg(
                [
                    res_dc["solution"]["bus"][str(net.reverse_bus_index_mapping[i])][
                        "va"
                    ]
                    for i in range(n_buses)
                ],
            )
            # convert to range [-180, 180]
            va = (va + 180) % 360 - 180
            X_bus[:, 16] = va
            X_bus[:, 17] = Pg_bus_dc
        else:
            X_bus[:, 16] = np.nan
            X_bus[:, 17] = np.nan

    # --- Generator data ---
    assert np.all(net.gencosts[:, NCOST] == 3), "NCOST should be 3"
    n_gens = net.gens.shape[0]
    n_cols = (
        len(GEN_COLUMNS) + len(DC_GEN_COLUMNS) if include_dc_res else len(GEN_COLUMNS)
    )

    X_gen = np.zeros((n_gens, n_cols))
    X_gen[:, 0] = scenario_index
    X_gen[:, 1] = list(range(n_gens))
    X_gen[:, 2] = net.gens[:, GEN_BUS]
    X_gen[net.idx_gens_in_service, 3] = pg_gen  # 0 if not in service
    X_gen[net.idx_gens_in_service, 4] = qg_gen  # 0 if not in service
    X_gen[:, 5] = net.gens[:, PMIN]
    X_gen[:, 6] = net.gens[:, PMAX]
    X_gen[:, 7] = net.gens[:, QMIN]
    X_gen[:, 8] = net.gens[:, QMAX]
    X_gen[:, 9] = net.gencosts[:, COST + 2]
    X_gen[:, 10] = net.gencosts[:, COST + 1]
    X_gen[:, 11] = net.gencosts[:, COST]
    X_gen[net.idx_gens_in_service, 12] = 1

    # slack gen (can be any generator connected to the ref node)
    slack_gen_idx = np.where(net.gens[:, GEN_BUS] == net.ref_bus_idx)[0]
    X_gen[slack_gen_idx, 13] = 1

    if include_dc_res:
        if res_dc is not None:
            X_gen[net.idx_gens_in_service, 14] = pg_gen_dc
        else:
            X_gen[net.idx_gens_in_service, 14] = np.nan

    # --- Y-bus ---
    Y_bus, Yf, Yt = makeYbus(net.baseMVA, net.buses, net.branches)

    i, j = np.nonzero(Y_bus)
    # note that Y_bus[i,j] can be != 0 even if a branch from i to j is not in service because there might be other branches connected to the same buses

    s = Y_bus[i, j]
    G = np.real(s)
    B = np.imag(s)

    edge_index = np.column_stack((i, j))
    edge_attr = np.stack((G, B)).T
    Y_bus = np.zeros(
        (edge_index.shape[0], edge_attr.shape[1] + edge_index.shape[1] + 1),
    )
    Y_bus[:, 0] = scenario_index
    Y_bus[:, 1:] = np.column_stack((edge_index, edge_attr))

    # ---- runtime data ----
    n_cols = (
        len(RUNTIME_COLUMNS) + len(DC_RUNTIME_COLUMNS)
        if include_dc_res
        else len(RUNTIME_COLUMNS)
    )
    X_runtime = np.zeros((1, n_cols))
    X_runtime[0, 0] = scenario_index
    X_runtime[0, 1] = res["solve_time"]
    if include_dc_res:
        if res_dc is not None:
            X_runtime[0, 2] = res_dc["solve_time"]
        else:
            X_runtime[0, 2] = np.nan
    return {
        "bus": X_bus,
        "gen": X_gen,
        "branch": X_branch,
        "Y_bus": Y_bus,
        "runtime": X_runtime,
    }


def process_scenario_pf_mode(
    net: Network,
    scenarios: np.ndarray,
    scenario_index: int,
    topology_generator: TopologyGenerator,
    generation_generator: GenerationGenerator,
    admittance_generator: AdmittanceGenerator,
    local_processed_data: List[np.ndarray],
    error_log_file: str,
    include_dc_res: bool,
    pf_fast: bool,
    dcpf_fast: bool,
    jl: Any,
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
        include_dc_res: Whether to include DC power flow results in output.
        pf_fast: Whether to use fast AC PF solver.
        dcpf_fast: Whether to use fast DC PF solver.
        jl: Julia interface object for running power flow calculations.

    Returns:
        Updated list of processed data (bus, gen, branch, Y_bus arrays)
    """
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
        res_dcpf = None
        if include_dc_res:
            try:
                res_dcpf = run_dcpf(perturbation, jl, fast=dcpf_fast)

            except Exception as e:
                with open(error_log_file, "a") as f:
                    f.write(
                        f"Caught an exception at scenario {scenario_index} when solving dcpf function: {e}\n",
                    )

        try:
            res = run_pf(perturbation, jl, fast=pf_fast)
        except Exception as e:
            with open(error_log_file, "a") as f:
                f.write(
                    f"Caught an exception at scenario {scenario_index} when solving in run_pf function: {e}\n",
                )
            continue

        # Append processed power flow data
        pf_data = pf_post_processing(
            scenario_index,
            perturbation,
            res,
            res_dcpf,
            include_dc_res,
        )
        local_processed_data.append(
            (
                pf_data["bus"],
                pf_data["gen"],
                pf_data["branch"],
                pf_data["Y_bus"],
                pf_data["runtime"],
            ),
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
    include_dc_res: bool,
    pf_fast: bool,
    dcpf_fast: bool,
    solver_log_dir: str,
    max_iter: int,
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
        include_dc_res: Whether to include DC power flow results in output.

    Returns:
        Tuple containing:
            - Exception object (None if successful)
            - Traceback string (None if successful)
            - List of processed data tuples (bus, gen, branch, Y_bus arrays)
    """

    try:
        jl = init_julia(max_iter, solver_log_dir)
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
                    include_dc_res,
                    jl,
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
                    include_dc_res,
                    pf_fast,
                    dcpf_fast,
                    jl,
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
    include_dc_res: bool,
    jl: Any,
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
        include_dc_res: Whether to include DC power flow results in output.

    Returns:
        Updated list of processed data (bus, gen, branch, Y_bus arrays)
    """

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
        res_dcopf = None
        if include_dc_res:
            try:
                res_dcopf = run_dcopf(perturbation, jl)
            except Exception as e:
                with open(error_log_file, "a") as f:
                    f.write(
                        f"Caught an exception at scenario {scenario_index} in run_dcopf function: {e}\n",
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
        pf_data = pf_post_processing(
            scenario_index,
            perturbation,
            res,
            res_dcopf,
            include_dc_res,
        )
        local_processed_data.append(
            (
                pf_data["bus"],
                pf_data["gen"],
                pf_data["branch"],
                pf_data["Y_bus"],
                pf_data["runtime"],
            ),
        )
    return local_processed_data
