"""
Power system network processing and scenario generation.

This module provides functionality for processing power system networks,
running power flow calculations, and generating perturbed scenarios
for data generation purposes.
"""

import os
import time
import traceback
from importlib import resources
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import gridfm_datakit.powsybl as powsybl
from gridfm_datakit.network import Network, branch_vectors, makeYbus
from gridfm_datakit.perturbations.admittance_perturbation import AdmittanceGenerator
from gridfm_datakit.perturbations.generator_perturbation import GenerationGenerator
from gridfm_datakit.perturbations.topology_perturbation import TopologyGenerator
from gridfm_datakit.process.solvers import run_dcopf, run_dcpf, run_opf, run_pf
from gridfm_datakit.utils.column_names import (
    BRANCH_COLUMNS,
    BUS_COLUMNS,
    DC_BRANCH_COLUMNS,
    DC_BUS_COLUMNS,
    DC_GEN_COLUMNS,
    DC_RUNTIME_COLUMNS,
    GEN_COLUMNS,
    RUNTIME_COLUMNS,
)
from gridfm_datakit.utils.idx_brch import (
    ANGMAX,
    ANGMIN,
    BR_B,
    BR_R,
    BR_STATUS,
    BR_X,
    F_BUS,
    RATE_A,
    SHIFT,
    T_BUS,
    TAP,
)
from gridfm_datakit.utils.idx_bus import (
    BASE_KV,
    BS,
    BUS_I,
    BUS_TYPE,
    GS,
    PD,
    PQ,
    PV,
    QD,
    REF,
    VMAX,
    VMIN,
)
from gridfm_datakit.utils.idx_cost import COST, NCOST
from gridfm_datakit.utils.idx_gen import GEN_BUS, PMAX, PMIN, QMAX, QMIN
from gridfm_datakit.utils.random_seed import custom_seed
from gridfm_datakit.process.solver_output import (
    SolverOutputConfig,
    build_router,
    set_active_router,
)


def init_julia(
    max_iter: int,
    solver_log_dir: str = None,
    dc_max_iter: Optional[int] = None,
    print_level: Optional[int] = None,
    output: Optional[SolverOutputConfig] = None,
    opf_formulation: str = "polar",
) -> Any:
    """Initialize Julia interface with PowerModels.jl.

    Sets up Julia environment and defines AC OPF/PF/DCPF entrypoints.

    Solver output is governed by a single :class:`SolverOutputConfig`. Pass
    ``output`` directly, or rely on the legacy ``solver_log_dir`` /
    ``print_level`` arguments, which are folded into an equivalent config.

    Args:
        max_iter: Maximum number of iterations for AC OPF solver.
        solver_log_dir: Legacy. Enables solver logging to files under this
            directory when ``output`` is None.
        dc_max_iter: Maximum number of iterations for DC OPF solver (default 1000).
        print_level: Legacy. Explicit Ipopt print level override.
        output: Solver-output config; takes precedence over the legacy args.
        opf_formulation: AC OPF coordinate formulation. ``"polar"`` preserves
            the historical formulation; ``"rectangular"`` uses ACR coordinates
            with starts from the current network state and is often faster.

    Returns:
        Julia interface object for running power flow calculations.

    Raises:
        RuntimeError: If Julia initialization fails.
    """
    if opf_formulation not in ("polar", "rectangular"):
        raise ValueError(
            "opf_formulation must be 'polar' or 'rectangular', "
            f"got {opf_formulation!r}",
        )

    if output is None:
        output = SolverOutputConfig.from_settings(
            log_dir=solver_log_dir,
            enable_solver_logs=solver_log_dir is not None,
        )
    # An explicit legacy print_level still wins over the verbosity-derived one.
    ipopt_print_level = output.ipopt_print_level if print_level is None else print_level
    # Ensure spawned workers use the same Julia project so package resolution is consistent.
    julia_project = None
    try:
        from juliapkg.state import STATE

        julia_project = STATE.get("project")
        if julia_project:
            os.environ.setdefault("JULIA_PROJECT", julia_project)
    except Exception:
        julia_project = None

    from juliacall import Main as jl

    # Importing juliacall resolves every packaged juliapkg.json against
    # JULIA_PROJECT before exposing Main. Running Pkg.instantiate again here is
    # redundant and serializes concurrent workers on the project lock.

    # Route every sub-system's output through one process-wide policy. The
    # solver entrypoints below are plain aliases to their cores; capture happens
    # in Python around each call (see process.solvers), not baked into Julia.
    router = build_router(output)
    set_active_router(router)

    print_level = ipopt_print_level

    try:
        # If dc_max_iter not provided, use 1000
        dc_iter = 1000 if dc_max_iter is None else dc_max_iter
        # Memento.config! sets the root logger; PowerModels.silence() is what
        # actually quiets PowerModels' own Info/Warn (see SolverOutputConfig).
        memento_level = output.memento_level
        silence_pm = "PowerModels.silence()" if output.silence_powermodels else ""
        # Base imports and logging config in Julia
        try:
            jl.seval(f"""
            using PowerModels
            using Ipopt
            using Memento
            Memento.config!("{memento_level}")
            {silence_pm}
            """)
        except Exception as e:
            msg = str(e)
            missing_pm = "Package PowerModels not found" in msg
            missing_ipopt = "Package Ipopt not found" in msg
            missing_memento = "Package Memento not found" in msg
            if missing_pm or missing_ipopt or missing_memento:
                jl.seval(f"""
                using Pkg
                Pkg.add("Ipopt")
                Pkg.add("PowerModels")
                Pkg.add("Memento")
                using PowerModels
                using Ipopt
                using Memento
                Memento.config!("{memento_level}")
                {silence_pm}
                """)
            else:
                raise

        # ----- AC-OPF cores -----
        # ACR is opt-in: it is generally faster on the benchmarked large cases,
        # but AC OPF is nonconvex and changing coordinates can change which local
        # optimum Ipopt reaches.  Keep the historical ACP path as the default.
        jl.seval(f'global _GFM_OPF_FORMULATION = "{opf_formulation}"')
        jl.seval(
            f"""
        function _gfm_set_acr_starts!(data)
            for (_, bus) in data["bus"]
                vm = bus["vm"]
                va = bus["va"]
                bus["vr_start"] = vm * cos(va)
                bus["vi_start"] = vm * sin(va)
            end
            for (_, gen) in data["gen"]
                gen["pg_start"] = gen["pg"]
                gen["qg_start"] = gen["qg"]
            end
            return data
        end

        function _gfm_set_acp_starts!(data)
            for (_, bus) in data["bus"]
                bus["vm_start"] = bus["vm"]
                bus["va_start"] = bus["va"]
            end
            for (_, gen) in data["gen"]
                gen["pg_start"] = gen["pg"]
                gen["qg_start"] = gen["qg"]
            end
            return data
        end

        function _gfm_clear_acr_starts!(data)
            for (_, bus) in data["bus"]
                delete!(bus, "vr_start")
                delete!(bus, "vi_start")
            end
            for (_, gen) in data["gen"]
                delete!(gen, "pg_start")
                delete!(gen, "qg_start")
            end
            return data
        end

        function _gfm_clear_acp_starts!(data)
            for (_, bus) in data["bus"]
                delete!(bus, "vm_start")
                delete!(bus, "va_start")
            end
            for (_, gen) in data["gen"]
                delete!(gen, "pg_start")
                delete!(gen, "qg_start")
            end
            return data
        end

        function _run_opf_polar_core(case_or_data)
            start_time = time()
            result = solve_ac_opf(
                case_or_data,
                optimizer_with_attributes(
                    Ipopt.Optimizer,
                    "tol" => 1e-6,
                    "print_level" => {print_level},
                    "max_iter" => {max_iter},
                ),
            )
            result["runtime"] = time() - start_time
            result["solution"]["pf"] = false
            return result
        end

        function _run_opf_rectangular_core(case_or_data)
            start_time = time()
            data = case_or_data isa AbstractString ?
                PowerModels.parse_file(case_or_data) : case_or_data
            _gfm_set_acr_starts!(data)
            result = nothing
            try
                result = solve_opf(
                    data,
                    ACRPowerModel,
                    optimizer_with_attributes(
                        Ipopt.Optimizer,
                        "tol" => 1e-6,
                        "print_level" => {print_level},
                        "max_iter" => {max_iter},
                    );
                    solution_processors=[sol_data_model!],
                )
            finally
                _gfm_clear_acr_starts!(data)
            end
            result["runtime"] = time() - start_time
            result["solution"]["pf"] = false
            return result
        end

        function _run_opf_core(case_or_data)
            if _GFM_OPF_FORMULATION == "rectangular"
                return _run_opf_rectangular_core(case_or_data)
            end
            return _run_opf_polar_core(case_or_data)
        end
        """,
        )

        # Output routing is handled in Python (fd-level capture around the call),
        # so the entrypoint is a plain alias to its core.
        jl.seval("const run_opf = _run_opf_core")

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

        jl.seval("const run_dcopf = _run_dcopf_core")

        # ----- Fast PF (direct computation) -----
        # *_data variants take a PowerModels data dict (built by _gfm_state);
        # the file entrypoints wrap them for warmups and direct file solves.
        # The _data variants mutate their argument (update_data!). The next
        # _gfm_state call fully resets the worker-local buffer before reuse.
        jl.seval("""
        function run_pf_fast_data(network)
            _gfm_set_acp_starts!(network)
            result = nothing
            try
                result = compute_ac_pf(network)
            finally
                _gfm_clear_acp_starts!(network)
            end

            if result["termination_status"] == false
                return result
            end

            update_data!(network, result["solution"])
            flows = calc_branch_flow_ac(network)

            result["solution"]["branch"] = flows["branch"]
            result["solution"]["pf"] = true
            return result
        end
        run_pf_fast(case_file) = run_pf_fast_data(PowerModels.parse_file(case_file))
        """)

        # ----- Fast DC-PF (direct computation) -----
        jl.seval("""
        function run_dcpf_fast_data(network)
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
        run_dcpf_fast(case_file) = run_dcpf_fast_data(PowerModels.parse_file(case_file))
        """)

        # ----- AC-PF core -----
        jl.seval(
            """
        function run_pf_data(network)
            _gfm_set_acp_starts!(network)
            result = nothing
            try
                result = solve_ac_pf(
                    network,
                    optimizer_with_attributes(
                        Ipopt.Optimizer,
                        "tol" => 1e-6,
                        "print_level" => {},
                        "max_iter" => {},
                    ),
                )
            finally
                _gfm_clear_acp_starts!(network)
            end

            if string(result["termination_status"]) != "LOCALLY_SOLVED"
                return result
            end

            update_data!(network, result["solution"])
            flows = calc_branch_flow_ac(network)

            result["solution"]["branch"] = flows["branch"]
            result["solution"]["pf"] = true
            return result
        end
        _run_pf_core(case_file) = run_pf_data(PowerModels.parse_file(case_file))
        """.format(print_level, max_iter),
        )

        jl.seval("const run_pf = _run_pf_core")

        # ----- DC-PF core -----
        jl.seval(
            """
        function run_dcpf_data(network)
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
        _run_dcpf_core(case_file) = run_dcpf_data(PowerModels.parse_file(case_file))
        """.format(print_level, dc_iter),
        )

        jl.seval("const run_dcpf = _run_dcpf_core")

        # ----- In-memory data path -----
        # Parse the MATPOWER case once per process (_gfm_init_base), then per
        # solve push only the fields the pipeline mutates (_gfm_state). The
        # transforms mirror PowerModels.parse_file bit-for-bit: make_per_unit!
        # (÷baseMVA, deg2rad on va), _rescale_cost_model! (cost[k]·mva^(n-k))
        # followed by _simplify_cost_terms! (leading zero coefficients
        # trimmed). Every other step of correct_network_data! only reads or
        # mutates fields the pipeline never changes between solves, so the
        # base parse covers them (verified in tests/test_pm_data_path.py).
        # A zero-pd load component is materialized for every bus so any load
        # scenario can be applied without re-parsing.
        jl.seval("""
        function _gfm_init_base(case_file)
            data = PowerModels.parse_file(case_file)
            bus_load = Dict{Int,String}()
            for (k, l) in data["load"]
                bus_load[l["load_bus"]] = k
            end
            next = isempty(data["load"]) ? 1 : maximum(parse(Int, k) for k in keys(data["load"])) + 1
            for (_, b) in data["bus"]
                bi = b["index"]
                if !haskey(bus_load, bi)
                    data["load"][string(next)] = Dict{String,Any}(
                        "source_id" => Any["bus", bi], "load_bus" => bi,
                        "status" => 1, "pd" => 0.0, "qd" => 0.0, "index" => next)
                    bus_load[bi] = string(next)
                    next += 1
                end
            end
            global _GFM_BASE = data
            global _GFM_WORK = deepcopy(data)
            global _GFM_BUS_LOAD = bus_load
            return nothing
        end

        function _gfm_state(bus_ids, bus_type, pd, qd, vm, va_deg,
                            pg, qg, vg, gen_status, cost,
                            br_status, br_r, br_x, br_b)
            # One worker executes solves serially, so reset and reuse a single
            # work dictionary instead of deep-copying the full parsed network
            # for every PF/OPF call.
            d = _GFM_WORK
            mva = d["baseMVA"]
            for r in eachindex(bus_ids)
                bi = Int(bus_ids[r])
                b = d["bus"][string(bi)]
                b["bus_type"] = Int(bus_type[r])
                b["vm"] = Float64(vm[r])
                b["va"] = deg2rad(Float64(va_deg[r]))
                l = d["load"][_GFM_BUS_LOAD[bi]]
                l["pd"] = Float64(pd[r]) / mva
                l["qd"] = Float64(qd[r]) / mva
            end
            ncost = size(cost, 2)
            for i in axes(cost, 1)
                g = d["gen"][string(i)]
                g["pg"] = Float64(pg[i]) / mva
                g["qg"] = Float64(qg[i]) / mva
                g["vg"] = Float64(vg[i])
                g["gen_status"] = Int(gen_status[i])
                c = [Float64(cost[i, k]) * Float64(mva)^(ncost - k) for k in 1:ncost]
                while !isempty(c) && c[1] == 0.0
                    popfirst!(c)
                end
                g["cost"] = c
                g["ncost"] = length(c)
            end
            for i in eachindex(br_status)
                br = d["branch"][string(i)]
                br["br_status"] = Int(br_status[i])
                br["br_r"] = Float64(br_r[i])
                br["br_x"] = Float64(br_x[i])
                br["b_fr"] = Float64(br_b[i]) / 2
                br["b_to"] = Float64(br_b[i]) / 2
            end
            return d
        end

        function _gfm_pack(result, branch_ids, gen_ids, bus_ids)
            sol = result["solution"]
            B = fill(NaN, length(branch_ids), 4)
            if haskey(sol, "branch")
                sb = sol["branch"]
                for (r, i) in enumerate(branch_ids)
                    br = sb[string(Int(i))]
                    B[r, 1] = br["pf"]
                    B[r, 2] = get(br, "qf", NaN)
                    B[r, 3] = br["pt"]
                    B[r, 4] = get(br, "qt", NaN)
                end
            end
            G = fill(NaN, length(gen_ids), 2)
            if haskey(sol, "gen")
                sg = sol["gen"]
                for (r, i) in enumerate(gen_ids)
                    g = sg[string(Int(i))]
                    G[r, 1] = g["pg"]
                    G[r, 2] = get(g, "qg", NaN)
                end
            end
            V = fill(NaN, length(bus_ids), 2)
            if haskey(sol, "bus")
                sbus = sol["bus"]
                for (r, i) in enumerate(bus_ids)
                    b = sbus[string(Int(i))]
                    V[r, 1] = get(b, "vm", NaN)
                    V[r, 2] = b["va"]
                end
            end
            return B, G, V
        end
        """)

        # Warm start all functions on a dummy case. Ipopt prints its one-time
        # license banner here at the C level; each warm-up runs inside its
        # channel's capture, so the banner lands wherever that channel points
        # (a file, or /dev/null when silent) instead of leaking to the console.
        dummy_case_file = str(
            resources.files("gridfm_datakit.process").joinpath("dummy.m"),
        )
        if output.to_console:
            print("\n ======= warm starting Julia interface =======\n", flush=True)

        # (channel name, Julia entrypoint). Fast paths share their channel with
        # the optimizer-based one; they warm up under the same capture.
        warmups = [
            ("opf", jl.run_opf),
            ("dcopf", jl.run_dcopf),
            ("pf", jl.run_pf_fast),
            ("pf", jl.run_pf),
            ("dcpf", jl.run_dcpf),
            ("dcpf", jl.run_dcpf_fast),
        ]
        for name, fn in warmups:
            channel = router.channel(name)
            channel.write_header(f" ======= warm starting {name} function =======")
            with channel.capture():
                fn(dummy_case_file)

        if output.to_console:
            print("\n ======= warm starting completed =======\n", flush=True)

    except Exception as e:
        raise RuntimeError("Error initializing Julia: {}".format(e))

    return jl


def _solution_arrays(
    res: Dict[str, Any],
    net: Network,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract solver solution values into dense arrays.

    Returns (all values exactly as the solver reported them — per-unit,
    radians; missing fields are NaN):

    - branch_flows: ``(n_branches_in_service, 4)`` — pf, qf, pt, qt
    - gen_pq: ``(n_gens_in_service, 2)`` — pg, qg (NaN-filled if the solution
      has no "gen" entry, e.g. fast DC-PF)
    - bus_vmva: ``(n_buses, 2)`` — vm, va, ordered by continuous bus index

    Julia results are packed in a single Julia-side pass (_gfm_pack) because
    each element access on a juliacall dict crosses the Python<->Julia
    boundary; plain-dict results (powsybl) use Python loops.
    """
    sol = res["solution"]
    ids_branch = net.idx_branches_in_service
    ids_gen = net.idx_gens_in_service
    n_buses = net.buses.shape[0]

    if isinstance(sol, dict):
        branch_flows = np.array(
            [
                (b["pf"], b.get("qf", np.nan), b["pt"], b.get("qt", np.nan))
                for b in (sol["branch"][str(i + 1)] for i in ids_branch)
            ],
        ).reshape(-1, 4)
        if "gen" in sol:
            gen_pq = np.array(
                [
                    (g["pg"], g.get("qg", np.nan))
                    for g in (sol["gen"][str(i + 1)] for i in ids_gen)
                ],
            ).reshape(-1, 2)
        else:
            gen_pq = np.full((len(ids_gen), 2), np.nan)
        bus_vmva = np.array(
            [
                (b.get("vm", np.nan), b["va"])
                for b in (
                    sol["bus"][str(net.reverse_bus_index_mapping[i])]
                    for i in range(n_buses)
                )
            ],
        ).reshape(-1, 2)
        return branch_flows, gen_pq, bus_vmva

    # Julia result: Julia is necessarily up already; importing Main here (not
    # at module level) keeps `import gridfm_datakit` from booting Julia.
    from juliacall import Main as jl

    solver_cache = getattr(net, "_solver_cache", None)
    if solver_cache is None:
        solver_cache = {}
        net._solver_cache = solver_cache
    rev = solver_cache.get("reverse_bus_ids")
    if rev is None:
        rev = np.empty(n_buses, dtype=np.int64)
        for new_idx, orig_idx in net.reverse_bus_index_mapping.items():
            rev[new_idx] = orig_idx
        solver_cache["reverse_bus_ids"] = rev
    branch_flows, gen_pq, bus_vmva = jl._gfm_pack(
        res,
        (ids_branch + 1).astype(np.int64),
        (ids_gen + 1).astype(np.int64),
        rev,
    )
    return np.asarray(branch_flows), np.asarray(gen_pq), np.asarray(bus_vmva)


def pf_preprocessing(net: Network, res: Dict[str, Any]) -> Network:
    """Set variables to the results of OPF.

    Copies the complete OPF operating point needed to initialize PF: active
    and reactive generator dispatch plus bus voltage magnitude and angle.

    Args:
        net: The power network to preprocess.
        res: OPF result dictionary containing solution data.

    Returns:
        Updated network with OPF results applied.
    """
    _, gen_pq, bus_vmva = _solution_arrays(res, net)

    net.Pg_gen = gen_pq[:, 0] * net.baseMVA
    net.Qg_gen = gen_pq[:, 1] * net.baseMVA
    net.Vm = bus_vmva[:, 0]
    # PowerModels reports radians; MATPOWER stores degrees.
    net.Va = np.rad2deg(bus_vmva[:, 1])

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
    X_branch[:, 1] = np.arange(n_branches)
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

    branch_flows, gen_pq, bus_vmva = _solution_arrays(res, net)
    if len(net.idx_branches_in_service) > 0:
        X_branch[net.idx_branches_in_service, 4:8] = branch_flows * net.baseMVA

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
            dc_flows, gen_pq_dc, bus_vmva_dc = _solution_arrays(res_dc, net)
            dc_flows = dc_flows * net.baseMVA
            pf_dc = dc_flows[:, 0]
            pt_dc = dc_flows[:, 2]
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
    gen_pq = gen_pq * net.baseMVA
    pg_gen = gen_pq[:, 0]
    qg_gen = gen_pq[:, 1]
    gen_bus = net.gens[net.idx_gens_in_service, GEN_BUS].astype(int)
    Pg_bus = np.bincount(gen_bus, weights=pg_gen, minlength=n_buses)
    Qg_bus = np.bincount(gen_bus, weights=qg_gen, minlength=n_buses)

    # Pg_bus / Qg_bus are indexed by 0-based bus INDEX; net.buses rows may be
    # in a different order (pypowsybl does not guarantee sorted bus export).
    # Build a bus-type array indexed by bus index so the mask aligns correctly.
    bus_type_by_idx = np.zeros(n_buses)
    bus_type_by_idx[net.buses[:, BUS_I].astype(int)] = net.buses[:, BUS_TYPE]

    assert np.all(Pg_bus[bus_type_by_idx == PQ] == 0)
    assert np.all(Qg_bus[bus_type_by_idx == PQ] == 0)

    if include_dc_res:
        if res_dc is not None:
            # check if "gen" key is in res_dc["solution"]
            if "gen" in res_dc["solution"]:
                pg_gen_dc = gen_pq_dc[:, 0] * net.baseMVA
            else:
                pg_gen_dc = apply_slack_single_gen(net, pg_gen, Pg_bus, pf_dc, pt_dc)
            Pg_bus_dc = np.bincount(gen_bus, weights=pg_gen_dc, minlength=n_buses)
            assert np.all(Pg_bus_dc[bus_type_by_idx == PQ] == 0)

    # Reindex Pg/Qg from bus-index order to bus-row order for X_bus assignment.
    bus_row_idx = net.buses[:, BUS_I].astype(int)
    X_bus[:, 4] = Pg_bus[bus_row_idx]
    X_bus[:, 5] = Qg_bus[bus_row_idx]

    # Voltage. Extraction (_solution_arrays) raises on any missing expected
    # bus key, so together with this length check the key sets must match.
    assert len(res["solution"]["bus"]) == n_buses, (
        "Buses in solution should match buses in network"
    )

    X_bus[:, 6] = bus_vmva[:, 0]
    va = np.rad2deg(bus_vmva[:, 1])

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
            va = np.rad2deg(bus_vmva_dc[:, 1])
            # convert to range [-180, 180]
            va = (va + 180) % 360 - 180
            X_bus[:, 16] = va
            X_bus[:, 17] = Pg_bus_dc[bus_row_idx]
        else:
            X_bus[:, 16] = np.nan
            X_bus[:, 17] = np.nan

    # --- Generator data ---

    n_cost = net.gencosts[0, NCOST]
    assert np.all(net.gencosts[:, NCOST] == n_cost), (
        "NCOST should be the same for all generators"
    )
    n_gens = net.gens.shape[0]
    n_cols = (
        len(GEN_COLUMNS) + len(DC_GEN_COLUMNS) if include_dc_res else len(GEN_COLUMNS)
    )

    X_gen = np.zeros((n_gens, n_cols))
    X_gen[:, 0] = scenario_index
    X_gen[:, 1] = np.arange(n_gens)
    X_gen[:, 2] = net.gens[:, GEN_BUS]
    X_gen[net.idx_gens_in_service, 3] = pg_gen  # 0 if not in service
    X_gen[net.idx_gens_in_service, 4] = qg_gen  # 0 if not in service
    X_gen[:, 5] = net.gens[:, PMIN]
    X_gen[:, 6] = net.gens[:, PMAX]
    X_gen[:, 7] = net.gens[:, QMIN]
    X_gen[:, 8] = net.gens[:, QMAX]

    if n_cost == 3:  # order in .m file is c2, c1, c0
        X_gen[:, 9] = net.gencosts[:, COST + 2]
        X_gen[:, 10] = net.gencosts[:, COST + 1]
        X_gen[:, 11] = net.gencosts[:, COST]

    if n_cost == 2:  # order in .m file is c1, c0, and there is no cp2 cost
        X_gen[:, 9] = net.gencosts[:, COST + 1]
        X_gen[:, 10] = net.gencosts[:, COST]
        X_gen[:, 11] = 0  # no cp2 cost for linear cost function

    if n_cost == 1:  # order in .m file is c0, and there is no cp1 or cp2 cost
        X_gen[:, 9] = net.gencosts[:, COST]
        X_gen[:, 10] = 0  # no cp1 cost for constant cost function
        X_gen[:, 11] = 0  # no cp2 cost for constant cost function

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
    pf_solver: str = "powermodel",
    *,
    meta: Optional[Dict] = None,
    scenario_data_index: Optional[int] = None,
) -> List[np.ndarray]:
    """Processes a load scenario in PF mode.

    In PF mode, OPF is run first to get generator setpoints, then topology
    perturbations are applied. This can lead to constraint violations (overloads,
    voltage violations) since the setpoints are not re-optimized for the new topology.

    Parameters
    ----------
    net:
        The base power network (copied internally before mutation).
    scenarios:
        Array of load scenarios with shape ``(n_loads, n_scenarios, 2)``.
    scenario_index:
        Global index of the current scenario to process and write.
    topology_generator:
        Generator for topology perturbations (line/transformer outages).
    generation_generator:
        Generator for generation cost perturbations.
    admittance_generator:
        Generator for line admittance perturbations.
    local_processed_data:
        List to accumulate processed data tuples.
    error_log_file:
        Path to error log file for recording failures.
    include_dc_res:
        Whether to include DC power flow results in output.
    pf_fast:
        Whether to use the fast AC PF solver (``compute_ac_pf`` from
        PowerModels.jl).  Only consulted when ``pf_solver='powermodel'``.
    dcpf_fast:
        Whether to use the fast DC PF solver (``compute_dc_pf`` from
        PowerModels.jl).  Only consulted when ``pf_solver='powermodel'``.
    jl:
        Julia interface object.  Always required — even when
        ``pf_solver='powsybl'`` Julia is used for the OPF step that
        produces the generator set-points before topology perturbation.
    pf_solver:
        Which engine to use for the power flow solve after topology
        perturbation.  Must be ``'powermodel'`` (default) or
        ``'powsybl'``.  OPF is always solved by PowerModels regardless
        of this value.

    Keyword-only arguments (only required when ``pf_solver='powsybl'``)
    -------------------------------------------------------------------
    meta:
    Optional dictionary containing metadata for PowSyBl processing, with keys:
        - pp_net: the PowSyBl network.
        - mapping_p2g: dictionary mapping from PowSyBl to GFM.
    scenario_data_index:
        Optional local index into a sliced ``scenarios`` tensor. Defaults to
        ``scenario_index`` for callers that pass the full tensor.

    Returns
    -------
    List[np.ndarray]
        Updated ``local_processed_data`` list with one tuple
        ``(bus, gen, branch, Y_bus, runtime)`` appended per successfully
        solved perturbation.

    Note
    ----
    Random seed is controlled by the calling context
    (``process_scenario_chunk`` or ``generate_power_flow_data``).
    """
    net = net.copy_for_perturbation()

    # apply the load scenario to the network
    data_index = scenario_index if scenario_data_index is None else scenario_data_index
    net.Pd = scenarios[:, data_index, 0]
    net.Qd = scenarios[:, data_index, 1]

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

    net_pf = pf_preprocessing(net, res)

    # Generate perturbed topologies
    perturbations = topology_generator.generate(net_pf)

    if pf_solver == "powsybl":
        powsybl.check_powsybl_available()
        if meta is None or "pp_net" not in meta or "mapping_p2g" not in meta:
            raise ValueError("Network seems to not be initialized for PowSyBl solver")
        pp_net = meta["pp_net"]
        mapping_p2g = meta["mapping_p2g"]
        base_variant_id = pp_net.get_working_variant_id()
        lf_params = powsybl.get_default_lf_params()

    # to get PF points that can violate some OPF inequality constraints (to train PF solvers that can handle points outside of normal operating limits), we apply the topology perturbation after OPF.
    # The setpoints are then no longer adapted to the new topology, and might lead to e.g. abranch overload or a voltage magnitude violation once we drop an element.
    for pert_index, perturbation in enumerate(perturbations):
        if pf_solver == "powermodel":
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

        if pf_solver == "powsybl":
            variant_id = f"scenario_{scenario_index}_perturbation_{pert_index}"
            pp_net.clone_variant(base_variant_id, variant_id)
            pp_net.set_working_variant(variant_id)
            try:
                powsybl.update_powsybl(pp_net, perturbation, mapping_p2g)

                res_dcpf = None
                if include_dc_res:
                    try:
                        start_time = time.perf_counter()
                        dcpf_metadata = powsybl.pypowsybl.loadflow.run_dc(
                            pp_net,
                            lf_params,
                        )
                        end_time = time.perf_counter()
                        solve_time = end_time - start_time
                        res_dcpf = powsybl.get_pf_res(
                            pp_net,
                            solve_time,
                            dcpf_metadata,
                            mapping_p2g,
                        )

                    except Exception as e:
                        with open(error_log_file, "a") as f:
                            f.write(
                                f"Caught an exception at scenario {scenario_index} when solving dcpf function with PowSyBl solver: {e}\n",
                            )
                try:
                    start_time = time.perf_counter()
                    pf_metadata = powsybl.pypowsybl.loadflow.run_ac(pp_net, lf_params)
                    end_time = time.perf_counter()
                    solve_time = end_time - start_time
                    res = powsybl.get_pf_res(
                        pp_net,
                        solve_time,
                        pf_metadata,
                        mapping_p2g,
                    )
                except Exception as e:
                    with open(error_log_file, "a") as f:
                        f.write(
                            f"Caught an exception at scenario {scenario_index} when solving in run_pf function with PowSyBl solver: {e}\n",
                        )
                    continue
            finally:
                pp_net.set_working_variant(base_variant_id)
                pp_net.remove_variant(variant_id)

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


# Per-worker state. Large, invariant objects are installed once by the process
# initializer instead of being serialized again with every scenario chunk.
_worker_jl = None
_worker_context: Optional[Dict[str, Any]] = None


def _initialize_scenario_worker(
    mode: str,
    net: Network,
    topology_generator: TopologyGenerator,
    generation_generator: GenerationGenerator,
    admittance_generator: AdmittanceGenerator,
    error_log_path: str,
    include_dc_res: bool,
    pf_fast: bool,
    dcpf_fast: bool,
    solver_log_dir: Optional[str],
    max_iter: int,
    seed: int,
    pf_solver: str,
    meta: Optional[Dict],
    opf_formulation: str = "polar",
) -> None:
    """Install immutable run state once in a spawned scenario worker."""
    global _worker_context, _worker_jl
    _worker_jl = None
    _worker_context = {
        "mode": mode,
        "net": net,
        "topology_generator": topology_generator,
        "generation_generator": generation_generator,
        "admittance_generator": admittance_generator,
        "error_log_path": error_log_path,
        "include_dc_res": include_dc_res,
        "pf_fast": pf_fast,
        "dcpf_fast": dcpf_fast,
        "solver_log_dir": solver_log_dir,
        "max_iter": max_iter,
        "seed": seed,
        "pf_solver": pf_solver,
        "meta": meta,
        "opf_formulation": opf_formulation,
    }


def _process_scenario_worker(
    task: Tuple[int, np.ndarray],
) -> Tuple[
    int,
    int,
    Union[None, Exception],
    Union[None, str],
    Optional[List[np.ndarray]],
]:
    """Process one sliced scenario tensor using process-local run state."""
    if _worker_context is None:
        raise RuntimeError("Scenario worker has not been initialized")

    scenario_index_offset, scenarios = task
    scenario_count = scenarios.shape[1]
    error, traceback_text, processed_data = process_scenario_chunk(
        _worker_context["mode"],
        0,
        scenario_count,
        scenarios,
        _worker_context["net"],
        None,
        _worker_context["topology_generator"],
        _worker_context["generation_generator"],
        _worker_context["admittance_generator"],
        _worker_context["error_log_path"],
        _worker_context["include_dc_res"],
        _worker_context["pf_fast"],
        _worker_context["dcpf_fast"],
        _worker_context["solver_log_dir"],
        _worker_context["max_iter"],
        _worker_context["seed"],
        _worker_context["pf_solver"],
        _worker_context["meta"],
        scenario_index_offset=scenario_index_offset,
        opf_formulation=_worker_context["opf_formulation"],
    )
    return (
        scenario_index_offset,
        scenario_count,
        error,
        traceback_text,
        processed_data,
    )


def process_scenario_chunk(
    mode: str,
    start_idx: int,
    end_idx: int,
    scenarios: np.ndarray,
    net: Network,
    progress_queue: Optional[Any],
    topology_generator: TopologyGenerator,
    generation_generator: GenerationGenerator,
    admittance_generator: AdmittanceGenerator,
    error_log_path: str,
    include_dc_res: bool,
    pf_fast: bool,
    dcpf_fast: bool,
    solver_log_dir: str,
    max_iter: int,
    seed: int,
    pf_solver: str = "powermodel",
    meta: Optional[Dict] = None,
    *,
    scenario_index_offset: int = 0,
    opf_formulation: str = "polar",
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
        pf_fast: Whether to use fast AC PF solver.
        dcpf_fast: Whether to use fast DC PF solver.
        solver_log_dir: Directory for solver logs.
        max_iter: Maximum iterations for the solver.
        seed: Global random seed for reproducibility.
        pf_solver: PF solver to use in pf mode; either 'powermodel' or 'powsybl'.
            OPF is always solved by PowerModels regardless of this value.
        meta: metadata dict; when pf_solver='powsybl', must contain 'network_path'
            and 'mapping_p2g'. 'pp_net' is loaded fresh per worker from 'network_path'.
        scenario_index_offset: Global index added to local indices when ``scenarios``
            contains only the current chunk.

    Returns:
        Tuple containing:
            - Exception object (None if successful)
            - Traceback string (None if successful)
            - List of processed data tuples (bus, gen, branch, Y_bus arrays)
    """

    global _worker_jl
    completed_scenarios = 0
    try:
        if _worker_jl is None:
            _worker_jl = init_julia(
                max_iter,
                solver_log_dir,
                opf_formulation=opf_formulation,
            )
        jl = _worker_jl

        # In distributed (spawn) workers pp_net is not passed; reload it here.
        if (
            pf_solver == "powsybl"
            and meta
            and "network_path" in meta
            and "pp_net" not in meta
        ):
            import gridfm_datakit.powsybl as _powsybl

            loaded_net = _powsybl.load_net(meta["network_path"])
            meta["pp_net"] = loaded_net.pp_net

        local_processed_data = []

        global_start_idx = start_idx + scenario_index_offset

        # Use custom_seed to set seed based on the global start index for this chunk
        # This ensures each chunk gets a unique but deterministic seed
        # we multiply by 20_000 to ensure there is no collision with other runs where the seed would be close to each other
        # example (assuming we have chunks of length 1, hence an increment of 1 between start indices)
        # Run A: base seed = 42 → scenario seeds = 42, 43, 44, …, 10041 (for 10,000 scenarios)
        # Run B: base seed = 120 → scenario seeds = 120, 121, 122, …, 10119
        # These sets overlap on seeds 120..10041 (so 9,922 overlapping seeds).
        # we also add 1 in case the seed is 0, to not have collision witht he seed used for the load perturbations
        with custom_seed(seed * 20_000 + global_start_idx + 1):
            for scenario_data_index in range(start_idx, end_idx):
                scenario_index = scenario_data_index + scenario_index_offset
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
                        scenario_data_index=scenario_data_index,
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
                        pf_solver,
                        meta=meta,
                        scenario_data_index=scenario_data_index,
                    )
                else:
                    raise ValueError(f"Invalid mode: {mode!r}")

                completed_scenarios += 1
                if progress_queue is not None:
                    progress_queue.put(1)

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
        if progress_queue is not None:
            for _ in range(end_idx - start_idx - completed_scenarios):
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
    *,
    scenario_data_index: Optional[int] = None,
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
        jl: Julia interface object for running power flow calculations.
        scenario_data_index: Optional local index into a sliced ``scenarios``
            tensor. Defaults to ``scenario_index``.

    Returns:
        Updated list of processed data (bus, gen, branch, Y_bus arrays)

    Note:
        Random seed is controlled by the calling context (process_scenario_chunk).
    """

    # apply the load scenario to the network
    data_index = scenario_index if scenario_data_index is None else scenario_data_index
    net.Pd = scenarios[:, data_index, 0]
    net.Qd = scenarios[:, data_index, 1]

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
