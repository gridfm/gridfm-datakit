"""
Dynamic simulation processing pipeline.

Provides the distributed outer loop (process_dynamic_simulations) and the
per-scenario processing unit (process_single_dynamic_simulation), mirroring
the multiprocessing pattern from generate.py but adapted for dynamic
simulations.

Worker processes are isolated: each initialises its own Julia instance and
its own copy of the pypowsybl network at chunk start, then reuses them for
all scenarios in that chunk.
"""


from __future__ import annotations

import multiprocessing
import traceback
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from gridfm_datakit.network import Network
from gridfm_datakit.dynamic.dynawo.simulate import run_dynawo_simulation, compute_balanced_static_state_dynawo
from gridfm_datakit.dynamic import DynamicResults
from gridfm_datakit.dynamic.dynawo import get_dynawo_simulation_parameters, generate_dynawo_mappings
from gridfm_datakit.process.process_network import init_julia
from gridfm_datakit.powsybl import load_net
from gridfm_datakit.utils.random_seed import custom_seed
from gridfm_datakit.utils.param_handler import NestedNamespace


# ---------------------------------------------------------------------------
# Public: distributed outer loop
# ---------------------------------------------------------------------------


def process_dynamic_simulations(
    network_path: str,
    scenarios: np.ndarray,
    dynamic_inputs: Any,
    dynamic_solver: str,
    config: NestedNamespace,
    error_log_file: str,
    seed: int,
) -> List[Dict[str, Any]]:
    # TODO: update doc
    """Distributed outer loop for dynamic simulation data generation.

    Splits scenarios into chunks and dispatches each chunk to a worker
    process. Each worker initialises a Julia instance and a local copy of the
    pypowsybl network once, then reuses them for all scenarios in the chunk.

    Args
    ----
    pp_net :
        Base pypowsybl network (read-only; workers deep-copy before mutating).
    gfm_net : Network
        Base gridfm network.
    scenarios : np.ndarray
        Load scenarios array, shape (n_loads, n_scenarios, 2).
    p2g_maps :
        MappingP2G built from the base network.
    dynamic_mappings :
        DynawoMappings (or future solver-equivalent).
    solver_params :
        pypowsybl.dynamic.Parameters.
    dynamic_solver : str
        Solver name ("dynawo" or future alternatives).
    args :
        Full NestedNamespace config.
    error_log_file : str
        Path to error log.
    seed : int
        Global seed — deterministically derived per-chunk seeds are computed
        from this value.

    Returns
    -------
    list of dict
        One dict per successfully processed scenario, each with keys:
        ``"pf_data"``, ``"dynamic_results"``, ``"scenario_index"``.
    """
    n_scenarios = config.load.scenarios
    large_chunk_size = config.settings.large_chunk_size
    num_processes = config.settings.num_processes
    max_iter = config.settings.max_iter
    solver_log_dir = getattr(config.settings, "solver_log_dir", None)

    large_chunks = np.array_split(
        range(n_scenarios),
        int(np.ceil(n_scenarios / large_chunk_size)),
    )

    all_results: List[Dict[str, Any]] = []

    for large_chunk_index, large_chunk in enumerate(large_chunks):
        chunk_size = len(large_chunk)
        scenario_chunks = np.array_split(
            large_chunk,
            min(num_processes, chunk_size),
        )

        tasks = [
            (
                chunk[0],
                chunk[-1] + 1,
                scenarios,
                network_path,
                dynamic_inputs,
                dynamic_solver,
                error_log_file,
                max_iter,
                solver_log_dir,
                seed,
                config,
            )
            for chunk in scenario_chunks
            if len(chunk) > 0
        ]

        _mp_ctx = multiprocessing.get_context("spawn")
        with _mp_ctx.Pool(processes=num_processes) as pool:
            results = pool.map(_process_dynamic_chunk, tasks)

        for chunk_results in results:
            if isinstance(chunk_results, Exception):
                print(f"Error in dynamic chunk: {chunk_results}")
            else:
                all_results.extend(chunk_results)

    return all_results

# ---------------------------------------------------------------------------
# Public: chunk worker
# ---------------------------------------------------------------------------


def _process_dynamic_chunk(args: Tuple) -> Union[List[Dict[str, Any]], List[Exception]]:
    """Worker function processing one chunk of scenarios.

    Initialises a Julia instance and a local pypowsybl network copy once,
    then iterates over all scenarios in the chunk.

    This is executed in a separate process (spawned), so it must be
    self-contained and importable.
    """
    (
        start_idx,
        end_idx,
        scenarios,
        network_path,
        dynamic_inputs,
        dynamic_solver,
        error_log_file,
        max_iter,
        solver_log_dir,
        seed,
        config
    ) = args

    if dynamic_solver=='dynawo':
        try:
            # Initialise Julia once per worker — avoids repeated JIT compilation
            julia = init_julia(max_iter, solver_log_dir)

            # TODO: discuss and delete this comment
            # Reloading powsybl network for each worker.
            # to_powsybl cannot work since gfm_network representation erases the original IDs
            # thus the dynamic model mappings would bug
            net = load_net(network_path)
            dynamic_mappings = generate_dynawo_mappings(dynamic_inputs)
            dynamic_solver_params = get_dynawo_simulation_parameters(config)
            ##
            # Build per-worker pypowsybl network copy
            # import gridfm_datakit.powsybl as powsybl
            ## pp_net cannot cross process boundaries; we reconstruct from gfm_net
            # converted = powsybl.to_powsybl(gfm_net)
            # pp_net_worker = converted.pp_net
            ####

            chunk_results: List[Dict[str, Any]] = []

            with custom_seed(seed * 20000 + start_idx):
                for scenario_index in range(start_idx, end_idx):
                    try:
                        result = process_single_dynamic_simulation(
                            pp_net=net.pp_net,
                            gfm_net=net.gfm_net,
                            scenarios=scenarios,
                            scenario_index=scenario_index,
                            p2g_maps=net.mapping_p2g,
                            dynamic_mappings=dynamic_mappings,
                            dynamic_solver_params=dynamic_solver_params,
                            dynamic_solver=dynamic_solver,
                            julia=julia,
                        )
                        chunk_results.append(result)
                    except Exception as e:
                        tb = traceback.format_exc()
                        with open(error_log_file, "a") as f:
                            f.write(
                                f"[dynamic] scenario {scenario_index} failed: {e}\n{tb}\n",
                            )

            return chunk_results

        except Exception as e:
            return [e]  # surfaced in the parent process
    raise NotImplementedError(
        f"Dynamic solver {dynamic_solver!r} is not implemented. "
        "Supported solvers: 'dynawo'.",
    )


# ---------------------------------------------------------------------------
# Public: single scenario
# ---------------------------------------------------------------------------    

def process_single_dynamic_simulation(
        pp_net: Any,
        gfm_net: Network,
        scenarios: np.ndarray,
        scenario_index: int,
        p2g_maps,
        dynamic_mappings: Any,
        dynamic_solver_params: Any,
        dynamic_solver: str,
        julia: Any,
        ) -> Dict[str, Any]:
    

    # Apply load scenario
    gfm_net.Pd = scenarios[:, scenario_index, 0]
    gfm_net.Qd = scenarios[:, scenario_index, 1]

    # TODO: add pertubations to scenario here, then work on a new clone for each perturbation
    base_variant_id = pp_net.get_working_variant_id()
    variant_id = f"scenario_{scenario_index}"
    pp_net.clone_variant(base_variant_id, variant_id)
    pp_net.set_working_variant(variant_id)

    # Step 1+2: balanced static state
    # (with PowSyBl, the clone is directly altered by the computation of the balanced static state)
    pp_net, pf_data = _compute_balanced_static_state(
        pp_net=pp_net,
        gfm_net=gfm_net,
        julia=julia,
        dynamic_solver=dynamic_solver,
        p2g_maps=p2g_maps,
        scenario_index=scenario_index,
    )

    # Step 3: dynamic simulation
    dyn_results = _run_dynamic_simulation(
        pp_net,
        dynamic_mappings,
        dynamic_solver_params,
        dynamic_solver,
    )

    # Step 4: combine
    combined = _combine_pf_and_dyn_res(pf_data, dyn_results)
    combined["scenario_index"] = scenario_index

    # Step 5: clean up
    pp_net.set_working_variant(base_variant_id)
    pp_net.remove_variant(variant_id)

    return combined

def _compute_balanced_static_state(
        pp_net,
        gfm_net: Network,
        julia,
        dynamic_solver,
        p2g_maps,
        scenario_index,
        ):
    """Wrapper around solver-specific balanced-state computation.

    Currently routes to ``compute_balanced_static_state_dynawo``.
    ``dynamic_solver`` is the extension point for future backends.
    """
    if dynamic_solver == 'dynawo':
        return compute_balanced_static_state_dynawo(pp_net=pp_net, 
                                                    gfm_net=gfm_net,
                                                    julia=julia,
                                                    p2g_maps=p2g_maps,
                                                    scenario_index=scenario_index,
                                                    )
    raise NotImplementedError(
        f"Dynamic solver {dynamic_solver!r} is not implemented. "
        "Supported solvers: 'dynawo'.",
    )

def _run_dynamic_simulation(
        network,
        dynamic_mappings,
        solver_parameters,
        dynamic_solver,
) -> DynamicResults:
    """Wrapper around solver-specific dynamic simulation run.

    Currently routes to ``run_dynawo_simulation``.
    """
    if dynamic_solver == 'dynawo':
        return run_dynawo_simulation(network, dynamic_mappings, solver_parameters)
    raise NotImplementedError(
        f"Dynamic solver {dynamic_solver!r} is not implemented. "
        "Supported solvers: 'dynawo'.",
    )

def _combine_pf_and_dyn_res(pf_data: Dict[str, Any], 
                            dynamic_results: DynamicResults,
) -> Dict[str, Any]:
   """Merge static PF snapshot with dynamic time-series into a single output dict.

    The alignment schema between the per-bus/branch/gen PF snapshot and the
    per-variable dynamic time-series is TBD (see architecture §12, Open
    Question #2). This function currently packages both outputs side-by-side
    so they can be saved independently by ``_save_generated_data``.

    Parameters
    ----------
    pf_data : dict
        Keys: ``"bus"``, ``"gen"``, ``"branch"``, ``"Y_bus"``, ``"runtime"``.
    dynamic_results : DynamicResults

    Returns
    -------
    dict
        Keys: ``"pf_data"`` and ``"dynamic_results"``.
    """
   return {
       "pf_data": pf_data,
       "dynamic_results": dynamic_results,
   }
