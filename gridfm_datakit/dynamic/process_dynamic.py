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

import copy
import logging
import multiprocessing
import traceback
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from gridfm_datakit.network import Network
from gridfm_datakit.dynamic.dynawo.simulate import (
    run_dynawo_simulation,
    compute_balanced_static_state_dynawo,
)
from gridfm_datakit.dynamic import DynamicResults
from gridfm_datakit.dynamic.dynawo import (
    get_dynawo_loadflow_parameters,
    get_dynawo_simulation_parameters,
    generate_dynawo_mappings,
)
from gridfm_datakit.process.process_network import init_julia
from gridfm_datakit.powsybl import load_net
from gridfm_datakit.utils.random_seed import custom_seed
from gridfm_datakit.utils.param_handler import (
    NestedNamespace,
    initialize_admittance_generator,
    initialize_generation_generator,
    initialize_topology_generator,
)

# Shares the pipeline logger configured in generate_dynamic._configure_logging.
# These call sites run in the parent process, so the parent's handler applies.
logger = logging.getLogger("gridfm_datakit.dynamic")


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
    """Distributed outer loop for dynamic simulation data generation.

    Splits scenarios into chunks and dispatches each chunk to a worker
    process. Each worker initialises a Julia instance and a local copy of the
    pypowsybl network once, then reuses them for all scenarios in the chunk.

    Args
    ----
    network_path : str
        Path to the network.
    scenarios : np.ndarray
        Load scenarios array, shape (n_loads, n_scenarios, 2).
    dynamic_inputs: DynamicInputs
        Dynamics inputs.
    dynamic_solver : str
        Solver name ("dynawo" or future alternatives).
    config :
        Full NestedNamespace config.
    error_log_file : str
        Path to error log.
    seed : int
        Global seed — deterministically derived per-chunk seeds are computed
        from this value.

    Returns
    -------
    list of dict
        One dict per successfully processed (scenario, topology-perturbation)
        sample, each with keys ``"pf_data"``, ``"dynamic_results"``,
        ``"scenario_index"``, ``"perturbation_index"``.
    """
    n_scenarios = config.load.scenarios
    large_chunk_size = config.settings.large_chunk_size
    num_processes = config.settings.num_processes
    max_iter = config.settings.max_iter
    solver_log_dir = getattr(config.settings, "solver_log_dir", None)

    # Perturbation generators — built once here (main process) from the config and
    # the base network, then passed to workers (they are picklable, mirroring the
    # static pipeline). Absent config blocks default to the identity ("none")
    # generator, so a scenario expands to exactly one sample.
    base_net = load_net(network_path).gfm_net
    _none = NestedNamespace(type="none")
    topology_generator = initialize_topology_generator(
        getattr(config, "topology_perturbation", _none),
        base_net,
    )
    # generation_perturbation randomises generation *cost* and admittance_perturbation
    # perturbs branch admittances, both pre-OPF. They therefore vary the initial
    # operating point Dynawo starts from — which machines are dispatched and at what
    # loading — and so do influence the trajectory. What they do NOT vary is the
    # dynamic model set or the event sequence: those come from the CSV inputs and are
    # identical across every sample. Only topology_perturbation changes the network
    # the dynamic models are built on. Wired for parity with the static pipeline;
    # their effect on the dynamic outputs is untested.
    generation_generator = initialize_generation_generator(
        getattr(config, "generation_perturbation", _none),
        base_net,
    )
    admittance_generator = initialize_admittance_generator(
        getattr(config, "admittance_perturbation", _none),
        base_net,
    )

    large_chunks = np.array_split(
        range(n_scenarios),
        int(np.ceil(n_scenarios / large_chunk_size)),
    )

    all_results: List[Dict[str, Any]] = []

    logger.info(
        "Dynamic generation: %d scenarios in %d chunk(s), %d worker(s).",
        n_scenarios,
        len(large_chunks),
        num_processes,
    )

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
                topology_generator,
                generation_generator,
                admittance_generator,
            )
            for chunk in scenario_chunks
            if len(chunk) > 0
        ]

        _mp_ctx = multiprocessing.get_context("spawn")
        with _mp_ctx.Pool(processes=num_processes) as pool:
            results = pool.map(_process_dynamic_chunk, tasks)

        for chunk_results in results:
            # A worker that dies before its per-scenario loop returns [exception]
            # (see _process_dynamic_chunk). Detect both a bare exception and the
            # list-wrapped form so a failed chunk never poisons all_results with
            # non-result objects (which would later crash _save_generated_data).
            if isinstance(chunk_results, Exception):
                logger.error("Error in dynamic chunk: %s", chunk_results)
            else:
                for scenario_result in chunk_results:
                    if isinstance(scenario_result, Exception):
                        logger.error("Error in dynamic chunk: %s", scenario_result)
                    else:
                        all_results.append(scenario_result)

        logger.info(
            "Chunk %d/%d done (%d scenarios) — %d samples so far.",
            large_chunk_index + 1,
            len(large_chunks),
            chunk_size,
            len(all_results),
        )

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
        config,
        topology_generator,
        generation_generator,
        admittance_generator,
    ) = args

    if dynamic_solver == "dynawo":
        try:
            # Initialise Julia once per worker — avoids repeated JIT compilation
            julia = init_julia(max_iter, solver_log_dir)

            # Each worker reloads the network from file rather than receiving it:
            # to_powsybl() erases the original element IDs, which the dynamic model
            # mappings are keyed on.
            net = load_net(network_path)

            # Built per worker because none of these objects can be pickled.
            dynamic_mappings = generate_dynawo_mappings(dynamic_inputs)
            dynamic_solver_params = get_dynawo_simulation_parameters(config)
            lf_params = get_dynawo_loadflow_parameters(config)

            chunk_results: List[Dict[str, Any]] = []

            # Same derivation as the static pipeline (process_network.py): the
            # x20_000 spreads chunk seeds so runs with nearby base seeds don't
            # overlap, and the +1 keeps seed=0/start_idx=0 from reusing the very
            # seed the load scenarios were drawn with (which would correlate the
            # perturbations with the loads instead of making them independent).
            with custom_seed(seed * 20_000 + start_idx + 1):
                for scenario_index in range(start_idx, end_idx):
                    try:
                        # One scenario expands to one result per topology
                        # perturbation, hence extend rather than append.
                        results = process_single_dynamic_simulation(
                            pp_net=net.pp_net,
                            gfm_net=net.gfm_net,
                            scenarios=scenarios,
                            scenario_index=scenario_index,
                            p2g_maps=net.mapping_p2g,
                            dynamic_mappings=dynamic_mappings,
                            dynamic_solver_params=dynamic_solver_params,
                            dynamic_solver=dynamic_solver,
                            julia=julia,
                            lf_params=lf_params,
                            topology_generator=topology_generator,
                            generation_generator=generation_generator,
                            admittance_generator=admittance_generator,
                            error_log_file=error_log_file,
                        )
                        chunk_results.extend(results)
                    except Exception as e:
                        # A single scenario failing must not abort the rest of the
                        # chunk (and the guarded logging must not escalate either).
                        _log_error(
                            error_log_file,
                            f"[dynamic] scenario {scenario_index} failed: {e}\n"
                            f"{traceback.format_exc()}\n",
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
    topology_generator: Any = None,
    generation_generator: Any = None,
    admittance_generator: Any = None,
    error_log_file: str = None,
    lf_params: Any = None,
) -> List[Dict[str, Any]]:
    """Process one load scenario, expanded over topology perturbations.

    The load scenario is applied, then generation and admittance perturbations
    (before OPF). Each resulting topology perturbation is processed independently:
    the balanced initial state is computed on the perturbed network (so OPF adapts
    the set-points to the topology and Dynawo initialises from a converged
    operating point), then the dynamic simulation is run. One sample is produced
    per ``(scenario_index, perturbation_index)``.

    Absent generators default to identity, so a scenario yields exactly one
    sample — the pre-perturbation behaviour.

    Returns a list of result dicts (possibly empty if every perturbation failed).
    """
    # Work on a private copy so the shared per-worker network is not mutated.
    gfm_net = copy.deepcopy(gfm_net)
    gfm_net.Pd = scenarios[:, scenario_index, 0]
    gfm_net.Qd = scenarios[:, scenario_index, 1]

    # The three generators are handled differently because they have different
    # cardinality, not by accident — this mirrors the static pipeline exactly
    # (see process_network.process_scenario_*).
    #
    #   generation / admittance : 1 -> 1. They consume a stream of networks and
    #       yield one perturbed network per input, so we take next() once. They
    #       modify the operating point the OPF then optimises.
    #   topology                : 1 -> N. It yields n_topology_variants outages
    #       for a single network, so it is the only generator that expands one
    #       load scenario into several samples — hence the loop below, and hence
    #       perturbation_index existing at all.
    #
    # Generation + admittance perturbations, applied before OPF.
    net_iter = iter([gfm_net])
    if generation_generator is not None:
        net_iter = generation_generator.generate(net_iter)
    if admittance_generator is not None:
        net_iter = admittance_generator.generate(net_iter)
    perturbed_base = next(net_iter)

    # Topology perturbations expand the scenario into one or more samples.
    if topology_generator is not None:
        topologies = topology_generator.generate(perturbed_base)
    else:
        topologies = [perturbed_base]

    base_variant_id = pp_net.get_working_variant_id()
    results: List[Dict[str, Any]] = []
    for perturbation_index, perturbed_net in enumerate(topologies):
        variant_id = f"scenario_{scenario_index}_perturbation_{perturbation_index}"
        variant_created = False
        try:
            # Inside the try: a failing clone_variant would otherwise escape this
            # function entirely — bypassing the handler meant to keep one bad
            # perturbation from dropping the whole scenario — and leave the working
            # variant dangling for the rest of the worker's chunk.
            pp_net.clone_variant(base_variant_id, variant_id)
            variant_created = True
            pp_net.set_working_variant(variant_id)

            # Step 1+2: balanced static state on the perturbed network
            _, pf_data = _compute_balanced_static_state(
                pp_net=pp_net,
                gfm_net=perturbed_net,
                julia=julia,
                dynamic_solver=dynamic_solver,
                p2g_maps=p2g_maps,
                scenario_index=scenario_index,
                lf_params=lf_params,
            )

            # Step 3: dynamic simulation
            dyn_results = _run_dynamic_simulation(
                pp_net,
                dynamic_mappings,
                dynamic_solver_params,
                dynamic_solver,
            )

            # Step 4: combine + label with the (scenario, perturbation) key
            combined = _combine_pf_and_dyn_res(pf_data, dyn_results)
            combined["scenario_index"] = scenario_index
            combined["perturbation_index"] = perturbation_index
            results.append(combined)
        except Exception as e:
            # A single perturbation failing must not drop the whole scenario.
            _log_error(
                error_log_file,
                f"[dynamic] scenario {scenario_index} perturbation "
                f"{perturbation_index} failed: {e}\n{traceback.format_exc()}\n",
            )
        finally:
            # Step 5: clean up the per-perturbation variant (only if it was created)
            pp_net.set_working_variant(base_variant_id)
            if variant_created:
                pp_net.remove_variant(variant_id)

    return results


def _log_error(error_log_file, message: str) -> None:
    """Append an error message to the log file, falling back to stderr."""
    if error_log_file is None:
        print(message)
        return
    try:
        with open(error_log_file, "a") as f:
            f.write(message)
    except OSError:
        print(message)


def _compute_balanced_static_state(
    pp_net,
    gfm_net: Network,
    julia,
    dynamic_solver,
    p2g_maps,
    scenario_index,
    lf_params=None,
):
    """Wrapper around solver-specific balanced-state computation.

    Currently routes to ``compute_balanced_static_state_dynawo``.
    ``dynamic_solver`` is the extension point for future backends.
    """
    if dynamic_solver == "dynawo":
        return compute_balanced_static_state_dynawo(
            pp_net=pp_net,
            gfm_net=gfm_net,
            julia=julia,
            p2g_maps=p2g_maps,
            scenario_index=scenario_index,
            lf_params=lf_params,
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
    if dynamic_solver == "dynawo":
        return run_dynawo_simulation(network, dynamic_mappings, solver_parameters)
    raise NotImplementedError(
        f"Dynamic solver {dynamic_solver!r} is not implemented. "
        "Supported solvers: 'dynawo'.",
    )


def _combine_pf_and_dyn_res(
    pf_data: Dict[str, Any],
    dynamic_results: DynamicResults,
) -> Dict[str, Any]:
    """Merge static PF snapshot with dynamic time-series into a single output dict.

    The alignment schema between the per-bus/branch/gen PF snapshot and the
    per-variable dynamic time-series is TBD (see architecture §12, Open
    Question #2). This function currently packages both outputs side-by-side
    so they can be saved independently by ``_save_generated_data``.

    Args
    ----
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
