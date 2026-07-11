"""Main data generation module for gridfm_datakit."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from tqdm import tqdm

import gridfm_datakit.powsybl as powsybl
from gridfm_datakit.network import (
    Network,
    get_pglib_file_path,
    load_net_from_file,
    load_net_from_pglib,
)
from gridfm_datakit.perturbations.load_perturbation import (
    load_scenarios_to_df,
    plot_load_scenarios_combined,
)
from gridfm_datakit.process.process_network import (
    _initialize_scenario_worker,
    _process_scenario_worker,
    init_julia,
    process_scenario_opf_mode,
    process_scenario_pf_mode,
)
from gridfm_datakit.save import (
    save_node_edge_data,
)
from gridfm_datakit.utils.param_handler import (
    NestedNamespace,
    get_load_scenario_generator,
    initialize_admittance_generator,
    initialize_generation_generator,
    initialize_topology_generator,
)
from gridfm_datakit.utils.random_seed import custom_seed
from gridfm_datakit.utils.utils import Tee, write_parquet, write_ram_usage_distributed


def _split_range(start: int, stop: int, n_parts: int) -> List[Tuple[int, int]]:
    """Split a range into balanced contiguous bounds without materializing indices."""
    length = stop - start
    if length < 0:
        raise ValueError("stop must be greater than or equal to start")
    if length == 0:
        return []
    if n_parts <= 0:
        raise ValueError("n_parts must be positive")

    n_parts = min(n_parts, length)
    quotient, remainder = divmod(length, n_parts)
    bounds = []
    cursor = start
    for part_index in range(n_parts):
        part_size = quotient + (part_index < remainder)
        bounds.append((cursor, cursor + part_size))
        cursor += part_size
    return bounds


def _setup_environment(
    config: Union[str, Dict[str, Any], NestedNamespace],
) -> Tuple[NestedNamespace, str, Dict[str, str], int]:
    """Setup the environment for data generation.

    Args:
        config: Configuration can be provided in three ways:
            1. Path to a YAML config file (str)
            2. Configuration dictionary (Dict)
            3. NestedNamespace object (NestedNamespace)

    Returns:
        Tuple of (args, base_path, file_paths, seed)
    """
    # Load config from file if a path is provided
    if isinstance(config, str):
        with open(config, "r") as f:
            config = yaml.safe_load(f)

    # Convert dict to NestedNamespace if needed
    if isinstance(config, dict):
        args = NestedNamespace(**config)
    else:
        args = config

        # Set global seed if provided, otherwise generate a unique seed for this generation
    if (
        hasattr(args.settings, "seed")
        and args.settings.seed is not None
        and args.settings.seed != ""
    ):
        seed = args.settings.seed
        print(f"Global random seed set to: {seed}")

    else:
        # Generate a unique seed for non-reproducible but independent scenarios
        # This ensures scenarios are i.i.d. within a run, but different across runs
        import secrets

        seed = secrets.randbelow(50_000)
        # chunk_seed = seed * 20000 + start_idx + 1 < 2^31 - 1
        # seed < (2,147,483,647 - n_scenarios) / 20,000 ~= 100_000 so taking 50_000 to be safe
        print(f"No seed provided. Using seed={seed}")

    # Resolve and validate the network reader.
    #
    # reader controls HOW the network file is parsed (independent of pf_solver).
    # source controls WHERE to get the file: 'pglib' (download) or 'file' (local).
    reader = getattr(args.network, "reader", "native")
    if reader not in ("native", "powsybl"):
        raise ValueError(
            f"network.reader must be 'native' or 'powsybl', got {reader!r}",
        )
    args.network.reader = reader

    # Resolve and validate the PF solver setting.
    #
    # pf_solver controls which engine is used to solve the power flow equations
    # in PF mode.  It is completely independent of network.source/reader.
    #
    # OPF is always solved by PowerModels (Julia) regardless of this setting.
    # In OPF mode the value is read and stored on args but is never consulted
    # during execution — it is kept here purely for consistency and logging.
    pf_solver = getattr(args.settings, "pf_solver", "powermodel")
    if pf_solver not in ("powermodel", "powsybl"):
        raise ValueError(
            f"settings.pf_solver must be 'powermodel' or 'powsybl', got {pf_solver!r}",
        )
    args.settings.pf_solver = pf_solver

    opf_formulation = getattr(args.settings, "opf_formulation", "polar")
    if opf_formulation not in ("polar", "rectangular"):
        raise ValueError(
            "settings.opf_formulation must be 'polar' or 'rectangular', "
            f"got {opf_formulation!r}",
        )
    args.settings.opf_formulation = opf_formulation

    # Setup output directory
    base_path = os.path.join(args.settings.data_dir, args.network.name, "raw")
    if os.path.exists(base_path) and args.settings.overwrite:
        shutil.rmtree(base_path)
    os.makedirs(base_path, exist_ok=True)

    # Setup solver logs directory under data_dir/solver_log
    solver_log_dir = (
        os.path.join(base_path, "solver_log")
        if args.settings.enable_solver_logs
        else None
    )
    os.makedirs(solver_log_dir, exist_ok=True) if solver_log_dir is not None else None

    # Setup file paths
    file_paths = {
        "tqdm_log": os.path.join(base_path, "tqdm.log"),
        "error_log": os.path.join(base_path, "error.log"),
        "args_log": os.path.join(base_path, "args.log"),
        "solver_log_dir": solver_log_dir,
        "bus_data": os.path.join(base_path, "bus_data.parquet"),
        "branch_data": os.path.join(base_path, "branch_data.parquet"),
        "gen_data": os.path.join(base_path, "gen_data.parquet"),
        "y_bus_data": os.path.join(base_path, "y_bus_data.parquet"),
        "runtime_data": os.path.join(base_path, "runtime_data.parquet"),
        "scenarios": os.path.join(
            base_path,
            f"scenarios_{args.load.generator}.parquet",
        ),
        "scenarios_plot": os.path.join(
            base_path,
            f"scenarios_{args.load.generator}.html",
        ),
        "scenarios_log": os.path.join(
            base_path,
            f"scenarios_{args.load.generator}.log",
        ),
    }

    # Initialize logs
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for log_file in [
        file_paths["tqdm_log"],
        file_paths["error_log"],
        file_paths["scenarios_log"],
        file_paths["args_log"],
    ]:
        with open(log_file, "a") as f:
            f.write(f"\nNew generation started at {timestamp}\n")
            if log_file == file_paths["args_log"]:
                yaml.safe_dump(args.to_dict(), f)

    return args, base_path, file_paths, seed


def _prepare_network_and_scenarios(
    args: NestedNamespace,
    file_paths: Dict[str, str],
    seed: int,
) -> Tuple[Network, np.ndarray, Dict[str, Any]]:
    """Prepare the network and generate load scenarios.

    Args:
        args: Configuration object
        file_paths: Dictionary of file paths
        seed: Global random seed for reproducibility.

    Returns:
        Tuple of (network, scenarios)
    """
    meta = {}
    reader = args.network.reader  # already validated in _setup_environment

    if args.network.source == "pglib":
        if reader == "powsybl":
            network_path = get_pglib_file_path(args.network.name)
            loaded_net = powsybl.load_net(network_path)
            meta["pp_net"] = loaded_net.pp_net
            meta["network_path"] = network_path
            meta["mapping_p2g"] = loaded_net.mapping_p2g
            net = loaded_net.gfm_net
        else:
            net = load_net_from_pglib(args.network.name)
    elif args.network.source == "file":
        if reader == "powsybl":
            network_path = (
                args.network.file
                if getattr(args.network, "file", None)
                else os.path.join(args.network.network_dir, args.network.name) + ".m"
            )
            loaded_net = powsybl.load_net(network_path)
            meta["pp_net"] = loaded_net.pp_net
            meta["network_path"] = network_path
            meta["mapping_p2g"] = loaded_net.mapping_p2g
            net = loaded_net.gfm_net
        else:
            net = load_net_from_file(
                os.path.join(args.network.network_dir, args.network.name) + ".m",
            )
    else:
        raise ValueError(
            f"network.source must be 'pglib' or 'file', got {args.network.source!r}",
        )

    # Generate load scenarios
    load_scenario_generator = get_load_scenario_generator(args.load)
    scenarios = load_scenario_generator(
        net,
        args.load.scenarios,
        file_paths["scenarios_log"],
        max_iter=args.settings.max_iter,
        seed=seed,
    )
    scenarios_df = load_scenarios_to_df(scenarios)
    write_parquet(scenarios_df, file_paths["scenarios"])
    if net.buses.shape[0] <= 100:
        plot_load_scenarios_combined(scenarios_df, file_paths["scenarios_plot"])
    else:
        print("Skipping plot of scenarios for large networks (number of buses > 100)")

    return net, scenarios, meta


def _save_generated_data(
    net: Network,
    processed_data: List,
    file_paths: Dict[str, str],
    base_path: str,
    args: NestedNamespace,
) -> None:
    """Save the generated data to files.

    Args:
        net: Network object
        processed_data: List of processed data arrays
        file_paths: Dictionary of file paths
        base_path: Base output directory
        args: Configuration object
    """
    if len(processed_data) > 0:
        save_node_edge_data(
            net,
            file_paths["bus_data"],
            file_paths["branch_data"],
            file_paths["gen_data"],
            file_paths["y_bus_data"],
            file_paths["runtime_data"],
            processed_data,
            include_dc_res=args.settings.include_dc_res,
        )


def generate_power_flow_data(
    config: Union[str, Dict[str, Any], NestedNamespace],
) -> Dict[str, str]:
    """Generate power flow data based on the provided configuration using sequential processing.

    Args:
        config: Configuration can be provided in three ways:
            1. Path to a YAML config file (str)
            2. Configuration dictionary (Dict)
            3. NestedNamespace object (NestedNamespace)
            The config must include settings, network, load, and perturbation configurations.

    Returns:
        Dictionary with paths to generated artifacts:
        {
            'tqdm_log': progress log file,
            'error_log': error log file,
            'args_log': configuration dump file,
            'bus_data': bus-level features CSV (BUS_COLUMNS),
            'branch_data': branch-level features CSV (BRANCH_COLUMNS),
            'gen_data': generator features CSV (GEN_COLUMNS),
            'y_bus_data': Y-bus nonzero entries CSV,
            'scenarios': load scenarios Parquet,
            'scenarios_plot': load scenarios plot HTML,
            'scenarios_log': load scenario generation log
        }

    Note:
        The function creates output files under {settings.data_dir}/{network.name}/raw/:

        - tqdm.log: Progress tracking
        - error.log: Error messages
        - args.log: Configuration parameters (YAML dump)
        - bus_data.parquet: Bus-level features for each scenario
        - branch_data.parquet: Branch-level features for each scenario
        - gen_data.parquet: Generator features for each scenario
        - y_bus_data.parquet: Nonzero Y-bus entries for each scenario
        - scenarios_{generator}.parquet: Load scenarios (per-element time series)
        - scenarios_{generator}.html: Load scenario plots
        - scenarios_{generator}.log: Load scenario generation notes
    """

    # Setup environment
    args, base_path, file_paths, seed = _setup_environment(config)

    # Prepare network and scenarios
    net, scenarios, meta = _prepare_network_and_scenarios(args, file_paths, seed)

    # Initialize topology generator
    topology_generator = initialize_topology_generator(args.topology_perturbation, net)

    # Initialize generation generator
    generation_generator = initialize_generation_generator(
        args.generation_perturbation,
        net,
    )

    # Initialize admittance generator
    admittance_generator = initialize_admittance_generator(
        args.admittance_perturbation,
        net,
    )

    jl = init_julia(
        args.settings.max_iter,
        file_paths["solver_log_dir"],
        opf_formulation=args.settings.opf_formulation,
    )

    processed_data = []
    # Flush results to disk periodically so memory stays bounded, mirroring
    # the distributed path. Configs without large_chunk_size keep the old
    # save-once-at-the-end behavior.
    flush_every = getattr(args.settings, "large_chunk_size", None)

    # Process scenarios sequentially with deterministic seed
    # Use custom_seed to control randomness for reproducibility
    with custom_seed(seed + 1):
        with open(file_paths["tqdm_log"], "a") as f:
            with tqdm(
                total=args.load.scenarios,
                desc="Processing scenarios",
                file=Tee(sys.stdout, f),
                miniters=5,
            ) as pbar:
                for scenario_index in range(args.load.scenarios):
                    # Process the scenario
                    if args.settings.mode == "opf":
                        processed_data = process_scenario_opf_mode(
                            net,
                            scenarios,
                            scenario_index,
                            topology_generator,
                            generation_generator,
                            admittance_generator,
                            processed_data,
                            file_paths["error_log"],
                            args.settings.include_dc_res,
                            jl,
                        )
                    elif args.settings.mode == "pf":
                        processed_data = process_scenario_pf_mode(
                            net,
                            scenarios,
                            scenario_index,
                            topology_generator,
                            generation_generator,
                            admittance_generator,
                            processed_data,
                            file_paths["error_log"],
                            args.settings.include_dc_res,
                            args.settings.pf_fast,
                            args.settings.dcpf_fast,
                            jl,
                            args.settings.pf_solver,
                            meta=meta,
                        )
                    else:
                        raise ValueError("Invalid mode!")

                    pbar.update(1)

                    if flush_every and (scenario_index + 1) % flush_every == 0:
                        _save_generated_data(
                            net,
                            processed_data,
                            file_paths,
                            base_path,
                            args,
                        )
                        processed_data = []

    # Save final data
    _save_generated_data(
        net,
        processed_data,
        file_paths,
        base_path,
        args,
    )

    return file_paths


def generate_power_flow_data_distributed(
    config: Union[str, Dict[str, Any], NestedNamespace],
) -> Dict[str, str]:
    """Generate power flow data based on the provided configuration using distributed processing.

    Args:
        config: Configuration can be provided in three ways:
            1. Path to a YAML config file (str)
            2. Configuration dictionary (Dict)
            3. NestedNamespace object (NestedNamespace)
            The config must include settings, network, load, and perturbation configurations.

    Returns:
        Dictionary with paths to generated artifacts (same as generate_power_flow_data)

    Note:
        The function creates output files under {settings.data_dir}/{network.name}/raw/:

        - tqdm.log: Progress tracking
        - error.log: Error messages
        - args.log: Configuration parameters (YAML dump)
        - bus_data.parquet: Bus-level features for each scenario
        - branch_data.parquet: Branch-level features for each scenario
        - gen_data.parquet: Generator features for each scenario
        - y_bus_data.parquet: Nonzero Y-bus entries for each scenario
        - scenarios_{generator}.parquet: Load scenarios (per-element time series)
        - scenarios_{generator}.html: Load scenario plots
        - scenarios_{generator}.log: Load scenario generation notes
    """
    # Setup environment
    args, base_path, file_paths, seed = _setup_environment(config)

    # check if mode is valid
    if args.settings.mode not in ["opf", "pf"]:
        raise ValueError("Invalid mode!")

    scenario_count = args.load.scenarios
    large_chunk_size = args.settings.large_chunk_size
    configured_processes = args.settings.num_processes
    if scenario_count <= 0:
        raise ValueError("load.scenarios must be positive")
    if large_chunk_size <= 0:
        raise ValueError("settings.large_chunk_size must be positive")
    if configured_processes <= 0:
        raise ValueError("settings.num_processes must be positive")

    # Prepare network and scenarios
    net, scenarios, meta = _prepare_network_and_scenarios(args, file_paths, seed)

    # Initialize topology generator
    topology_generator = initialize_topology_generator(args.topology_perturbation, net)

    # Initialize generation generator
    generation_generator = initialize_generation_generator(
        args.generation_perturbation,
        net,
    )

    # Initialize admittance generator
    admittance_generator = initialize_admittance_generator(
        args.admittance_perturbation,
        net,
    )

    # Match np.array_split's balanced chunk boundaries without first allocating
    # an int64 array containing every scenario index.
    n_large_chunks = (scenario_count + large_chunk_size - 1) // large_chunk_size
    large_chunks = _split_range(0, scenario_count, n_large_chunks)
    max_large_chunk_size = max(stop - start for start, stop in large_chunks)
    worker_count = min(configured_processes, max_large_chunk_size)

    # pp_net wraps JVM state and cannot cross process boundaries. Spawned
    # workers reload it once from network_path; the in-process path can reuse it.
    worker_meta = (
        meta
        if worker_count == 1
        else {key: value for key, value in meta.items() if key != "pp_net"}
    )
    worker_initargs = (
        args.settings.mode,
        net,
        topology_generator,
        generation_generator,
        admittance_generator,
        file_paths["error_log"],
        args.settings.include_dc_res,
        args.settings.pf_fast,
        args.settings.dcpf_fast,
        file_paths["solver_log_dir"],
        args.settings.max_iter,
        seed,
        args.settings.pf_solver,
        worker_meta,
        args.settings.opf_formulation,
    )

    with open(file_paths["tqdm_log"], "a") as f:
        with tqdm(
            total=scenario_count,
            desc="Processing scenarios",
            file=Tee(sys.stdout, f),
            miniters=5,
        ) as pbar:

            def process_large_chunks(
                executor: Optional[ProcessPoolExecutor] = None,
            ) -> None:
                for large_start, large_stop in large_chunks:
                    write_ram_usage_distributed(f)
                    chunk_size = large_stop - large_start
                    scenario_chunks = _split_range(
                        large_start,
                        large_stop,
                        min(worker_count, chunk_size),
                    )
                    results_by_start = {}

                    if executor is None:
                        for start, stop in scenario_chunks:
                            result = _process_scenario_worker(
                                (start, scenarios[:, start:stop, :]),
                            )
                            results_by_start[start] = result
                    else:
                        futures = {}
                        for start, stop in scenario_chunks:
                            # A contiguous slice is serialized roughly twice as
                            # fast as a strided view and contains only this task's
                            # scenarios, not the tensor for the entire run.
                            scenario_slice = np.ascontiguousarray(
                                scenarios[:, start:stop, :],
                            )
                            future = executor.submit(
                                _process_scenario_worker,
                                (start, scenario_slice),
                            )
                            futures[future] = (start, stop)

                        for future in as_completed(futures):
                            start, stop = futures[future]
                            try:
                                results_by_start[start] = future.result()
                            except Exception as error:
                                raise RuntimeError(
                                    f"Scenario worker failed for [{start}, {stop})",
                                ) from error

                    processed_data = []
                    for start in sorted(results_by_start):
                        (
                            returned_start,
                            processed_count,
                            error,
                            traceback_text,
                            local_processed_data,
                        ) = results_by_start[start]
                        if returned_start != start:
                            raise RuntimeError(
                                f"Scenario worker returned chunk {returned_start}, expected {start}",
                            )
                        if error is not None:
                            raise RuntimeError(
                                f"Scenario chunk starting at {start} failed: {error}\n"
                                f"{traceback_text}",
                            ) from error
                        if local_processed_data is None:
                            raise RuntimeError(
                                f"Scenario chunk starting at {start} returned no data",
                            )
                        pbar.update(processed_count)
                        processed_data.extend(local_processed_data)

                    # Save processed data
                    _save_generated_data(
                        net,
                        processed_data,
                        file_paths,
                        base_path,
                        args,
                    )

            if worker_count == 1:
                # Avoid spawning a process, serializing state, and starting a
                # second Python interpreter when no parallelism was requested.
                _initialize_scenario_worker(*worker_initargs)
                process_large_chunks()
            else:
                mp_context = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=mp_context,
                    initializer=_initialize_scenario_worker,
                    initargs=worker_initargs,
                ) as executor:
                    process_large_chunks(executor)

    return file_paths
