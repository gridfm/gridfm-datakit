"""Main data generation module for gridfm_datakit."""

import numpy as np
import os
from gridfm_datakit.save import (
    save_node_edge_data,
)
from gridfm_datakit.process.process_network import (
    process_scenario_opf_mode,
    process_scenario_pf_mode,
    process_scenario_chunk,
)
from gridfm_datakit.utils.param_handler import (
    NestedNamespace,
    get_load_scenario_generator,
    initialize_topology_generator,
    initialize_generation_generator,
    initialize_admittance_generator,
)
from gridfm_datakit.network import (
    load_net_from_file,
    load_net_from_pglib,
)
from gridfm_datakit.perturbations.load_perturbation import (
    load_scenarios_to_df,
    plot_load_scenarios_combined,
)
import gc
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, Manager
import shutil
from gridfm_datakit.utils.utils import write_ram_usage_distributed, Tee
import yaml
from typing import List, Tuple, Any, Dict, Union
import sys
from gridfm_datakit.network import Network
from gridfm_datakit.process.process_network import init_julia


def _setup_environment(
    config: Union[str, Dict[str, Any], NestedNamespace],
) -> Tuple[NestedNamespace, str, Dict[str, str]]:
    """Setup the environment for data generation.

    Args:
        config: Configuration can be provided in three ways:
            1. Path to a YAML config file (str)
            2. Configuration dictionary (Dict)
            3. NestedNamespace object (NestedNamespace)

    Returns:
        Tuple of (args, base_path, file_paths)
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

    return args, base_path, file_paths


def _prepare_network_and_scenarios(
    args: NestedNamespace,
    file_paths: Dict[str, str],
) -> Tuple[Network, np.ndarray]:
    """Prepare the network and generate load scenarios.

    Args:
        args: Configuration object
        file_paths: Dictionary of file paths

    Returns:
        Tuple of (network, scenarios)
    """
    if args.network.source == "pglib":
        net = load_net_from_pglib(args.network.name)
    elif args.network.source == "file":
        net = load_net_from_file(
            os.path.join(args.network.network_dir, args.network.name) + ".m",
        )
    else:
        raise ValueError("Invalid grid source!")

    # Generate load scenarios
    load_scenario_generator = get_load_scenario_generator(args.load)
    scenarios = load_scenario_generator(
        net,
        args.load.scenarios,
        file_paths["scenarios_log"],
        max_iter=args.settings.max_iter,
    )
    scenarios_df = load_scenarios_to_df(scenarios)
    scenarios_df.to_parquet(file_paths["scenarios"], index=False, engine="pyarrow")
    if net.buses.shape[0] <= 100:
        plot_load_scenarios_combined(scenarios_df, file_paths["scenarios_plot"])
    else:
        print("Skipping plot of scenarios for large networks (number of buses > 100)")

    return net, scenarios


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
    args, base_path, file_paths = _setup_environment(config)

    # Prepare network and scenarios
    net, scenarios = _prepare_network_and_scenarios(args, file_paths)

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

    jl = init_julia(args.settings.max_iter, file_paths["solver_log_dir"])

    processed_data = []

    # Process scenarios sequentially
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
                    )
                else:
                    raise ValueError("Invalid mode!")

                pbar.update(1)

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
    args, base_path, file_paths = _setup_environment(config)

    # check if mode is valid
    if args.settings.mode not in ["opf", "pf"]:
        raise ValueError("Invalid mode!")

    # Prepare network and scenarios
    net, scenarios = _prepare_network_and_scenarios(args, file_paths)

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

    # Setup multiprocessing
    manager = Manager()
    progress_queue = manager.Queue()

    # Process scenarios in chunks
    large_chunks = np.array_split(
        range(args.load.scenarios),
        np.ceil(args.load.scenarios / args.settings.large_chunk_size).astype(int),
    )

    with open(file_paths["tqdm_log"], "a") as f:
        with tqdm(
            total=args.load.scenarios,
            desc="Processing scenarios",
            file=Tee(sys.stdout, f),
            miniters=5,
        ) as pbar:
            for large_chunk_index, large_chunk in enumerate(large_chunks):
                write_ram_usage_distributed(f)
                chunk_size = len(large_chunk)
                scenario_chunks = np.array_split(
                    large_chunk,
                    min(args.settings.num_processes, chunk_size),
                )

                tasks = [
                    (
                        args.settings.mode,
                        chunk[0],
                        chunk[-1] + 1,
                        scenarios,
                        net,
                        progress_queue,
                        topology_generator,
                        generation_generator,
                        admittance_generator,
                        file_paths["error_log"],
                        args.settings.include_dc_res,
                        args.settings.pf_fast,
                        args.settings.dcpf_fast,
                        file_paths["solver_log_dir"],
                        args.settings.max_iter,
                    )
                    for chunk in scenario_chunks
                ]

                # Run parallel processing
                with Pool(processes=args.settings.num_processes) as pool:
                    results = [
                        pool.apply_async(process_scenario_chunk, task) for task in tasks
                    ]

                    # Update progress
                    completed = 0
                    while completed < chunk_size:
                        progress_queue.get()
                        pbar.update(1)
                        completed += 1

                    # Gather results
                    processed_data = []

                    for result in results:
                        (
                            e,
                            traceback,
                            local_processed_data,
                        ) = result.get()
                        if isinstance(e, Exception):
                            print(f"Error in process_scenario_chunk: {e}")
                            print(traceback)
                            sys.exit(e)
                        processed_data.extend(local_processed_data)

                    pool.close()
                    pool.join()

                # Save processed data
                _save_generated_data(
                    net,
                    processed_data,
                    file_paths,
                    base_path,
                    args,
                )

                del processed_data
                gc.collect()

    return file_paths
