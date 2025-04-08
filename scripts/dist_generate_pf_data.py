# Distributed version of the data generation script

import numpy as np
import os
import argparse
from GridDataGen.utils.io import *
from GridDataGen.utils.process_network import *
from GridDataGen.utils.config import *
from GridDataGen.utils.stats import *
from GridDataGen.utils.param_handler import *
from GridDataGen.utils.load import *
from pandapower.auxiliary import pandapowerNet
import gc
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, Manager
from multiprocessing import Queue
from GridDataGen.utils.topology_perturbation import initialize_generator
import psutil
import shutil
import yaml


def write_ram_usage(tqdm_log):
    process = psutil.Process(os.getpid())  # Parent process
    mem_usage = process.memory_info().rss / 1024**2  # Parent memory in MB

    # Sum memory usage of all child processes
    for child in process.children(recursive=True):
        mem_usage += child.memory_info().rss / 1024**2

    tqdm_log.write(f"Total RAM usage (Parent + Children): {mem_usage:.2f} MB\n")


def process_scenario_chunk(
    start_idx: int,
    end_idx: int,
    scenarios: np.ndarray,
    net: pandapowerNet,
    progress_queue: Queue,
    generator,
    no_stats: bool,
    error_log_path,
) -> list:
    """
    Create data for all scenarios in scenario indexed between start_idx and end_idx
    """
    if not no_stats:
        local_stats = Stats()
    local_csv_data = []
    local_adjacency_lists = []

    for scenario_index in range(start_idx, end_idx):

        local_csv_data, local_adjacency_lists, local_stats = process_scenario(
            net,
            scenarios,
            scenario_index,
            generator,
            no_stats,
            local_csv_data,
            local_adjacency_lists,
            local_stats,
            error_log_path,
        )

        progress_queue.put(1)  # update queue

    return local_csv_data, local_adjacency_lists, local_stats


def main(args):
    """
    Main routine that loads the network, splits scenarios into large chunks of 10,000,
    runs multiple processes in parallel, and saves data incrementally to avoid high memory usage.
    """
    base_path = os.path.join(args.settings.data_dir, args.network.name, "raw")
    if os.path.exists(base_path) and args.settings.overwrite:
        shutil.rmtree(base_path)
    os.makedirs(base_path, exist_ok=True)

    tqdm_log = open(os.path.join(base_path, "tqdm.log"), "a")
    error_log_path = os.path.join(base_path, "error.log")
    args_log_path = os.path.join(base_path, "args.log")
    node_path = os.path.join(base_path, "pf_node.csv")
    edge_path = os.path.join(base_path, "pf_edge.csv")
    scenarios_csv_path = os.path.join(
        base_path, "scenarios_" + args.load.generator + ".csv"
    )
    scenarios_plot_path = os.path.join(
        base_path, "scenarios_" + args.load.generator + ".html"
    )
    scenarios_log = os.path.join(base_path, "scenarios_" + args.load.generator + ".log")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tqdm_log.write(f"\nNew generation started at {timestamp}\n")
    with open(error_log_path, "a") as f:
        f.write(f"\nNew generation started at {timestamp}\n")
    with open(scenarios_log, "a") as f:
        f.write(f"\nNew generation started at {timestamp}\n")
    with open(args_log_path, "a") as f:
        f.write(f"\nNew generation started at {timestamp}\n")
        yaml.dump(base_config, f)

    # Load network and scenario data
    if args.network.source == "pandapower":
        net = load_net_from_pp(args.network.name)
    elif args.network.source == "pglib":
        net = load_net_from_pglib(args.network.name)
    elif args.network.source == "file":
        net = load_net_from_file(args.network.name)
    else:
        raise ValueError("Invalid grid source!")

    network_preprocessing(net)
    assert (net.sgen["scaling"] == 1).all(), "Scaling factor >1 not supported yet!"

    load_scenario_generator = get_load_scenario_generator(args)
    scenarios = load_scenario_generator(net, args.load.scenarios, scenarios_log)
    scenarios_df = load_scenarios_to_df(scenarios)
    scenarios_df.to_csv(scenarios_csv_path, index=False)
    plot_load_scenarios_combined(scenarios_df, scenarios_plot_path)

    # Initialize the topology generator
    generator = initialize_generator(
        args.topology_perturbation.type,
        args.topology_perturbation.n_topology_variants,
        args.topology_perturbation.k,
        net,
    )

    # Setup multiprocessing
    manager = Manager()
    progress_queue = manager.Queue()

    # Determine large scenario chunks (each with up to args.large_chunk_size scenarios)
    large_chunks = np.array_split(
        range(args.load.scenarios),
        np.ceil(args.load.scenarios / args.settings.large_chunk_size).astype(int),
    )

    with tqdm(
        total=args.load.scenarios,
        desc="Processing scenarios",
        file=tqdm_log,
        miniters=5,
    ) as pbar:
        for large_chunk_index, large_chunk in enumerate(large_chunks):
            tqdm_log.write(
                f"\n=======\nProcessing large chunk {large_chunk_index + 1} of {len(large_chunks)}\n"
            )
            write_ram_usage(tqdm_log)
            chunk_size = len(large_chunk)
            # Further split the large chunk into smaller parallel chunks
            scenario_chunks = np.array_split(large_chunk, args.settings.num_processes)
            tasks = [
                (
                    chunk[0],
                    chunk[-1] + 1,
                    scenarios,
                    net,
                    progress_queue,
                    generator,
                    args.settings.no_stats,
                    error_log_path,
                )
                for chunk in scenario_chunks
            ]

            # Run parallel processing
            with Pool(processes=args.settings.num_processes) as pool:
                results = [
                    pool.apply_async(process_scenario_chunk, task) for task in tasks
                ]

                # Progress bar update
                completed = 0
                while completed < chunk_size:
                    progress_queue.get()
                    pbar.update(1)
                    completed += 1

                # Gather results from all processes
                csv_data = []
                adjacency_lists = []
                global_stats = Stats() if not args.settings.no_stats else None
                for result in results:
                    local_csv_data, local_adjacency_lists, local_stats = result.get()
                    csv_data.extend(local_csv_data)
                    adjacency_lists.extend(local_adjacency_lists)
                    if not args.settings.no_stats and local_stats:
                        global_stats.merge(local_stats)

                pool.close()
                pool.join()

            # Save processed data immediately to avoid large memory consumption
            save_node_edge_data(net, node_path, edge_path, csv_data, adjacency_lists)
            if not args.settings.no_stats:
                global_stats.save(base_path)
                plot_stats(base_path)

            del csv_data, adjacency_lists, global_stats
            gc.collect()

    print("Data generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()

    # Load the base config
    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    args = NestedNamespace(**base_config)
    main(args)
