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
from GridDataGen.utils.topology_perturbation import initialize_generator
import psutil
import shutil
import yaml
from multiprocessing import Pool, Manager
from multiprocessing import Queue
from math import ceil


def write_ram_usage(tqdm_log):
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / 1024**2  # Memory in MB
    tqdm_log.write(f"RAM usage: {mem_usage:.2f} MB\n")


def process_one_scenario(
    perturbed_topology,
    no_stats,
    local_csv_data,
    local_adjacency_lists,
    local_branch_idx_removed,
    local_stats,
    error_log_file,
    tqdm_log,
    topology_counter,
):
    """
    Process a load scenario
    """

    try:
        run_pf(perturbed_topology)
    except Exception as e:
        with open(error_log_file, "a") as f:
            f.write(
                f"Caught an exception for topology {topology_counter} in run_pf function: {e}\n"
            )
        return (
            local_csv_data,
            local_adjacency_lists,
            local_branch_idx_removed,
            local_stats,
        )

    # Append processed power flow data
    local_csv_data.extend(pf_post_processing(perturbed_topology))
    local_adjacency_lists.append(get_adjacency_list(perturbed_topology))
    local_branch_idx_removed.append(
        get_branch_idx_removed(perturbed_topology._ppc["branch"])
    )
    if not no_stats:
        local_stats.update(perturbed_topology)

    return local_csv_data, local_adjacency_lists, local_branch_idx_removed, local_stats


def process_topology_chunk(
    start_idx,
    end_idx,
    perturbed_topologies,
    no_stats,
    error_log_file,
    progress_queue,
):
    """
    Process a chunk of topologies in parallel
    """
    print(f"Processing chunk {start_idx} to {end_idx - 1}")
    local_csv_data = []
    local_adjacency_lists = []
    local_branch_idx_removed = []
    local_stats = Stats() if not no_stats else None

    for topology_counter in range(start_idx, end_idx):
        perturbed_topology = perturbed_topologies[topology_counter]
        try:
            (
                local_csv_data,
                local_adjacency_lists,
                local_branch_idx_removed,
                local_stats,
            ) = process_one_scenario(
                perturbed_topology,
                no_stats,
                local_csv_data,
                local_adjacency_lists,
                local_branch_idx_removed,
                local_stats,
                error_log_file,
                None,  # tqdm_log not needed in parallel processing
                topology_counter,
            )
        except Exception as e:
            with open(error_log_file, "a") as f:
                f.write(f"Caught an exception for topology {topology_counter}: {e}\n")

        progress_queue.put(1)

    return local_csv_data, local_adjacency_lists, local_branch_idx_removed, local_stats


def main(args):
    """
    Main routine that loads the network and processes scenarios in parallel.
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
    branch_idx_removed_path = os.path.join(base_path, "branch_idx_removed.csv")
    edge_params_path = os.path.join(base_path, "edge_params.csv")
    bus_params_path = os.path.join(base_path, "bus_params.csv")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tqdm_log.write(f"\nNew generation started at {timestamp}\n")
    with open(error_log_path, "a") as f:
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

    save_edge_params(net, edge_params_path)
    save_bus_params(net, bus_params_path)

    run_opf(net)

    # Initialize the topology generator
    generator = initialize_generator(
        args.topology_perturbation,
        net,
    )

    # Generate all perturbed topologies
    perturbed_topologies = generator.generate(net)
    total_topologies = len(perturbed_topologies)

    # Setup multiprocessing
    manager = Manager()
    progress_queue = manager.Queue()

    # Split topologies into chunks for parallel processing
    chunks = np.array_split(range(total_topologies), args.settings.num_processes)

    # Process scenarios in parallel
    with tqdm(
        total=total_topologies,
        desc="Processing topologies",
        file=tqdm_log,
    ) as pbar:
        # Create tasks for parallel processing
        tasks = [
            (
                chunk[0],
                chunk[-1] + 1,
                perturbed_topologies,
                args.settings.no_stats,
                error_log_path,
                progress_queue,
            )
            for chunk in chunks
        ]

        # Run parallel processing
        with Pool(processes=args.settings.num_processes) as pool:
            print(f"Processing {len(tasks)} tasks")
            results = [pool.apply_async(process_topology_chunk, task) for task in tasks]

            # Progress bar update
            completed = 0
            while completed < total_topologies:
                progress_queue.get()
                pbar.update(1)
                completed += 1

            # Gather results from all processes
            csv_data = []
            adjacency_lists = []
            branch_idx_removed = []
            global_stats = Stats() if not args.settings.no_stats else None

            for result in results:
                (
                    local_csv_data,
                    local_adjacency_lists,
                    local_branch_idx_removed,
                    local_stats,
                ) = result.get()
                csv_data.extend(local_csv_data)
                adjacency_lists.extend(local_adjacency_lists)
                branch_idx_removed.extend(local_branch_idx_removed)
                if not args.settings.no_stats and local_stats:
                    global_stats.merge(local_stats)

            pool.close()
            pool.join()

    # Save final data
    save_node_edge_data(net, node_path, edge_path, csv_data, adjacency_lists)
    save_branch_idx_removed(branch_idx_removed, branch_idx_removed_path)

    if not args.settings.no_stats:
        global_stats.save(base_path)
        plot_stats(base_path)

    print("Data generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="scripts/config/Texas2k_case1_2016summerpeak_contingency.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()

    # Load the base config
    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    args = NestedNamespace(**base_config)
    main(args)
