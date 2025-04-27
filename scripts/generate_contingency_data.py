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




def process_scenario_contingency(
    net,
    scenarios,
    scenario_index,
    generator,
    no_stats,
    local_csv_data,
    local_adjacency_lists,
    local_stats,
    error_log_file,
):
    """
    Process a load scenario
    """
    net = copy.deepcopy(net)
    net.load.p_mw = scenarios[net.load.bus, scenario_index, 0]
    net.load.q_mvar = scenarios[net.load.bus, scenario_index, 1]
    try:
        run_opf(net)
    except Exception as e:
        with open(error_log_file, "a") as f:
            f.write(
                f"Caught an exception at scenario {scenario_index} in run_opf function: {e}\n"
            )
        return local_csv_data, local_adjacency_lists, local_stats

    net_pf = copy.deepcopy(net)
    net_pf = pf_preprocessing(net_pf)

    # Generate perturbed topologies
    perturbed_topologies = generator.generate(net_pf)

    for perturbed_topology in perturbed_topologies:
        try:
            run_pf(perturbed_topology)
        except Exception as e:
            with open(error_log_file, "a") as f:
                f.write(
                    f"Caught an exception at scenario {scenario_index} in run_pf function: {e}\n"
                )

                # TODO 1: What to do when the network does not converge for AC-PF? -> we dont have targets for regression!!
        

        # Append processed power flow data
        local_csv_data.extend(pf_post_processing(net_pf))
        local_adjacency_lists.append(get_adjacency_list(net_pf))
        if not no_stats:
            local_stats.update(net_pf)

    return local_csv_data, local_adjacency_lists, local_stats


def write_ram_usage(tqdm_log):
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / 1024**2  # Memory in MB
    tqdm_log.write(f"RAM usage: {mem_usage:.2f} MB\n")


def main(args):
    """
    Main routine that loads the network and processes scenarios sequentially.
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
        args.topology_perturbation.elements,
        net,
    )

    # Initialize data structures
    csv_data = []
    adjacency_lists = []
    global_stats = Stats() if not args.settings.no_stats else None

    # Process scenarios sequentially
    with tqdm(
        total=args.load.scenarios,
        desc="Processing scenarios",
        file=tqdm_log,
        miniters=5,
    ) as pbar:
        for scenario_index in range(args.load.scenarios):

            # Process the scenario
            csv_data, adjacency_lists, global_stats = process_scenario_contingency(
                net,
                scenarios,
                scenario_index,
                generator,
                args.settings.no_stats,
                csv_data,
                adjacency_lists,
                global_stats,
                error_log_path,
            )

            pbar.update(1)

    # Save final data
    save_node_edge_data(net, node_path, edge_path, csv_data, adjacency_lists)
    if not args.settings.no_stats:
        global_stats.save(base_path)
        plot_stats(base_path)

    print("Data generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="scripts/config/default.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()

    # Load the base config
    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    args = NestedNamespace(**base_config)
    main(args)
