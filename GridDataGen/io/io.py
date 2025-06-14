import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd
from importlib import resources
from pandapower.auxiliary import pandapowerNet
import os
import requests
from pandapower.pypower.idx_brch import T_BUS, F_BUS, RATE_A, BR_STATUS
from pandapower.pypower.idx_bus import BUS_I, BUS_TYPE, VMIN, VMAX, BASE_KV
from pandapower.pypower.makeYbus import branch_vectors
import psutil


def write_ram_usage_distributed(tqdm_log):
    process = psutil.Process(os.getpid())  # Parent process
    mem_usage = process.memory_info().rss / 1024**2  # Parent memory in MB

    # Sum memory usage of all child processes
    for child in process.children(recursive=True):
        mem_usage += child.memory_info().rss / 1024**2

    tqdm_log.write(f"Total RAM usage (Parent + Children): {mem_usage:.2f} MB\n")


def load_net_from_pp(grid_name: str) -> pandapowerNet:
    """Loads a network from the pandapower library.

    Args:
        grid_name: Name of the grid case file in pandapower library.

    Returns:
        pandapowerNet: Loaded power network configuration.
    """
    network = getattr(pn, grid_name)()
    return network


def load_net_from_file(grid_name: str) -> pandapowerNet:
    """Loads a network from a matpower file.

    Args:
        grid_name: Name of the matpower file (without extension).

    Returns:
        pandapowerNet: Loaded power network configuration with reindexed buses.
    """
    file_path = str(resources.files(f"GridDataGen.grids").joinpath(f"{grid_name}.m"))
    network = pp.converter.from_mpc(str(file_path))

    old_bus_indices = network.bus.index
    new_bus_indices = range(len(network.bus))

    # Create a mapping dictionary
    bus_mapping = dict(zip(old_bus_indices, new_bus_indices))

    # Reindex the buses in the network
    pp.reindex_buses(network, bus_mapping)

    return network


def load_net_from_pglib(grid_name: str) -> pandapowerNet:
    """Loads a power grid network from PGLib.

    Downloads the network file if not locally available and loads it into a pandapower network.
    The buses are reindexed to ensure continuous indices.

    Args:
        grid_name: Name of the grid file without the prefix 'pglib_opf_' (e.g., 'case14_ieee', 'case118_ieee').

    Returns:
        pandapowerNet: Loaded power network configuration with reindexed buses.

    Raises:
        requests.exceptions.RequestException: If download fails.
    """
    # Construct file paths
    file_path = str(
        resources.files(f"GridDataGen.grids").joinpath(f"pglib_opf_{grid_name}.m")
    )

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Download file if not exists
    if not os.path.exists(file_path):
        url = f"https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/pglib_opf_{grid_name}.m"
        response = requests.get(url)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            f.write(response.content)

    # Load network from file
    network = pp.converter.from_mpc(file_path)

    old_bus_indices = network.bus.index
    new_bus_indices = range(len(network.bus))

    # Create a mapping dictionary
    bus_mapping = dict(zip(old_bus_indices, new_bus_indices))

    # Reindex the buses in the network
    pp.reindex_buses(network, bus_mapping)

    return network


def save_edge_params(net: pandapowerNet, path: str):
    """Saves edge parameters for the network to a CSV file.

    Extracts and saves branch parameters including admittance matrices and rate limits.

    Args:
        net: The power network.
        path: Path where the edge parameters CSV file should be saved.
    """
    pp.rundcpp(net)  # need to run dcpp to create the ppc structure
    ppc = net._ppc
    to_bus = np.real(ppc["branch"][:, T_BUS])
    from_bus = np.real(ppc["branch"][:, F_BUS])
    Ytt, Yff, Yft, Ytf = branch_vectors(ppc["branch"], ppc["branch"].shape[0])
    Ytt_r = np.real(Ytt)
    Ytt_i = np.imag(Ytt)
    Yff_r = np.real(Yff)
    Yff_i = np.imag(Yff)
    Yft_r = np.real(Yft)
    Yft_i = np.imag(Yft)
    Ytf_r = np.real(Ytf)
    Ytf_i = np.imag(Ytf)

    rate_a = np.real(ppc["branch"][:, RATE_A])
    edge_params = pd.DataFrame(
        np.column_stack(
            (
                from_bus,
                to_bus,
                Yff_r,
                Yff_i,
                Yft_r,
                Yft_i,
                Ytf_r,
                Ytf_i,
                Ytt_r,
                Ytt_i,
                rate_a,
            )
        ),
        columns=[
            "from_bus",
            "to_bus",
            "Yff_r",
            "Yff_i",
            "Yft_r",
            "Yft_i",
            "Ytf_r",
            "Ytf_i",
            "Ytt_r",
            "Ytt_i",
            "rate_a",
        ],
    )
    # comvert everything to float32
    edge_params = edge_params.astype(np.float32)
    edge_params.to_csv(path, index=False)


def save_bus_params(net: pandapowerNet, path: str):
    """Saves bus parameters for the network to a CSV file.

    Extracts and saves bus parameters including voltage limits and base values.

    Args:
        net: The power network.
        path: Path where the bus parameters CSV file should be saved.
    """
    idx = net.bus.index
    base_kv = net.bus.vn_kv
    bus_type = net.bus.type
    vmin = net.bus.min_vm_pu
    vmax = net.bus.max_vm_pu

    bus_params = pd.DataFrame(
        np.column_stack((idx, bus_type, vmin, vmax, base_kv)),
        columns=["bus", "type", "vmin", "vmax", "baseKV"],
    )
    bus_params.to_csv(path, index=False)


def save_branch_idx_removed(branch_idx_removed, path: str):
    """Saves indices of removed branches for each scenario.

    Appends the removed branch indices to an existing CSV file or creates a new one.

    Args:
        branch_idx_removed: List of removed branch indices for each scenario.
        path: Path where the branch indices CSV file should be saved.
    """
    if os.path.exists(path):
        existing_df = pd.read_csv(path, usecols=["scenario"])
        if not existing_df.empty:
            last_scenario = existing_df["scenario"].iloc[-1]
    else:
        last_scenario = -1

    scenario_idx = np.arange(
        last_scenario + 1, last_scenario + 1 + len(branch_idx_removed)
    )
    branch_idx_removed_df = pd.DataFrame(branch_idx_removed)
    branch_idx_removed_df.insert(0, "scenario", scenario_idx)
    branch_idx_removed_df.to_csv(
        path, mode="a", header=not os.path.exists(path), index=False
    )  # append to existing file or create new one


def save_node_edge_data(
    net: pandapowerNet,
    node_path: str,
    edge_path: str,
    csv_data: list,
    adjacency_lists: list,
    mode: str = "pf",
):
    """Saves generated node and edge data to CSV files.

    Saves generated data for nodes and edges,
    appending to existing files if they exist.

    Args:
        net: The power network.
        node_path: Path where node data should be saved.
        edge_path: Path where edge data should be saved.
        csv_data: List of node-level data for each scenario.
        adjacency_lists: List of edge-level adjacency lists for each scenario.
        mode: Analysis mode, either 'pf' for power flow or 'contingency' for contingency analysis.
    """
    n_buses = net.bus.shape[0]

    # Determine last scenario index
    last_scenario = -1
    if os.path.exists(node_path):
        existing_df = pd.read_csv(node_path, usecols=["scenario"])
        if not existing_df.empty:
            last_scenario = existing_df["scenario"].iloc[-1]

    # Create DataFrame for node data
    if mode == "pf":
        df = pd.DataFrame(
            csv_data,
            columns=[
                "bus",
                "Pd",
                "Qd",
                "Pg",
                "Qg",
                "Vm",
                "Va",
                "PQ",
                "PV",
                "REF",
            ],
        )
    elif (
        mode == "contingency"
    ):  # we add the dc voltage to the node data for benchmarking purposes
        df = pd.DataFrame(
            csv_data,
            columns=[
                "bus",
                "Pd",
                "Qd",
                "Pg",
                "Qg",
                "Vm",
                "Va",
                "PQ",
                "PV",
                "REF",
                "Vm_dc",
                "Va_dc",
            ],
        )

    df["bus"] = df["bus"].astype("int64")

    # Shift scenario indices
    scenario_indices = np.repeat(
        range(last_scenario + 1, last_scenario + 1 + (df.shape[0] // n_buses)), n_buses
    )  # repeat each scenario index n_buses times since there are n_buses rows for each scenario
    df.insert(0, "scenario", scenario_indices)

    # Append to CSV
    df.to_csv(node_path, mode="a", header=not os.path.exists(node_path), index=False)

    # Create DataFrame for edge data
    adj_df = pd.DataFrame(
        np.concatenate(adjacency_lists),
        columns=["index1", "index2", "G", "B"],
    )

    adj_df[["index1", "index2"]] = adj_df[["index1", "index2"]].astype("int64")

    # Shift scenario indices
    scenario_indices = np.concatenate(
        [
            np.full(adjacency_lists[i].shape[0], last_scenario + 1 + i, dtype="int64")
            for i in range(len(adjacency_lists))
        ]
    )  # for each scenario, we repeat the scenario index as many times as there are edges in the scenario
    adj_df.insert(0, "scenario", scenario_indices)

    # Append to CSV
    adj_df.to_csv(
        edge_path, mode="a", header=not os.path.exists(edge_path), index=False
    )
