import pandapower as pp
import numpy as np
import pandas as pd
from pandapower.auxiliary import pandapowerNet
import os
from pandapower.pypower.idx_brch import T_BUS, F_BUS, RATE_A
from pandapower.pypower.makeYbus import branch_vectors
from typing import List


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
            ),
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


def save_branch_idx_removed(branch_idx_removed: List[List[int]], path: str):
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
        last_scenario + 1,
        last_scenario + 1 + len(branch_idx_removed),
    )
    branch_idx_removed_df = pd.DataFrame(branch_idx_removed)
    branch_idx_removed_df.insert(0, "scenario", scenario_idx)
    branch_idx_removed_df.to_csv(
        path,
        mode="a",
        header=not os.path.exists(path),
        index=False,
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
        range(last_scenario + 1, last_scenario + 1 + (df.shape[0] // n_buses)),
        n_buses,
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
        ],
    )  # for each scenario, we repeat the scenario index as many times as there are edges in the scenario
    adj_df.insert(0, "scenario", scenario_indices)

    # Append to CSV
    adj_df.to_csv(
        edge_path,
        mode="a",
        header=not os.path.exists(edge_path),
        index=False,
    )
