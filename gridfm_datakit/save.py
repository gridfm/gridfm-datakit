import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from pandapower import pandapowerNet
from gridfm_datakit.utils.config import (
    BUS_COLUMNS,
    DC_BUS_COLUMNS,
    GEN_COLUMNS,
    BRANCH_COLUMNS,
)


def _process_and_save(args):
    """Worker function for one dataset type (bus/gen/branch/y_bus)."""
    data_type, processed_data, path, last_scenario, n_buses, dcpf = args

    if data_type == "bus":
        bus_columns = BUS_COLUMNS + DC_BUS_COLUMNS if dcpf else BUS_COLUMNS
        bus_data = np.concatenate([item[0] for item in processed_data], axis=0)
        df = pd.DataFrame(bus_data, columns=bus_columns)
        df["bus"] = df["bus"].astype("int64")
        scenario_indices = np.repeat(
            range(last_scenario + 1, last_scenario + 1 + (df.shape[0] // n_buses)),
            n_buses,
        )
        df.insert(0, "scenario", scenario_indices)

    elif data_type == "gen":
        gen_data = np.concatenate([item[1] for item in processed_data], axis=0)
        df = pd.DataFrame(gen_data, columns=GEN_COLUMNS)
        df["bus"] = df["bus"].astype("int64")
        scenario_indices = np.concatenate(
            [
                np.full(item[1].shape[0], last_scenario + 1 + i, dtype="int64")
                for i, item in enumerate(processed_data)
            ],
        )
        df.insert(0, "scenario", scenario_indices)

    elif data_type == "branch":
        branch_data = np.concatenate([item[2] for item in processed_data], axis=0)
        df = pd.DataFrame(branch_data, columns=BRANCH_COLUMNS)
        df[["from_bus", "to_bus"]] = df[["from_bus", "to_bus"]].astype("int64")
        scenario_indices = np.concatenate(
            [
                np.full(item[2].shape[0], last_scenario + 1 + i, dtype="int64")
                for i, item in enumerate(processed_data)
            ],
        )
        df.insert(0, "scenario", scenario_indices)

    elif data_type == "y_bus":
        y_bus_data = np.concatenate([item[3] for item in processed_data])
        df = pd.DataFrame(y_bus_data, columns=["index1", "index2", "G", "B"])
        df[["index1", "index2"]] = df[["index1", "index2"]].astype("int64")
        scenario_indices = np.concatenate(
            [
                np.full(item[3].shape[0], last_scenario + 1 + i, dtype="int64")
                for i, item in enumerate(processed_data)
            ],
        )
        df.insert(0, "scenario", scenario_indices)

    else:
        raise ValueError(f"Unknown data type: {data_type}")

    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def save_node_edge_data(
    net: pandapowerNet,
    node_path: str,
    branch_path: str,
    gen_path: str,
    y_bus_path: str,
    processed_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    dcpf: bool = False,
) -> None:
    """Fully parallel version — each (bus, gen, branch, y_bus) runs in its own process."""
    n_buses = net.bus.shape[0]

    # Determine last scenario index (only once)
    last_scenario = -1
    if os.path.exists(node_path):
        existing_df = pd.read_csv(node_path, usecols=["scenario"])
        if not existing_df.empty:
            last_scenario = existing_df["scenario"].iloc[-1]

    # Define arguments per data type
    tasks = [
        ("bus", processed_data, node_path, last_scenario, n_buses, dcpf),
        ("gen", processed_data, gen_path, last_scenario, n_buses, dcpf),
        ("branch", processed_data, branch_path, last_scenario, n_buses, dcpf),
        ("y_bus", processed_data, y_bus_path, last_scenario, n_buses, dcpf),
    ]

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_process_and_save, task) for task in tasks]
        for f in futures:
            f.result()  # wait for each task to finish
