import numpy as np
import pandas as pd
from pandapower.auxiliary import pandapowerNet
import os
from gridfm_datakit.utils.config import (
    BUS_COLUMNS,
    BRANCH_COLUMNS,
    GEN_COLUMNS,
    DC_BUS_COLUMNS,
)
from typing import List


def save_node_edge_data(
    net: pandapowerNet,
    node_path: str,
    branch_path: str,
    gen_path: str,
    y_bus_path: str,
    processed_data: List[tuple],
    dcpf: bool = False,
) -> None:
    """Save generated power flow data to CSV files.

    This function takes processed power flow results and saves them to four CSV files:
    - Bus data (bus_data.csv): Bus-level features for each scenario
    - Generator data (gen_data.csv): Generator features for each scenario
    - Branch data (branch_data.csv): Branch features and admittances for each scenario
    - Y-bus data (y_bus_data.csv): Nonzero Y-bus entries for each scenario

    The arrays must conform to the column schemas declared in `gridfm_datakit.utils.config`:
    - Bus data uses BUS_COLUMNS (+ DC_BUS_COLUMNS if dcpf=True)
    - Generator data uses GEN_COLUMNS
    - Branch data uses BRANCH_COLUMNS
    - Y-bus data uses ["index1", "index2", "G", "B"]

    Each CSV file includes a "scenario" column as the first column to identify
    which power flow scenario each row belongs to.

    Args:
        net: Pandapower network (used for determining bus count for scenario indexing).
        node_path: Output file path for bus data CSV (bus_data.csv).
        branch_path: Output file path for branch data CSV (branch_data.csv).
        gen_path: Output file path for generator data CSV (gen_data.csv).
        y_bus_path: Output file path for Y-bus data CSV (y_bus_data.csv).
        processed_data: List of tuples, each containing (bus_array, gen_array, branch_array, y_bus_array)
            from power flow post-processing for one or more scenarios.
        dcpf: If True, includes DC power flow results (Vm_dc, Va_dc columns) in bus data.

    Note:
        - Files are created in append mode, allowing incremental data generation
        - Scenario indices are automatically assigned based on existing data
        - Bus and generator indices are converted to int64 for consistency
        - Headers are only written if the file doesn't already exist
    """
    n_buses = net.bus.shape[0]

    # Determine last scenario index
    last_scenario = -1
    if os.path.exists(node_path):
        existing_df = pd.read_csv(node_path, usecols=["scenario"])
        if not existing_df.empty:
            last_scenario = existing_df["scenario"].iloc[-1]

    # Select columns for bus data
    if dcpf:
        bus_columns = BUS_COLUMNS + DC_BUS_COLUMNS
    else:
        bus_columns = BUS_COLUMNS

    # --- BUS DATA ---
    bus_data = np.concatenate([item[0] for item in processed_data], axis=0)
    bus_df = pd.DataFrame(bus_data, columns=bus_columns)
    bus_df["bus"] = bus_df["bus"].astype("int64")
    scenario_indices = np.repeat(
        range(last_scenario + 1, last_scenario + 1 + (bus_df.shape[0] // n_buses)),
        n_buses,
    )
    bus_df.insert(0, "scenario", scenario_indices)
    bus_df.to_csv(
        node_path,
        mode="a",
        header=not os.path.exists(node_path),
        index=False,
    )

    # --- GEN DATA ---
    gen_data = np.concatenate([item[1] for item in processed_data], axis=0)
    gen_df = pd.DataFrame(gen_data, columns=GEN_COLUMNS)
    gen_df["bus"] = gen_df["bus"].astype("int64")
    # Each scenario may have a different number of generators, so count per scenario
    scenario_indices = np.concatenate(
        [
            np.full(processed_data[i][1].shape[0], last_scenario + 1 + i, dtype="int64")
            for i in range(len(processed_data))
        ],
    )
    gen_df.insert(0, "scenario", scenario_indices)
    gen_df.to_csv(gen_path, mode="a", header=not os.path.exists(gen_path), index=False)

    # --- BRANCH DATA ---
    branch_data = np.concatenate([item[2] for item in processed_data], axis=0)
    branch_df = pd.DataFrame(branch_data, columns=BRANCH_COLUMNS)
    branch_df[["from_bus", "to_bus"]] = branch_df[["from_bus", "to_bus"]].astype(
        "int64",
    )
    scenario_indices = np.concatenate(
        [
            np.full(processed_data[i][2].shape[0], last_scenario + 1 + i, dtype="int64")
            for i in range(len(processed_data))
        ],
    )
    branch_df.insert(0, "scenario", scenario_indices)
    branch_df.to_csv(
        branch_path,
        mode="a",
        header=not os.path.exists(branch_path),
        index=False,
    )

    # --- Y-BUS DATA (nonzero admittance matrix entries) ---
    y_bus_df = pd.DataFrame(
        np.concatenate([item[3] for item in processed_data]),
        columns=["index1", "index2", "G", "B"],
    )
    y_bus_df[["index1", "index2"]] = y_bus_df[["index1", "index2"]].astype("int64")
    scenario_indices = np.concatenate(
        [
            np.full(processed_data[i][3].shape[0], last_scenario + 1 + i, dtype="int64")
            for i in range(len(processed_data))
        ],
    )
    y_bus_df.insert(0, "scenario", scenario_indices)
    y_bus_df.to_csv(
        y_bus_path,
        mode="a",
        header=not os.path.exists(y_bus_path),
        index=False,
    )
