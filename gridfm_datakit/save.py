"""
Data saving and export functionality for power system scenarios.

This module provides functions for saving processed power system data
to parquet files with proper formatting and scenario indexing.
"""

import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from gridfm_datakit.utils.column_names import (
    BUS_COLUMNS,
    DC_BUS_COLUMNS,
    GEN_COLUMNS,
    DC_GEN_COLUMNS,
    BRANCH_COLUMNS,
    DC_BRANCH_COLUMNS,
    RUNTIME_COLUMNS,
    DC_RUNTIME_COLUMNS,
    YBUS_COLUMNS,
)
from gridfm_datakit.network import Network


def _process_and_save(args: Tuple[str, List[np.ndarray], str, int, int, bool]) -> None:
    """Worker function for processing and saving one dataset type (bus/gen/branch/y_bus).

    Args:
        args: Tuple containing:
            - data_type: Type of data to process ("bus", "gen", "branch", "y_bus")
            - processed_data: List of processed data arrays
            - path: Output file path
            - last_scenario: Last scenario index processed
            - n_buses: Number of buses in the network
            - include_dc_res: Whether DC power flow data is included
    """
    data_type, processed_data, path, last_scenario, n_buses, include_dc_res = args

    if data_type == "bus":
        bus_columns = BUS_COLUMNS + DC_BUS_COLUMNS if include_dc_res else BUS_COLUMNS
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
        df = pd.DataFrame(
            gen_data,
            columns=GEN_COLUMNS + DC_GEN_COLUMNS if include_dc_res else GEN_COLUMNS,
        )
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
        df = pd.DataFrame(
            branch_data,
            columns=BRANCH_COLUMNS + DC_BRANCH_COLUMNS
            if include_dc_res
            else BRANCH_COLUMNS,
        )
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
        df = pd.DataFrame(y_bus_data, columns=YBUS_COLUMNS)
        df[["index1", "index2"]] = df[["index1", "index2"]].astype("int64")
        scenario_indices = np.concatenate(
            [
                np.full(item[3].shape[0], last_scenario + 1 + i, dtype="int64")
                for i, item in enumerate(processed_data)
            ],
        )
        df.insert(0, "scenario", scenario_indices)

    elif data_type == "runtime":
        runtime_data = np.concatenate([item[4] for item in processed_data])
        df = pd.DataFrame(
            runtime_data,
            columns=RUNTIME_COLUMNS + DC_RUNTIME_COLUMNS
            if include_dc_res
            else RUNTIME_COLUMNS,
        )
        scenario_indices = np.concatenate(
            [
                np.full(item[4].shape[0], last_scenario + 1 + i, dtype="int64")
                for i, item in enumerate(processed_data)
            ],
        )
        df.insert(0, "scenario", scenario_indices)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    df.to_parquet(
        path,
        append=True if os.path.exists(path) else False,
        engine="fastparquet",
        index=False,
    )


def save_node_edge_data(
    net: Network,
    node_path: str,
    branch_path: str,
    gen_path: str,
    y_bus_path: str,
    runtime_path: str,
    processed_data: List[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
    include_dc_res: bool = False,
) -> None:
    """Save processed power system data to parquet files using parallel processing.

    This function saves bus, generator, branch, and Y-bus data to separate parquet files
    using parallel processing for improved performance.

    Args:
        net: Network object containing system topology information.
        node_path: Path for saving bus/node data parquet file.
        branch_path: Path for saving branch data parquet file.
        gen_path: Path for saving generator data parquet file.
        y_bus_path: Path for saving Y-bus data parquet file.
        processed_data: List of tuples containing processed data arrays for each scenario.
        include_dc_res: Whether DC power flow data is included in the output.
    """
    n_buses = net.buses.shape[0]

    # Determine last scenario index (only once)
    last_scenario = -1
    if os.path.exists(node_path):
        existing_df = pd.read_parquet(
            node_path,
            columns=["scenario"],
            engine="fastparquet",
        )
        if not existing_df.empty:
            last_scenario = existing_df["scenario"].iloc[-1]

    # Define arguments per data type
    tasks = [
        ("bus", processed_data, node_path, last_scenario, n_buses, include_dc_res),
        ("gen", processed_data, gen_path, last_scenario, n_buses, include_dc_res),
        ("branch", processed_data, branch_path, last_scenario, n_buses, include_dc_res),
        ("y_bus", processed_data, y_bus_path, last_scenario, n_buses, include_dc_res),
        (
            "runtime",
            processed_data,
            runtime_path,
            last_scenario,
            n_buses,
            include_dc_res,
        ),
    ]

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_process_and_save, task) for task in tasks]
        for f in futures:
            f.result()  # wait for each task to finish
