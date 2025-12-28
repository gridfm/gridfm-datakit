import os
import psutil
from typing import TextIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd


n_scenario_per_partition = 200  # Number of scenarios per partition


def get_num_scenarios(data_dir: str) -> int:
    """Get total number of scenarios from data directory.

    Reads from n_scenarios.txt metadata file in the data directory.

    Args:
        data_dir: Directory containing parquet files and n_scenarios.txt

    Returns:
        Total number of scenarios

    Raises:
        ValueError: If n_scenarios.txt metadata file not found
    """
    n_scenarios_file = os.path.join(data_dir, "n_scenarios.txt")
    if os.path.exists(n_scenarios_file):
        with open(n_scenarios_file, "r") as f:
            return int(f.read().strip())

    else:
        print(
            f"No n_scenarios metadata file found in {data_dir}, using bus_data.parquet to get total number of scenarios",
        )
        return int(
            pd.read_parquet(
                os.path.join(data_dir, "bus_data.parquet"),
                engine="pyarrow",
            )["scenario"].max()
            + 1,
        )


def write_ram_usage_distributed(tqdm_log: TextIO) -> None:
    process = psutil.Process(os.getpid())  # Parent process
    mem_usage = process.memory_info().rss / 1024**2  # Parent memory in MB

    # Sum memory usage of all child processes
    for child in process.children(recursive=True):
        mem_usage += child.memory_info().rss / 1024**2

    tqdm_log.write(f"Total RAM usage (Parent + Children): {mem_usage:.2f} MB\n")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def read_partitions(
    base_path: str,
    sampled: list,
    max_workers: int = None,
) -> pd.DataFrame:
    """Read sampled partition folders in parallel and concatenate them."""
    if max_workers is None:
        from os import cpu_count

        max_workers = min(32, cpu_count())  # sensible default

    dfs = []

    # Submit all partition reads to the ThreadPool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                pd.read_parquet,
                os.path.join(base_path, f"scenario_partition={k}"),
                engine="pyarrow",
            ): k
            for k in sampled
        }

        # Collect results as they complete with tqdm
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Reading {len(sampled)} partitions from {base_path}",
        ):
            df = future.result()
            dfs.append(df)

    # Concatenate all partitions
    return pd.concat(dfs, ignore_index=True)
