import os
import psutil
from typing import TextIO


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

    raise ValueError(f"No n_scenarios metadata file found in {data_dir}")


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


# Open your log file
