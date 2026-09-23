import os
import time
import errno
import psutil
from typing import List, Optional, TextIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd


n_scenario_per_partition = 200  # Number of scenarios per partition


def write_parquet(
    df: pd.DataFrame,
    path: str,
    partition_cols: Optional[List[str]] = None,
) -> None:
    """Write a DataFrame to parquet with space-optimized settings.

    Uses zstd compression (smaller and faster to write than the snappy
    default) and BYTE_STREAM_SPLIT encoding for float columns, which
    groups the bytes of each float across values so zstd can exploit the
    shared exponent/sign bytes of physical quantities.

    Args:
        df: DataFrame to write.
        path: Output file path (directory when partition_cols is set).
        partition_cols: Columns to partition the dataset by.
    """
    float_cols = [
        c
        for c in df.columns
        if df[c].dtype.kind == "f" and c not in (partition_cols or [])
    ]
    df.to_parquet(
        path,
        partition_cols=partition_cols,
        engine="pyarrow",
        index=False,
        compression="zstd",
        use_dictionary=False,
        column_encoding={c: "BYTE_STREAM_SPLIT" for c in float_cols},
    )


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


def _retry_on_eagain(func, retries: int = 50, delay: float = 0.01):
    """Call ``func`` retrying on ``EAGAIN``/``BlockingIOError``.

    When many worker processes share a stdout/pipe (e.g. heavy multiprocessing),
    the file descriptor can be non-blocking and a write/flush may raise
    ``BlockingIOError`` ([Errno 11]). Retry briefly instead of propagating the
    error, which would otherwise fail data generation.

    The common (no-error) path is a single ``try`` with no loop or allocation,
    so this adds negligible overhead; the retry loop only runs after a failure.
    """
    try:
        return func()
    except OSError as e:
        if not (isinstance(e, BlockingIOError) or e.errno == errno.EAGAIN):
            raise
    # Slow path: only reached when the first attempt hit EAGAIN.
    for _ in range(retries - 1):
        time.sleep(delay)
        try:
            return func()
        except OSError as e:
            if not (isinstance(e, BlockingIOError) or e.errno == errno.EAGAIN):
                raise
    # Final attempt; let a persistent failure propagate.
    return func()


def _blocking_write(stream, data) -> None:
    """Write to a stream, tolerating ``EAGAIN``/``BlockingIOError``."""
    _retry_on_eagain(lambda: stream.write(data))


def write_ram_usage_distributed(tqdm_log: TextIO) -> None:
    process = psutil.Process(os.getpid())  # Parent process
    mem_usage = process.memory_info().rss / 1024**2  # Parent memory in MB

    # Sum memory usage of all child processes
    for child in process.children(recursive=True):
        mem_usage += child.memory_info().rss / 1024**2

    _blocking_write(
        tqdm_log,
        f"Total RAM usage (Parent + Children): {mem_usage:.2f} MB\n",
    )


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            _blocking_write(s, data)
            _retry_on_eagain(s.flush)

    def flush(self):
        for s in self.streams:
            _retry_on_eagain(s.flush)


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
