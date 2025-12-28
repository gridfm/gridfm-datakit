import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from gridfm_datakit.utils.utils import read_partitions

# -------------------------------
# User config
# -------------------------------
path = "/dccstor/gridfm/powermodels_data/v3/pf/case2000_goc/raw/branch_data.parquet"  # root folder with scenario_partition=...
num_partitions = 1900  # total partitions
num_samples = 100  # partitions to load in this test


# -------------------------------
# Helper
# -------------------------------
def bench(name, func):
    t0 = time.time()
    out = func()
    dt = time.time() - t0
    nrows = len(out) if hasattr(out, "__len__") else "?"
    print(f"{name:<40}: {dt:.4f} s   rows={nrows}")
    return out


# -------------------------------
# Sample partition IDs
# -------------------------------
unsorted_partitions = np.random.choice(num_partitions, size=num_samples, replace=False)
sampled_partitions = np.sort(unsorted_partitions)
print("Sampled partitions:", sampled_partitions)


# -------------------------------
# Method 1: Pandas + filters
# -------------------------------
def load_pandas_filtered():
    return pd.read_parquet(
        path,
        engine="pyarrow",
        filters=[("scenario_partition", "in", unsorted_partitions)],
    )


# -------------------------------
# Method 2: PyArrow fragments
# -------------------------------
def load_pyarrow_fragments():
    partitioning = ds.HivePartitioning.discover()
    dataset = ds.dataset(path, format="parquet", partitioning=partitioning)

    f = ds.field("scenario_partition").isin(sampled_partitions)
    fragments = list(dataset.get_fragments(filter=f))

    tables = [frag.to_table() for frag in fragments]
    return pa.concat_tables(tables).to_pandas() if tables else pd.DataFrame()


# -------------------------------
# Method 3: Pandas + sorted partitions
# -------------------------------
def load_pandas_sorted():
    return pd.read_parquet(
        path,
        engine="pyarrow",
        filters=[("scenario_partition", "in", sampled_partitions)],  # already sorted
    )


# -------------------------------
# Method 4: Direct file load + concat (multiprocessing)
# -------------------------------
def load_direct_files_parallel():
    return read_partitions(
        base_path=path,
        sampled=sampled_partitions,
    )


# -------------------------------
# Run benchmarks
# -------------------------------
df1 = bench("1) Pandas", load_pandas_filtered)
# df2 = bench("2) PyArrow fragments",     load_pyarrow_fragments)
df3 = bench("2) Pandas + sorted", load_pandas_sorted)
df4 = bench("3) Direct load + concat (parallel)", load_direct_files_parallel)
