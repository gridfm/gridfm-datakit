import time
import shutil
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

# -----------------------------
# Load data
# -----------------------------
df_test = pd.read_parquet(
    "/dccstor/gridfm/powermodels_data/v1/opf/case118_ieee/raw/branch_data.parquet",
)

print(f"Data loaded: {df_test.shape[0]} rows, {df_test.shape[1]} columns")


# -----------------------------
# Benchmark Pandas writer
# -----------------------------
def benchmark_pandas(n):
    outdir = "/dccstor/gridfm/powermodels_data/del/to_delete_pandas"
    shutil.rmtree(outdir, ignore_errors=True)
    df_test["partition"] = df_test["scenario"] // n
    start = time.time()
    df_test.to_parquet(
        outdir,
        engine="pyarrow",
        partition_cols=["partition"],
        index=False,
    )
    end = time.time()

    return end - start


# -----------------------------
# Benchmark PyArrow writer
# -----------------------------
def benchmark_pyarrow(n):
    outdir = "/dccstor/gridfm/powermodels_data/del/to_delete_pa"
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs("/dccstor/gridfm/powermodels_data/del/", exist_ok=True)

    table = pa.Table.from_pandas(df_test, preserve_index=False)

    pa.set_cpu_count(4)
    df_test["partition"] = df_test["scenario"] // n

    start = time.time()
    pq.write_to_dataset(
        table,
        root_path=outdir,
        partition_cols=["partition"],
        use_threads=True,
    )
    end = time.time()

    return end - start


# -----------------------------
# Run benchmarks
# -----------------------------
pandas_time_200 = benchmark_pandas(200)
pa_time_200 = benchmark_pyarrow(200)

print("\n=== RESULTS ===")
print(f"Pandas to_parquet 200: {pandas_time_200:.4f} seconds")
print(f"PyArrow write_to_dataset 200: {pa_time_200:.4f} seconds")
print(f"Speed-up: {pandas_time_200 / pa_time_200:.2f}× faster\n")

pandas_time_500 = benchmark_pandas(500)
pa_time_500 = benchmark_pyarrow(500)

print("\n=== RESULTS ===")
print(f"Pandas to_parquet 500: {pandas_time_500:.4f} seconds")
print(f"PyArrow write_to_dataset 500: {pa_time_500:.4f} seconds")
print(f"Speed-up: {pandas_time_500 / pa_time_500:.2f}× faster\n")
