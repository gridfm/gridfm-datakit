import os
import psutil

def write_ram_usage_distributed(tqdm_log):
    process = psutil.Process(os.getpid())  # Parent process
    mem_usage = process.memory_info().rss / 1024**2  # Parent memory in MB

    # Sum memory usage of all child processes
    for child in process.children(recursive=True):
        mem_usage += child.memory_info().rss / 1024**2

    tqdm_log.write(f"Total RAM usage (Parent + Children): {mem_usage:.2f} MB\n")
