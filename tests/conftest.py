# tests/conftest.py
import multiprocessing as mp
import os

# Force spawn for all multiprocessing pools
try:
    print("SKIP_LARGE_GRIDS ", os.getenv("SKIP_LARGE_GRIDS", "0") == "1")
    mp.set_start_method("spawn", force=True)
    print("Set multiprocessing start method to 'spawn'")
except RuntimeError:
    # Start method already set (e.g., if tests run multiple times)
    pass
