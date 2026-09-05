"""
Shared utilities for plot generation.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
from pathlib import Path

# Configuration
# Default location of the sampled parquet snapshot (see README).
# Override per-invocation with --data-dir, which calls set_datasets_folder().
datasets_folder = Path(__file__).parent / "dataset_sampled"


def set_datasets_folder(path):
    """Point the loaders at a different sampled-dataset directory."""
    global datasets_folder
    datasets_folder = Path(path)
    return datasets_folder

# Dataset configurations by mode
dataset_config = {
    "opf": {
        "datasets": [
            "gridfm_datakit_opf",
            "pglearn",
            "opflearn",
            "opfdata",
            
        ],
        "gen_data_available": True,
    },
    "pf": {
        "datasets": [
            "gridfm_datakit_pf",
            "pfdelta",
        ],
        "gen_data_available": False,
    },
}

# IBM Design System color palette for all plots
# https://www.ibm.com/design/language/color
IBM_COLORS = [
    '#0F62FE',  # IBM Blue Core
    '#DA291C',  # IBM Red Core
    '#24A148',  # IBM Green
    '#F1C21B',  # IBM Yellow
    '#FF832B',  # IBM Orange
    '#8E42C9',  # IBM Purple
    '#00B4A0',  # IBM Teal
    '#005D5D',  # IBM Dark Teal
    '#A56EDA',  # IBM Purple Light
    '#42BE65',  # IBM Green Light
]


def load_selected_datasets(selected=None, mode="opf"):
    """Load bus and gen data for selected datasets."""
    print(f"Loading datasets in {mode} mode...")
    bus_data = {}
    gen_data = {}
    description = {}

    targets = selected if selected else dataset_config[mode]["datasets"]
    for name in targets:
        bus_file = datasets_folder / f"{name}_bus_data.parquet"
        gen_file = datasets_folder / f"{name}_gen_data.parquet"
        print(f"Loading {name}...")
        
        df_bus = pd.read_parquet(bus_file)
        bus_data[name] = df_bus
        
        if gen_file.exists():
            df_gen = pd.read_parquet(gen_file)
            gen_data[name] = df_gen
        
        description[name] = name

    return bus_data, gen_data, description


def entropy_from_samples_fixed(x, *, base=2, bins=100, rng=None):
    """Shannon entropy using fixed bins. Returns (entropy, empty_bins)."""

    if np.allclose(x, x[0]):
        # Return 0 entropy when all samples have the same value (no variation)
        return (0.0, bins)

    counts, _ = np.histogram(x, bins=bins, range=rng)
    empty_bins = np.sum(counts == 0)
    
    return (float(entropy(counts, base=base)), int(empty_bins))


def entropy_circular_from_deg_fixed(x_deg, *, base=2, bins=100):
    """Shannon entropy for angular data using fixed bins on (-π, π]."""
    x = np.asarray(x_deg, dtype=float)

    theta = (np.deg2rad(x) + np.pi) % (2 * np.pi) - np.pi
    R = np.hypot(np.cos(theta).mean(), np.sin(theta).mean())  # length on the unit circle
    if 1.0 - R <= 1e-6:
        # Return 0 entropy when all angular samples cluster at the same angle (R ≈ 1)
        return 0.0

    counts, _ = np.histogram(theta, bins=bins, range=(-np.pi, np.pi))
    return float(entropy(counts, base=base))