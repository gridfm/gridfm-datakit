"""
Test cases for validating the integrity and physical consistency of generated power flow data.

These tests generate data from all available config files and validate using our comprehensive
validation functions to ensure physical consistency across different networks and scenarios.
"""

import pytest
import yaml
import shutil
import os
from gridfm_datakit.generate import generate_power_flow_data_distributed
from gridfm_datakit.validation import validate_generated_data
from gridfm_datakit.utils.param_handler import NestedNamespace
import copy


SKIP_LARGE_GRIDS = os.getenv("SKIP_LARGE_GRIDS", "0") == "1"


small_grids = ["case24_ieee_rts"]

large_grids = [
    "case118_ieee",
    "case197_snem",
    "case240_pserc",
    "case300_ieee",
]

test_cases = small_grids if SKIP_LARGE_GRIDS else small_grids + large_grids


def get_configs():
    default_config_path = "scripts/config/default.yaml"

    with open(default_config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    args = NestedNamespace(**config_dict)

    configs = []
    for name in test_cases:
        new_args = copy.deepcopy(args)
        new_args.network.name = name
        configs.append((name, new_args))
    return configs


# ---- Use readable IDs here ----
configs = get_configs()
param_combinations = [(cfg, mode) for _, cfg in configs for mode in ["opf", "pf"]]
param_ids = [f"{name}-{mode}" for name, _ in configs for mode in ["opf", "pf"]]


@pytest.mark.parametrize("args,mode", param_combinations, ids=param_ids)
def test_data_validation(args, mode):
    """Test each config file by generating data and running validation in both PF and OPF modes."""

    args.load.scenarios = 5
    args.settings.mode = mode
    args.topology_perturbation.n_topology_variants = 5
    # Isolate outputs per xdist worker to avoid cross-worker cleanup and clashes
    worker = os.environ.get("PYTEST_XDIST_WORKER", "local")
    base_dir = f"./tests/test_data_validation_{worker}"
    args.settings.data_dir = f"{base_dir}/{args.network.name}_{mode}"

    # Generate and validate data
    file_paths = generate_power_flow_data_distributed(args)
    validate_generated_data(file_paths, mode, 100.0, n_partitions=10)


@pytest.fixture(scope="session", autouse=True)
def cleanup():
    """Clean up test data after tests complete."""
    yield

    # clean this worker's output directory only
    worker = os.environ.get("PYTEST_XDIST_WORKER", "local")
    base_dir = f"./tests/test_data_validation_{worker}"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir, ignore_errors=True)
        print(f"Cleaned up: {base_dir}")


if __name__ == "__main__":
    # pytest.main([__file__, "-v"])
    test_data_validation("scripts/config/default.yaml")
