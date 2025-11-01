"""
Test cases for validating the integrity and physical consistency of generated power flow data.

These tests generate data from all available config files and validate using our comprehensive
validation functions to ensure physical consistency across different networks and scenarios.
"""

import pytest
import glob
import yaml
import shutil
import os
from pathlib import Path
from gridfm_datakit.generate import generate_power_flow_data_distributed
from gridfm_datakit.validation import validate_generated_data
from gridfm_datakit.utils.param_handler import NestedNamespace


def get_config_files():
    """Get all YAML config files, excluding slow ones."""
    config_files = glob.glob("scripts/config/*.yaml") + glob.glob("tests/config/*.yaml")

    # Exclude slow configs
    excluded = [
        "scripts/config/Texas2k_case1_2016summerpeak.yaml",  # too slow
        "scripts/config/case1354_pegase.yaml",  # too slow
        "scripts/config/case179_goc.yaml",  # bad convergence
    ]

    return [f for f in config_files if f not in excluded]


@pytest.mark.parametrize("config_path", get_config_files())
def test_data_validation(config_path):
    """Test each config file by generating data and running validation."""
    config_name = Path(config_path).stem

    # Load and modify config for testing
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    args = NestedNamespace(**config_dict)
    args.load.scenarios = 5
    args.topology_perturbation.n_topology_variants = 5
    # Isolate outputs per xdist worker to avoid cross-worker cleanup and clashes
    worker = os.environ.get("PYTEST_XDIST_WORKER", "local")
    base_dir = f"./tests/test_data_validation_{worker}"
    args.settings.data_dir = f"{base_dir}/{config_name}"

    # Generate and validate data
    file_paths = generate_power_flow_data_distributed(args, plot=False)
    mode = args.settings.mode
    validate_generated_data(file_paths, mode, n_scenarios=10)


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
    pytest.main([__file__, "-v"])
