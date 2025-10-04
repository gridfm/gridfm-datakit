"""
Test cases for genertaing data from gridfm_datakit.generate module
"""

import pytest
import os
from pandapower.auxiliary import pandapowerNet
import shutil
import yaml
from gridfm_datakit.utils.param_handler import NestedNamespace
from gridfm_datakit.generate import (
    _setup_environment,
    _prepare_network_and_scenarios,
    generate_power_flow_data,
    generate_power_flow_data_distributed,
)


@pytest.fixture(params=["secure", "unsecure"])
def conf(request):
    """
    Loads configuration files for both secure and unsecure modes.
    This fixture reads the configuration files and returns both for parametrized testing.
    """
    config_paths = {
        "secure": "tests/config/default_secure.yaml",
        "unsecure": "tests/config/default_unsecure.yaml",
    }

    path = config_paths[request.param]
    with open(path, "r") as f:
        base_config = yaml.safe_load(f)
        args = NestedNamespace(**base_config)
    return args


# Test set up environment function
def test_setup_environment(conf):
    """
    Tests if environment setup works correctly
    """
    args, base_path, file_paths = _setup_environment(conf)
    assert isinstance(file_paths, dict), "File paths should be a dictionary"
    assert "y_bus_data" in file_paths, (
        "Y-bus data file path should be in the dictionary"
    )
    assert "bus_data" in file_paths, "Bus data file path should be in the dictionary"
    assert "branch_data" in file_paths, (
        "Branch data file path should be in the dictionary"
    )
    assert "gen_data" in file_paths, (
        "Generator data file path should be in the dictionary"
    )
    assert os.path.exists(base_path), "Base path should exist"


def test_fail_setup_environment():
    """
    Tests if environment setup fails with a non-existent configuration file
    """
    # Test with a non-existent configuration file
    with pytest.raises(FileNotFoundError):
        args, base_path, file_paths = _setup_environment(
            "scripts/config/non_existent_config.yaml",
        )


# Test prepare network and scenarios function
def test_prepare_network_and_scenarios(conf):
    """
    Tests if network and scenarios are prepared correctly
    """
    # Ensure the configuration is valid
    args, base_path, file_paths = _setup_environment(conf)
    net, scenarios = _prepare_network_and_scenarios(args, file_paths)

    assert isinstance(net, pandapowerNet), "Network should be a pandapowerNet object"
    assert len(scenarios) > 0, "There should be at least one scenario"
    # Check if the network has been loaded correctly
    assert "bus" in net.keys(), "Network should contain bus data"
    assert "line" in net.keys(), "Network should contain line data"


def test_fail_prepare_network_and_scenarios():
    """
    Tests if preparing network and scenarios fails with a non-existent configuration file
    """
    # Test with a non-existent configuration file
    config = "scripts/config/non_existent_config.yaml"
    with pytest.raises(FileNotFoundError):
        args, base_path, file_paths = _setup_environment(config)
        net, scenarios = _prepare_network_and_scenarios(args, file_paths)


def test_fail_prepare_network_and_scenarios_config():
    """
    Tests if preparing network and scenarios fails with an invalid grid source in the configuration file
    """
    config = "tests/config/default_unsecure.yaml"
    args, base_path, file_paths = _setup_environment(config)
    args.network.source = "invalid_source"  # Set invalid source
    with pytest.raises(ValueError, match="Invalid grid source!"):
        net, scenarios = _prepare_network_and_scenarios(args, file_paths)


# Test save network function
def test_save_generated_data():
    """
    Tests if saving generated data works correctly by processing a single scenario
    and verifying that output files are created with correct structure.
    Uses config without perturbations to be sure scenarios converge
    """
    # Use config without perturbations to make sure we don't run into errors with perturbations
    config_path = "tests/config/default_without_perturbation.yaml"
    file_paths = generate_power_flow_data_distributed(config_path)
    print(file_paths)

    # Verify that output files were created
    assert os.path.exists(file_paths["bus_data"]), "Bus data CSV should be created"
    assert os.path.exists(file_paths["branch_data"]), (
        "Branch data CSV should be created"
    )
    assert os.path.exists(file_paths["gen_data"]), (
        "Generator data CSV should be created"
    )
    assert os.path.exists(file_paths["y_bus_data"]), "Y-bus data CSV should be created"

    # Verify files have content (not empty)
    assert os.path.getsize(file_paths["bus_data"]) > 0, (
        "Bus data CSV should not be empty"
    )
    assert os.path.getsize(file_paths["branch_data"]) > 0, (
        "Branch data CSV should not be empty"
    )
    assert os.path.getsize(file_paths["gen_data"]) > 0, (
        "Generator data CSV should not be empty"
    )
    assert os.path.getsize(file_paths["y_bus_data"]) > 0, (
        "Y-bus data CSV should not be empty"
    )


# Test generate pf data function
def test_generate_pf_data(conf):
    """
    Tests if power flow data generation works correctly.
    Requires config path as input.
    """
    file_paths = generate_power_flow_data(conf)
    assert isinstance(file_paths, dict), "File paths should be a dictionary"


def test_fail_generate_pf_data():
    """
    Tests if power flow data generation fails with a non-existent configuration file
    """
    # Test with a non-existent configuration file
    config = "scripts/config/non_existent_config.yaml"
    with pytest.raises(FileNotFoundError):
        generate_power_flow_data(config)


# Clean up generated files after tests
@pytest.fixture(scope="module", autouse=True)
def cleanup_generated_files():
    """
    Cleans up generated files after tests.
    This fixture runs after all tests in the module have completed.
    """
    yield  # This allows tests to run first

    # Only clean up directories that were actually created by these tests
    cleanup_paths = [
        "./tests/test_data_without_perturbation",  # Created by test_save_generated_data() using default_without_perturbation.yaml
        "./tests/test_data_unsecure",  # Created by test_generate_pf_data() using default_unsecure.yaml
        "./tests/test_data_secure",  # Created by test_generate_pf_data() using default_secure.yaml
    ]

    for path in cleanup_paths:
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
                print(f"Cleaned up: {path}")
        except Exception as e:
            print(f"Warning: Could not clean up {path}: {e}")
