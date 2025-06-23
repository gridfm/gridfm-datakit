"""
Test cases for genertaing data from gridfm_datakit.generate module
pytest tests/ --config scripts/config/default.yaml
"""

import pytest
import os
from pandapower.auxiliary import pandapowerNet
from tqdm import tqdm
from gridfm_datakit.utils.param_handler import initialize_generator
import shutil
from gridfm_datakit.utils.utils import Tee
import yaml
from gridfm_datakit.utils.param_handler import NestedNamespace
from gridfm_datakit.utils.stats import Stats
from gridfm_datakit.process.process_network import (
    process_scenario_contingency,
    process_scenario,
)
from gridfm_datakit.generate import (
    _setup_environment,
    _prepare_network_and_scenarios,
    _save_generated_data,
    generate_power_flow_data,
    generate_power_flow_data_distributed,
)
import sys


@pytest.fixture
def conf():
    """
    Loads the default configuration from a YAML file.
    This fixture reads the configuration file located at "tests/config/default.yaml
    """
    path = "tests/config/default.yaml"  # Default path to the config file
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
    assert "edge_data" in file_paths, "Network file path should be in the dictionary"
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
    Tests if preparing network and scenarios fails with an incorrect grid name in the configuration file
    """
    # Test with a non-existent configuration file
    config = "tests/config/default.yaml"
    args, base_path, file_paths = _setup_environment(config)
    args.network.name = "non_existent_grid"
    args.network.source = "pandapower"
    with pytest.raises(AttributeError, match="Invalid grid source!"):
        if args.network.source == "pandapower":
            try:
                net, scenarios = _prepare_network_and_scenarios(args, file_paths)
            except AttributeError:
                raise AttributeError("Invalid grid source!")


# Test save network function
def test_save_generated_data(conf):
    """
    Tests if saving generated data works correctly
    """
    # Setup environment
    args, base_path, file_paths = _setup_environment(conf)

    # Prepare network and scenarios
    net, scenarios = _prepare_network_and_scenarios(args, file_paths)

    # Initialize topology generator and data structures
    generator = initialize_generator(args.topology_perturbation, net)
    csv_data = []
    adjacency_lists = []
    branch_idx_removed = []
    global_stats = Stats() if not args.settings.no_stats else None

    # Process scenarios sequentially
    with open(file_paths["tqdm_log"], "a") as f:
        with tqdm(
            total=args.load.scenarios,
            desc="Processing scenarios",
            file=Tee(sys.stdout, f),
            miniters=5,
        ) as pbar:
            for scenario_index in range(args.load.scenarios):
                # Process the scenario
                if args.settings.mode == "pf":
                    csv_data, adjacency_lists, branch_idx_removed, global_stats = (
                        process_scenario(
                            net,
                            scenarios,
                            scenario_index,
                            generator,
                            args.settings.no_stats,
                            csv_data,
                            adjacency_lists,
                            branch_idx_removed,
                            global_stats,
                            file_paths["error_log"],
                        )
                    )
                elif args.settings.mode == "contingency":
                    csv_data, adjacency_lists, branch_idx_removed, global_stats = (
                        process_scenario_contingency(
                            net,
                            scenarios,
                            scenario_index,
                            generator,
                            args.settings.no_stats,
                            csv_data,
                            adjacency_lists,
                            branch_idx_removed,
                            global_stats,
                            file_paths["error_log"],
                        )
                    )

                pbar.update(1)

    # Save final data
    _save_generated_data(
        net,
        csv_data,
        adjacency_lists,
        branch_idx_removed,
        global_stats,
        file_paths,
        base_path,
        args,
    )
    assert file_paths is not None, "File paths should not be None"


# Test generate pf data function
def test_generate_pf_data():
    """
    Tests if power flow data generation works correctly.
    Requires config path as input.
    """
    config = "tests/config/default.yaml"
    file_paths = generate_power_flow_data(config)
    assert isinstance(file_paths, dict), "File paths should be a dictionary"


def test_fail_generate_pf_data():
    """
    Tests if power flow data generation fails with a non-existent configuration file
    """
    # Test with a non-existent configuration file
    config = "scripts/config/non_existent_config.yaml"
    with pytest.raises(FileNotFoundError):
        generate_power_flow_data(config)


# Test generate pf data distributed function
def test_generate_pf_data_distributed():
    """
    Tests if distributed power flow data generation works correctly.
    Requires config path as input.
    """
    config = "tests/config/default.yaml"
    file_paths = generate_power_flow_data_distributed(config)
    assert isinstance(file_paths, dict), "File paths should be a dictionary"


def test_fail_generate_pf_data_distributed():
    """
    Tests if distributed power flow data generation fails with a non-existent configuration file
    """
    # Test with a non-existent configuration file
    config = "scripts/config/non_existent_config.yaml"
    with pytest.raises(FileNotFoundError):
        generate_power_flow_data_distributed(config)


# Test generate pf data function for contingency scenarios
def test_generate_pf_data_contingency():
    """
    Tests if power flow data generation works correctly.
    Requires config path as input.
    """
    config = "tests/config/default_contingency.yaml"
    file_paths = generate_power_flow_data(config)
    assert isinstance(file_paths, dict), "File paths should be a dictionary"


# Test generate pf data distributed function for contingency scenarios
def test_generate_pf_data_distributed_contingency():
    """
    Tests if distributed power flow data generation works correctly.
    Requires config path as input.
    """
    config = "tests/config/default_contingency.yaml"
    file_paths = generate_power_flow_data_distributed(config)
    assert isinstance(file_paths, dict), "File paths should be a dictionary"


def test_fail_generate_pf_data_contingency():
    """
    Tests if power flow data generation fails with a non-existent configuration file
    """
    # Test with a non-existent configuration file
    config = "scripts/config/non_existent_config.yaml"
    with pytest.raises(FileNotFoundError):
        generate_power_flow_data(config)


def test_fail_generate_pf_data_distributed_contingency():
    """
    Tests if distributed power flow data generation fails with a non-existent configuration file
    """
    # Test with a non-existent configuration file
    config = "scripts/config/non_existent_config.yaml"
    with pytest.raises(FileNotFoundError):
        generate_power_flow_data_distributed(config)


# Clean up generated files after tests
@pytest.fixture(scope="module", autouse=True)
def cleanup_generated_files():
    """
    Cleans up generated files after tests.
    This fixture runs after all tests in the module have completed.
    """
    # Define the paths to the generated files
    generated_files = ["tests/test_data/*", "tests/test_data_contingency/*"]

    # Remove the generated files if they exist
    for file_path in generated_files:
        if os.path.exists(file_path):
            os.remove(file_path)

    # Remove the base data directory if it exists
    if os.path.exists("tests/test_data"):
        shutil.rmtree("tests/test_data")
    # Remove the base data directory if it exists
    if os.path.exists("tests/test_data_contingency"):
        shutil.rmtree("tests/test_data_contingency")


# Run the cleanup fixture after all tests in the module
@pytest.fixture(scope="session", autouse=True)
def cleanup_session():
    """
    Cleans up session-wide resources after all tests in the session have completed.
    This fixture runs once per test session.
    """
    # Perform any necessary cleanup here, if needed
    pass
