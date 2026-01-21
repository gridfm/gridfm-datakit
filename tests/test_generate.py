"""
Test cases for generating data from gridfm_datakit.generate module
"""

import pytest
import os

import shutil
import tempfile
import yaml
import pandas as pd
from gridfm_datakit.network import Network
from gridfm_datakit.utils.param_handler import NestedNamespace
from gridfm_datakit.generate import (
    _setup_environment,
    _prepare_network_and_scenarios,
    generate_power_flow_data,
    generate_power_flow_data_distributed,
)


@pytest.fixture(params=["opf", "pf"])
def conf(request):
    """
    Loads configuration files for both opf and pf modes.
    This fixture reads the configuration files and returns both for parametrized testing.
    """
    config_paths = {
        "opf": "tests/config/default_opf_mode_test.yaml",
        "pf": "tests/config/default_pf_mode_test.yaml",
    }

    path = config_paths[request.param]
    with open(path, "r") as f:
        base_config = yaml.safe_load(f)
        args = NestedNamespace(**base_config)
    # Isolate outputs per xdist worker
    worker = os.environ.get("PYTEST_XDIST_WORKER", "local")
    args.settings.data_dir = f"./tests/test_data_{request.param}_mode_{worker}"
    return args


# Test set up environment function
def test_setup_environment(conf):
    """
    Tests if environment setup works correctly
    """
    args, base_path, file_paths, seed = _setup_environment(conf)
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
        args, base_path, file_paths, seed = _setup_environment(
            "scripts/config/non_existent_config.yaml",
        )


# Test prepare network and scenarios function
def test_prepare_network_and_scenarios(conf):
    """
    Tests if network and scenarios are prepared correctly
    """
    # Ensure the configuration is valid
    args, base_path, file_paths, seed = _setup_environment(conf)
    net, scenarios = _prepare_network_and_scenarios(args, file_paths, seed)

    assert isinstance(net, Network), "Network should be a Network object"
    assert len(scenarios) > 0, "There should be at least one scenario"


def test_fail_prepare_network_and_scenarios():
    """
    Tests if preparing network and scenarios fails with a non-existent configuration file
    """
    # Test with a non-existent configuration file
    config = "scripts/config/non_existent_config.yaml"
    with pytest.raises(FileNotFoundError):
        args, base_path, file_paths, seed = _setup_environment(config)
        net, scenarios = _prepare_network_and_scenarios(args, file_paths, seed)


def test_fail_prepare_network_and_scenarios_config(conf):
    """
    Tests if preparing network and scenarios fails with an invalid grid source in the configuration file
    """
    args, base_path, file_paths, seed = _setup_environment(conf)
    conf.network.source = "invalid_source"  # Set invalid source
    with pytest.raises(ValueError, match="Invalid grid source!"):
        net, scenarios = _prepare_network_and_scenarios(conf, file_paths, seed)


# Test save network function
def test_save_generated_data():
    """
    Tests if saving generated data works correctly by processing a single scenario
    and verifying that output files are created with correct structure.
    Uses config without perturbations to be sure scenarios converge
    """
    # Use config without perturbations to make sure we don't run into errors with perturbations
    config_path = "tests/config/default_without_perturbation_test.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    args = NestedNamespace(**cfg)
    worker = os.environ.get("PYTEST_XDIST_WORKER", "local")
    args.settings.data_dir = f"./tests/test_data_without_perturbation_{worker}"
    file_paths = generate_power_flow_data_distributed(args)
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
@pytest.fixture(scope="session", autouse=True)
def cleanup_generated_files():
    """
    Cleans up generated files after tests.
    This fixture runs after all tests in the module have completed.
    """
    yield  # This allows tests to run first

    # Only clean up directories that were actually created by these tests
    worker = os.environ.get("PYTEST_XDIST_WORKER", "local")
    cleanup_paths = [
        f"./tests/test_data_without_perturbation_{worker}",
        f"./tests/test_data_pf_mode_{worker}",
        f"./tests/test_data_opf_mode_{worker}",
    ]

    for path in cleanup_paths:
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
                print(f"Cleaned up: {path}")
        except Exception as e:
            print(f"Warning: Could not clean up {path}: {e}")


def test_setup_environment_overwrite_behavior():
    """Ensure _setup_environment respects the overwrite flag by deleting or preserving existing output.

    Steps:
    - Create a temp data_dir and initial base_path
    - Create a marker file inside base_path
    - Call _setup_environment with overwrite=False: marker should persist
    - Call _setup_environment with overwrite=True: marker should be removed
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "settings": {
                "data_dir": tmpdir,
                "overwrite": False,
                "include_dc_res": False,
                "enable_solver_logs": False,
                "pf_fast": False,
                "mode": "opf",
            },
            "network": {"name": "case24_ieee_rts"},
            # minimal required sections for downstream functions
            "load": {"generator": "agg_load_profile", "scenarios": 1},
            "topology_perturbation": {"type": "none"},
            "generation_perturbation": {"type": "none"},
            "admittance_perturbation": {"type": "none"},
        }

        # First setup creates directories
        args, base_path, file_paths, seed = _setup_environment(config)
        marker = os.path.join(base_path, "marker.txt")
        os.makedirs(base_path, exist_ok=True)
        with open(marker, "w") as f:
            f.write("keep or remove")

        # overwrite=False should keep existing contents
        config["settings"]["overwrite"] = False
        _args2, base_path2, _, _ = _setup_environment(config)
        assert base_path2 == base_path
        assert os.path.exists(marker), "Marker should persist when overwrite is False"

        # overwrite=True should remove and recreate directory
        config["settings"]["overwrite"] = True
        _args3, base_path3, _, _ = _setup_environment(config)
        assert base_path3 == base_path
        assert not os.path.exists(marker), (
            "Marker should be removed when overwrite is True"
        )


def test_parquet_append_vs_overwrite():
    """Verify that saves append rows when overwrite is False, and reset when overwrite is True.

    - Run generation with scenarios=1 -> record row counts
    - Run again with overwrite=False and scenarios=2 -> counts should increase
    - Run again with overwrite=True and scenarios=1 -> counts should reset to baseline
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Base config
        with open("tests/config/default_without_perturbation_test.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        args = NestedNamespace(**cfg)
        args.settings.data_dir = tmpdir

        # First run: scenarios=1
        args.settings.overwrite = True
        args.load.scenarios = 1
        args.load.generator = "powergraph"
        file_paths = generate_power_flow_data_distributed(args)

        df_bus_1 = pd.read_parquet(file_paths["bus_data"], engine="pyarrow")

        n_bus_1 = len(df_bus_1)

        # Second run: scenarios=2, overwrite=False -> should append
        args.settings.overwrite = False
        args.load.scenarios = 2
        file_paths = generate_power_flow_data_distributed(args)

        df_bus_2 = pd.read_parquet(file_paths["bus_data"], engine="pyarrow")

        assert len(df_bus_2) > n_bus_1, "Bus rows should increase when overwrite=False"

        # Third run: scenarios=1, overwrite=True -> should reset (<= baseline + small variance)
        args.settings.overwrite = True
        args.load.scenarios = 1
        file_paths = generate_power_flow_data_distributed(args)

        df_bus_3 = pd.read_parquet(file_paths["bus_data"], engine="pyarrow")

        assert len(df_bus_3) <= len(df_bus_2) - n_bus_1, (
            "Bus rows should reset when overwrite=True"
        )


def test_reproducibility_with_same_seed():
    """
    Test that generating data with the same seed produces identical results.
    Uses case14_ieee network with 10 scenarios and 2 processes.
    """
    # Load base config
    config_path = "tests/config/default_pf_mode_test.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Modify config for this test
    config["network"]["name"] = "case14_ieee"
    config["load"]["scenarios"] = 6
    config["settings"]["num_processes"] = 2
    config["settings"]["seed"] = 42  # Fixed seed for reproducibility
    config["settings"]["overwrite"] = True

    # Use temporary directories for both runs
    with (
        tempfile.TemporaryDirectory() as tmpdir1,
        tempfile.TemporaryDirectory() as tmpdir2,
    ):
        # First run
        config["settings"]["data_dir"] = tmpdir1
        config1 = NestedNamespace(**config)
        file_paths_1 = generate_power_flow_data_distributed(config1)

        # Second run with same seed
        config["settings"]["data_dir"] = tmpdir2
        config2 = NestedNamespace(**config)
        file_paths_2 = generate_power_flow_data_distributed(config2)

        # Load all generated parquet files and compare
        data_files = ["bus_data", "gen_data", "branch_data", "y_bus_data"]

        for data_type in data_files:
            df1 = pd.read_parquet(file_paths_1[data_type], engine="pyarrow")
            df2 = pd.read_parquet(file_paths_2[data_type], engine="pyarrow")

            # sort by load_scenario_idx
            df1 = df1.sort_values(by="load_scenario_idx", kind="stable").reset_index(
                drop=True,
            )
            df2 = df2.sort_values(by="load_scenario_idx", kind="stable").reset_index(
                drop=True,
            )

            # Check that shapes match
            assert df1.shape == df2.shape, (
                f"{data_type}: Shape mismatch - Run 1: {df1.shape}, Run 2: {df2.shape}"
            )

            # Check that columns match
            assert list(df1.columns) == list(df2.columns), (
                f"{data_type}: Column mismatch - Run 1: {df1.columns}, Run 2: {df2.columns}"
            )

            # Check that all values are identical
            # Use pandas testing to handle floating point comparison properly
            pd.testing.assert_frame_equal(
                df1,
                df2,
                check_dtype=True,
                check_exact=False,
                rtol=1e-10,
                atol=1e-10,
                obj=f"{data_type}",
            )

            print(
                f"   ✓ {data_type}: Both runs produced identical data ({df1.shape[0]} rows)",
            )

        # Also check the scenarios file
        scenarios_1 = pd.read_parquet(file_paths_1["scenarios"], engine="pyarrow")
        scenarios_2 = pd.read_parquet(file_paths_2["scenarios"], engine="pyarrow")

        os.makedirs("tests/test_data", exist_ok=True)

        pd.testing.assert_frame_equal(
            scenarios_1,
            scenarios_2,
            check_dtype=True,
            check_exact=False,
            rtol=1e-10,
            atol=1e-10,
        )

        print(
            f"   ✓ scenarios: Both runs produced identical data ({scenarios_1.shape[0]} rows)",
        )
        print("\n✓ All data files are identical across runs with the same seed!")


if __name__ == "__main__":
    test_reproducibility_with_same_seed()
