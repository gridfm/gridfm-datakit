import pytest
import yaml
import glob
from gridfm_datakit.utils.param_handler import (
    NestedNamespace,
    get_load_scenario_generator,
    initialize_generator,
)
from gridfm_datakit.generate import generate_power_flow_data_distributed
import shutil


@pytest.mark.parametrize("yaml_path", glob.glob("scripts/config/*.yaml"))
def test_yaml_config_valid(yaml_path):
    """
    Tests if all YAML configuration files in the scripts/config directory can be loaded without errors.
    This ensures that the configurations are valid and can be parsed correctly.
    """
    with open(yaml_path) as f:
        config_dict = yaml.safe_load(f)
    args = NestedNamespace(**config_dict)
    # Call param handler functions; should not raise exceptions
    if hasattr(args, "agg_load_profile"):
        get_load_scenario_generator(args)
    if hasattr(args, "n_minus_k"):
        initialize_generator(args, args.base_net)


@pytest.mark.parametrize("yaml_path", glob.glob("scripts/config/*.yaml"))
def test_yaml_config_can_run(yaml_path):
    """
    Tests if all YAML configuration files in the scripts/config directory can be run without errors.
    This ensures that we can generate data from all configurations.
    """
    with open(yaml_path) as f:
        config_dict = yaml.safe_load(f)
    args = NestedNamespace(**config_dict)
    args.load.scenarios = 2
    args.settings.large_chunk_size = 2
    args.settings.num_processes = 2
    args.settings.data_dir = "./data_pytest_tmp"
    file_paths = generate_power_flow_data_distributed(args)
    assert file_paths is not None
    # delete the data_pytest_tmp directory
    shutil.rmtree("./data_pytest_tmp")
