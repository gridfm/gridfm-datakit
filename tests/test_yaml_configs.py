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
from concurrent.futures import ProcessPoolExecutor


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


excluded_files = [
    "scripts/config/Texas2k_case1_2016summerpeak.yaml",
    "scripts/config/case1354_pegase.yaml",
]

yaml_files = [f for f in glob.glob("scripts/config/*.yaml") if f not in excluded_files]


def run_generation(yaml_path):
    try:
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        args = NestedNamespace(**config_dict)
        args.load.scenarios = 2
        args.settings.large_chunk_size = 2
        args.settings.num_processes = 2
        args.settings.data_dir = (
            f"./data_pytest_tmp/{yaml_path.split('/')[-1].replace('.yaml', '')}"
        )
        file_paths = generate_power_flow_data_distributed(args)
        assert file_paths is not None
        shutil.rmtree(args.settings.data_dir)
        return yaml_path, "OK"
    except Exception as e:
        return yaml_path, f"FAIL: {e}"


def test_all_yaml_configs_in_parallel():
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_generation, yaml_files))

    for yaml_path, status in results:
        print(f"{yaml_path}: {status}")
        assert status == "OK"
