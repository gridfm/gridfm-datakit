import pytest
import yaml
import glob
from GridDataGen.utils.param_handler import (
    NestedNamespace,
    get_load_scenario_generator,
    initialize_generator,
)


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
