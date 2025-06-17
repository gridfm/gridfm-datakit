import pytest
import yaml
import glob
from GridDataGen.save import *
from GridDataGen.process.process_network import *
from GridDataGen.utils.config import *
from GridDataGen.utils.stats import *
from GridDataGen.utils.param_handler import *
from GridDataGen.network import *
from GridDataGen.perturbations.load_perturbation import *
from pandapower.auxiliary import pandapowerNet
from GridDataGen.utils.param_handler import initialize_generator
from GridDataGen.utils.utils import write_ram_usage_distributed, Tee
from GridDataGen.perturbations.topology_perturbation import TopologyGenerator
from GridDataGen.perturbations.load_perturbation import LoadScenarioGeneratorBase

@pytest.mark.parametrize("yaml_path", glob.glob("scripts/config/*.yaml"))
def test_yaml_config_valid(yaml_path):
    with open(yaml_path) as f:
        config_dict = yaml.safe_load(f)

    args = NestedNamespace(**config_dict)
    # Call your param handler functions; they should not raise exceptions
    if hasattr(args, "agg_load_profile"):
        get_load_scenario_generator(args)
    if hasattr(args, "n_minus_k"):
        initialize_generator(args, args.base_net)