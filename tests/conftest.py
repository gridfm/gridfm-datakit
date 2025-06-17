"""
Configuration for pytest fixtures and command line options.
To run: pytest tests/test_save.py --config scripts/config/default.yaml
"""
import pytest
import yaml
from GridDataGen.save import *
from GridDataGen.process.process_network import *
from GridDataGen.utils.config import *
from GridDataGen.utils.stats import *
from GridDataGen.utils.param_handler import *
from GridDataGen.network import *
from GridDataGen.perturbations.load_perturbation import *
from GridDataGen.perturbations.topology_perturbation import TopologyGenerator

def pytest_addoption(parser):
    parser.addoption("--config", action='store', default='scripts/config/default.yaml', help="path to the config file")

@pytest.fixture
def conf(request):
    path = request.config.getoption("--config")
    with open(path, 'r') as f:
        base_config = yaml.safe_load(f)
        args = NestedNamespace(**base_config)
    return args