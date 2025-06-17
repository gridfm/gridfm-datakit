"""
Test cases for loading network from config in GridDataGen.network module
pytest tests/ --config scripts/config/default.yaml 
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
import glob

@pytest.mark.parametrize("yaml_path", glob.glob("tests/config/*.yaml"))
def test_load_config(yaml_path):
    """
    Test loading configuration from a YAML file.
    """
    with open(yaml_path, 'r') as f:
        base_config = yaml.safe_load(f)
        args = NestedNamespace(**base_config)
    
    # Check if the configuration is loaded correctly
    assert args.network.name is not None, "Network name should not be None"
    assert args.network.source in ["pandapower", "pglib", "file"], "Network source should be one of ['pandapower', 'pglib', 'file']"

@pytest.fixture
def conf():
    path = "scripts/config/default.yaml"  # Default path to the config file
    with open(path, 'r') as f:
        base_config = yaml.safe_load(f)
        args = NestedNamespace(**base_config)
    return args

# Load from configuration - pglib
def test_load_network_from_config(conf):
    if conf.network.source == "pandapower": # Source is set to pandapower in default config
        load_net_from_pp(conf.network.name)
    elif conf.network.source == "pglib":
        load_net_from_pglib(conf.network.name)
    elif conf.network.source == "file":
        load_net_from_file(conf.network.name)
    else:
        raise ValueError("Invalid grid source!")
    
# Load from pandapower
def test_load_network_from_pp(conf):
    conf.network.source = "pandapower" # Set the source to pandapower
    if conf.network.source == "pandapower":
        load_net_from_pp(conf.network.name)
    elif conf.network.source == "pglib":
        load_net_from_pglib(conf.network.name)
    elif conf.network.source == "file":
        load_net_from_file(conf.network.name)
    else:
        raise ValueError("Invalid grid source!")

# Load from file
def test_load_network_from_file(conf):
    conf.network.source = "file" # Set the source to file
    network_path = "GridDataGen/grids/pglib_opf_case24_ieee_rts.m" # Requires path to grid file with .m extension
    if conf.network.source == "pandapower":
        load_net_from_pp(conf.network.name)
    elif conf.network.source == "pglib":
        load_net_from_pglib(conf.network.name)
    elif conf.network.source == "file":
        load_net_from_file(network_path)
    else:
        raise ValueError("Invalid grid source!")
    
#### Fail case - Load from config with invalid source
def test_fail_load_network_from_config(conf):
    conf.network.name = "invalid_network"  # Set an invalid network name
    conf.network.source = "invalid_source"  # Set an invalid source
    with pytest.raises(ValueError, match="Invalid grid source!"):
        if conf.network.source == "pandapower":
            load_net_from_pp(conf.network.name)
        elif conf.network.source == "pglib":
            load_net_from_pglib(conf.network.name)
        elif conf.network.source == "file":
            load_net_from_file(conf.network.name)
        else:
            raise ValueError("Invalid grid source!")
        
def test_fail_load_network_from_pp(conf):
    conf.network.name = "invalid_network"  # Set an invalid network name
    conf.network.source = "pandapower"  # Set an invalid source
    with pytest.raises(AttributeError, match="Invalid grid source!"):
        if conf.network.source == "pandapower":
            try:
                load_net_from_pp(conf.network.name)
            except AttributeError:
                raise AttributeError("Invalid grid source!")
            
def test_fail_load_network_from_file(conf):
    conf.network.source = "file" # Set the source to file
    network_path = "invalid/path/invalid_file.csv" # Invalid path
    with pytest.raises(ValueError, match="Invalid grid source!"):
        if conf.network.source == "file":
            try:
                load_net_from_file(network_path)
            except UnboundLocalError:
                raise ValueError("Invalid grid source!")

