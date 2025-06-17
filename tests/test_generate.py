"""
Test cases for genertaing data from GridDataGen.generate module
pytest tests/ --config scripts/config/default.yaml 
"""
import pytest
import numpy as np
import os
from GridDataGen.save import *
from GridDataGen.process.process_network import *
from GridDataGen.utils.config import *
from GridDataGen.utils.stats import *
from GridDataGen.utils.param_handler import *
from GridDataGen.network import *
from GridDataGen.perturbations.load_perturbation import *
from pandapower.auxiliary import pandapowerNet
import gc
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, Manager
from multiprocessing import Queue
from GridDataGen.utils.param_handler import initialize_generator
import shutil
from GridDataGen.utils.utils import write_ram_usage_distributed, Tee
import yaml
from typing import List, Tuple, Any, Dict, Optional, Union
from GridDataGen.perturbations.topology_perturbation import TopologyGenerator
from GridDataGen.perturbations.load_perturbation import LoadScenarioGeneratorBase
from GridDataGen.generate import _setup_environment, _prepare_network_and_scenarios, generate_power_flow_data, generate_power_flow_data_distributed
import sys

# Test set up environment function
def test_setup_environment(conf):
    args, base_path, file_paths = _setup_environment(conf)
    assert isinstance(file_paths, dict), "File paths should be a dictionary"
    assert 'edge_data' in file_paths, "Network file path should be in the dictionary"
    assert os.path.exists(base_path), "Base path should exist"

def test_fail_setup_environment():
    conf = 'scripts/config/non_existent_config.yaml'
    # Test with a non-existent configuration file
    with pytest.raises(FileNotFoundError):
        args, base_path, file_paths = _setup_environment('non_existent_config.yaml')

# Test prepare network and scenarios function
def test_prepare_network_and_scenarios(conf):
    args, base_path, file_paths = _setup_environment(conf)
    net, scenarios = _prepare_network_and_scenarios(args, file_paths)
    
    assert isinstance(net, pandapowerNet), "Network should be a pandapowerNet object"
    #assert isinstance(scenarios, list), "Scenarios should be a list"
    assert len(scenarios) > 0, "There should be at least one scenario"
    
    # Check if the network has been loaded correctly
    assert 'bus' in net.keys(), "Network should contain bus data"
    assert 'line' in net.keys(), "Network should contain line data"

# Test generate pf data function
def test_generate_pf_data():
    config = 'scripts/config/default.yaml'
    file_paths = generate_power_flow_data(config)

# Test generate pf data distributed function
def test_generate_pf_data_distributed():
    config = 'scripts/config/default.yaml'
    file_paths = generate_power_flow_data_distributed(config)