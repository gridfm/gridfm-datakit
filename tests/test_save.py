"""
Test cases for save functions in GridDataGen.save module
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

def test_save_functions(conf):
        if conf.network.source == "pandapower":
            net = load_net_from_pp(conf.network.name)
        elif conf.network.source == "pglib":
            net = load_net_from_pglib(conf.network.name)
        elif conf.network.source == "file":
            net = load_net_from_file(conf.network.name)
        else:
            raise ValueError("Invalid grid source!")
        # Save the network data
        save_edge_params(net, "tests/test_data/test_edge_params.csv")
        save_bus_params(net, "tests/test_data/test_bus_params.csv")
        # Sample list of branch_idx_removed
        branch_idx_removed = [[], [], [17], [36], [], [30], [28], [2], [14], [], [], [], [], [], [20], [], [], [28], [], [12], [28], [30], [], [], [], [24], [19], [], [37], [], [], [31], [], [23], [], [], [], [2], [14], [], [], [17], [], [], [10], [19], [], [23], [7], [1], [24], [36], [], [], [], [34], [], [], [22], [], [], [32], [29], [], [], [37], [], [26], [], [], [28], [25], [34], [17], [], [22], [], [31], [2], [], [12], [], [], [], [], [], [7], [], [], [1], [], [36], [], [], [28], [], [], [28], [22], [], [], [10], [35], [13], [14], [14], [], [], [], [], [], [], [], [32], [5], [], [], [], [23], [20], [37], [30], [29], [], [], [], [], [], [27], [35], [30], [], [], [2], [], [], [26], [], [], [27], [35], [], [], [], [], [27], [], [], [], [], [], [], [2], [2], [14], [], [15], [], [], [32], [], [17], [31], [], [], [], [], [17], [], [27], [], [], [], [], [], [], [25], [26], [2], [], [], [], [], [2], [26], [29], [29], [], [15], [14], [], [12], [], [14], [], [0], [], [], [37], [], [30], [], [18], [], [19], [27], [], [30], [], [], [], [36], [22], [], [20], [], [], [], [18], [37], [], [], [10], [], [13], [0], [], [], [], [], [2], []]
        save_branch_idx_removed(branch_idx_removed, "tests/test_data/test_branch_idx_removed.csv")
        #save_node_edge_data -- need to create a mock network with node and edge data