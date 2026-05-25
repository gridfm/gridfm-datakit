"""Run dynamic simulations using dynawo."""
import copy
import pypowsybl as pp
import time
from typing import Any

from gridfm_datakit.dynamic import DynamicResults
from gridfm_datakit.dynamic.dynawo import DynawoMappings
from gridfm_datakit.network import Network
from gridfm_datakit.powsybl import to_powsybl
from gridfm_datakit.powsybl.preprocess_pf_res import preprocess_pp_pf_res
from gridfm_datakit.powsybl.utils.lf_parameters import get_default_lf_parameters
from gridfm_datakit.process.solvers import run_opf
from gridfm_datakit.process.process_network import pf_preprocessing


def run_dynawo_simulation(
        pp_net: pp.network.Network,
        dynamic_mappings: DynawoMappings,
        parameters: Any, #TODO assign pypowsybl.dynamic.Parameters() ?
        ):
    """
    Runs the dynamic simulation.
    Args:
        network
        dynamic_mappings
        parameters
    
    Returns:
        dynamic_res
        report_node
    """
    sim = pp.dynamic.Simulation()
    report_node = pp.report.Reporter()
    dyn_res = sim.run(
        pp_net,
        dynamic_mappings.dynamic_model_mapping,
        dynamic_mappings.event_mapping,
        dynamic_mappings.variable_mapping,
        parameters=parameters,
        report_node=report_node)
    formated_dyn_res = _format_dynamic_res(dyn_res)
    return DynamicResults(formated_dyn_res, report_node)

def _format_dynamic_res(dyn_res):
    """Format dynamic results."""
    # TODO: define the outputs format needed for graph-kit
    return dyn_res

def compute_balanced_static_state_dynawo(
        pp_net: pp.network.Network,
        perturbed_network: Network,
        jl):
    
    res = run_opf(perturbed_network, jl)

    # create an updated network
    net_pf = copy.deepcopy(perturbed_network)
    net_pf = pf_preprocessing(net_pf, res)

    # convert to powsybl network, keep the mapping for formatting purpose
    # TODO: replace by update_powsybl
    pp_net = update_powsybl(pp_net, perturbed_network)
    _conv = to_powsybl(net_pf)
    
    # get the mappings
    _, map_bus_p2g, map_branch_p2g, map_gen_p2g = _conv.pp_net, _conv.map_bus_p2g, _conv.map_branch_p2g, _conv.map_gen_p2g
    
    # a powsybl network is necessary to run dynawo
    # run powsybl pf to reach a new balanced state after conversion
    lf_param = get_default_lf_parameters()
    start_time = time.perf_counter()
    pf_metadata = pp.loadflow.run_ac(pp_net, lf_param)
    end_time = time.perf_counter()
    solve_time = end_time - start_time
    res = preprocess_pp_pf_res(
        pp_net,
        solve_time,
        pf_metadata,
        map_bus_p2g,
        map_branch_p2g,
        map_gen_p2g)
    return pp_net, res
