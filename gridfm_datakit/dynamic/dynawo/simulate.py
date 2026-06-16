"""Run dynamic simulations using dynawo."""
import copy
import pypowsybl as pp
import time
from typing import Any

from gridfm_datakit.dynamic import DynamicResults
from gridfm_datakit.dynamic.dynawo import DynawoMappings
from gridfm_datakit.network import Network
from gridfm_datakit.powsybl import update_powsybl, get_default_lf_params, get_pf_res, MappingP2G
from gridfm_datakit.process.solvers import run_opf
from gridfm_datakit.process.process_network import pf_preprocessing


def run_dynawo_simulation(
        pp_net: pp.network.Network,
        dynawo_mapping: DynawoMappings,
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
    report_node = pp.report.ReportNode()
    dyn_res = sim.run(
        pp_net,
        dynawo_mapping.dynamic_model_mapping,
        dynawo_mapping.event_mapping,
        dynawo_mapping.variable_mapping,
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
        mapping_p2g: MappingP2G,
        jl):
    
    res = run_opf(perturbed_network, jl)

    # create an updated network
    net_pf = copy.deepcopy(perturbed_network)
    net_pf = pf_preprocessing(net_pf, res)

    # convert to powsybl network, keep the mapping for formatting purpose
    # TODO: replace by update_powsybl
    update_powsybl(pp_net, perturbed_network, mapping_p2g)
    
    # a powsybl network is necessary to run dynawo
    # run powsybl pf to reach a new balanced state after conversion
    lf_param = get_default_lf_params()
    start_time = time.perf_counter()
    pf_metadata = pp.loadflow.run_ac(pp_net, lf_param)
    end_time = time.perf_counter()
    solve_time = end_time - start_time
    res = get_pf_res(
        pp_net,
        solve_time,
        pf_metadata,
        mapping_p2g
        )
    return pp_net, res
