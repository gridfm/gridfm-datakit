"""
PowSyBl power flow results preprocessing module.

Format Powsybl power flow format into Power Model power flow format.
Compatible with the existing pipeline.
"""

from typing import Dict, Any
import pypowsybl as pp
from gridfm_datakit.network import Network as gfm_network


def preprocess_pp_pf_res(
    pp_net: "pp.network.Network",
    solve_time: float,
    pf_metadata: list,
    map_bus_p2g,
    map_branch_p2g,
    map_gen_p2g
) -> Dict[Any, Any]:
    """ Format pypowsybl power flow results for the pf_post_process function.

    Args:
        pp_net: PyPowSyBl network. It contains power flow results
        solve_time: power flow solving time
        pf_metadaa: power flow metadata
        map_bus_p2g: mapping of bus indexes from powsybl to gridfm
        map_branch_p2g: mapping of branch indexes from powsybl to gridfm
        map_gen_p2g: mapping of generator indexes from powsybl to gridfm

    Returns:
        Power flow results in a nested Dict format, similar to PowerModel's power flow results
    """
    # Retrieve computation status
    pf_status = pf_metadata[0].status_text
    
    # Check whether the power flow computation is susscefful and raise a error if not
    if not _is_power_flow_computed(pf_status):
        raise ValueError(f'Power flow computation failed. The returned power flow status:{pf_status}')

    # Store initial per-unit status
    initial_per_unit_status = pp_net.per_unit

    # Conform with PowerModel's pf results
    pp_net.per_unit=True

    pp_pf_res = {}
    # Filling the results
    pp_pf_res["solution"] = {
        "baseMVA": pp_net.nominal_apparent_power,
        'gen': _format_gens_res(pp_net, map_gen_p2g),
        'branch': _format_branch_res(pp_net, map_branch_p2g),
        'multiinfrastructure': None, # TODO check whether really not needed
        'multinetwork': None, # TODO check whether really not needed
        'bus': _format_buses_res(pp_net, map_bus_p2g),
        'per_unit': pp_net.per_unit,
        "pf": _is_power_flow_computed(pf_status)
        }
    pp_pf_res['solve_time'] = solve_time
    
    pp_net.per_unit = initial_per_unit_status

    # Adding slack to generator of the slack bus
    # This is to conform with the current implementation of the pf_post_processing function. 
    # Active power mismatch is counted separately
    pp_pf_res = _add_slack_results(pp_net, pp_pf_res, pf_metadata, map_gen_p2g)
    return pp_pf_res

def _format_gens_res(
    pp_net: "pp.network.Network",
    map_gen_p2g: Dict[str, float],
) -> Dict[str, float]:
    """ Format PowSyBl power flow results for generators.

    Args:
        pp_net: PowSyBl network. It contains power flow results.
        map_gen_p2g: Mapping of the generator indexes from PowSyBl to GridFM.

    Returns:
        Dict containing active and reactive injections for the generators.
    """
    gen_dict = {}
    df_gens_pp = pp_net.get_generators()
    for idx in df_gens_pp.index:
        if df_gens_pp.loc[idx, 'connected']:
            gen_dict[str(int(map_gen_p2g[idx]+1))] = {
                'pg': -float(df_gens_pp.loc[idx]['p']), # sign convention in PowSyBl, negative = injection
                'qg': -float(df_gens_pp.loc[idx]['q'])  # sign convention in PowSyBl, negative = injection
            }
    return gen_dict

def _format_branch_res(
    pp_net: "pp.network.Network",
    map_branch_p2g: Dict[Any, Any]
) -> Dict[str, float]:
    """ Format PowSyBl power flow results for the branches (lines and 2-windings transformers).

    Args:
        pp_net: PowSyBl network. It contains power flow results.
        map_gens_p2g: Mapping of the branch indexes from PowSyBl to GridFM.

    Returns:
        Dict containing active and reactive flows in both directions for each branch.
    """
    branch_dict = {}
    df_branches_pp = pp_net.get_branches()
    for idx in df_branches_pp.index:
        branch_dict[str(int(map_branch_p2g[idx]+1))] = {
            'pf': float(df_branches_pp.loc[idx]['p1']),
            'qf': float(df_branches_pp.loc[idx]['q1']),
            'pt': float(df_branches_pp.loc[idx]['p2']),
            'qt': float(df_branches_pp.loc[idx]['q2'])
        }
    return branch_dict

def _format_buses_res(
    pp_net: "pp.network.Network",
    map_bus_p2g: Dict[Any, Any],
) -> Dict[str, float]:
    """ Format PowSyBl power flow results for the buses.

    Args:
        pp_net: PowSyBl network. It contains power flow results.
        map_buses_p2g: Mapping of the bus indexes from PowSyBl to GridFM.

    Returns:
        Dict containing voltage magnitude and angle for each bus.
    """
    bus_dict = {}
    df_buses_pp = pp_net.get_buses()
    for idx in df_buses_pp.index:
        bus_dict[str(int(map_bus_p2g[idx]+1))] = {
            'vm': float(df_buses_pp.loc[idx]['v_mag']),
            'va': float(df_buses_pp.loc[idx]['v_angle'])
        }
    return bus_dict

def _is_power_flow_computed(pf_status: str):
    """ Check whether the power flow computation was successful."""
    return pf_status == 'Converged'

def _add_slack_results(pp_net, pf_res, pf_metadata, map_gen_p2g):
    """Add slack results to a generator attached to the slack bus."""

    slack_bus = pf_metadata[0].slack_bus_results[0].id
    slack_res = pf_metadata[0].slack_bus_results[0].active_power_mismatch
    
    df_gens = pp_net.get_generators()
    slack_gen_id = df_gens[df_gens['bus_id'] == slack_bus].index[0] # assigning slack results to the first generator attached to the slack bus.

    pf_res['solution']['gen'][str(int(map_gen_p2g[slack_gen_id]+1))]['pg'] += slack_res/pf_res['solution']['baseMVA']
    return pf_res
