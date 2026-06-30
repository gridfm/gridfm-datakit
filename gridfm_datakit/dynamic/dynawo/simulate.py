"""
Dynawo simulation functions.

Provides two public functions:

- compute_balanced_static_state_dynawo  : OPF -> update powsybl network -> AC-PF
- run_dynawo_simulation                 : run Dynawo via pypowsybl.dynamic using DynawoMappings and simulation parameters

Contains a helper function _format_dynamic_res that formats the raw Dynawo outputs obtained via pypowsybl.dynamic.
"""

from __future__ import annotations

import copy
import pypowsybl as pp
import time
from typing import Any, Dict, Tuple

from gridfm_datakit.dynamic import DynamicResults
from gridfm_datakit.dynamic.dynawo import DynawoMappings
from gridfm_datakit.network import Network
from gridfm_datakit import powsybl
from gridfm_datakit.process.solvers import run_opf
from gridfm_datakit.process.process_network import pf_preprocessing, pf_post_processing

# ---------------------------------------------------------------------------
# Public: balanced static state
# ---------------------------------------------------------------------------


def compute_balanced_static_state_dynawo(
        pp_net: pp.network.Network,
        gfm_net: Network,
        julia: Any,
        p2g_maps,
        scenario_index: int = 0,
) -> Tuple[Any, Dict[str, Any]]:
    """Compute the balanced initial conditions for a dynamic siulation.
    
    Runs the four-step sequence required to produce a consistent initial
    state for Dynawo:

    1. **OPF** via Julia/PowerModels on the randomised gfm network
       → optimal dispatch for the current scenario.
    2. **update_powsybl** (powsybl submodule)
       → applies OPF results (Pg, Vm setpoints) onto the pypowsybl object
       with correct per-unit conventions.
    3. **AC-PF** via pypowsybl OpenLoadFlow
       → verifies convergence and produces the balanced initial state.
    4. **get_pf_res / pf_post_processing** (powsybl submodule)
       → formats pypowsybl PF results in the gridfm column schema
       with ID-based bus index assignment.

    Args
    ----
    pp_net: 
        pypowsybl network. 
        The caller must pass a per-worker *clone/variant* to avoid
        cross-scenario contamination
    gfm_net:
        randomised gridfm network for the current scenario
        (with applied load scenario and perturbations)
    julia:
        Initialised Julia interface (from "init_julia")
    scenario_index: int
        Used to label the results row (matches ``pf_post_processing``'s
        ``scenario_index`` argument)
    
    Returns
    -------
    pp_net:
        The updated pypowsybl network, balanced and ready for dynamic
        simulation.
    pf_data: dict
        Power flow results in gridfm column schema with keys:
        ``"bus"``, ``"gen"``, ``"branch"``, ``"Y_bus"``, ``"runtime"``
    
    Raises
    ------
    RuntimeError
        If OPF fails to converge.
    ValueError
        If the AC power flow does not converge.
    """
    # Step 1: run OPF on the gfm network to get optimal dispatch
    opf_res = run_opf(gfm_net, julia)

    # Step 2: apply OPF setpoints to gfm_net, then push to pypowsybl
    gfm_net_pf = copy.deepcopy(gfm_net)
    gfm_net_pf = pf_preprocessing(gfm_net_pf, opf_res)

    # mapping_p2g = powsybl.build_p2g_maps(gfm_net_pf, pp_net); received as args to avoid repeated computation
    powsybl.update_powsybl(pp_net, gfm_net_pf, p2g_maps)
    
    # Step 3: run AC-PF via pypowsybl OpenLoadFlow
    lf_params = powsybl.get_default_lf_params()
    # TODO: delete once the fix merged and replace with default lf parameters
    ##############
    lf_params = pp.loadflow.Parameters(distributed_slack=False,
                                  read_slack_bus=True,
                                  write_slack_bus=True,
                                  provider_parameters={
                                      'slackBusSelectionMode': 'LARGEST_GENERATOR' # default: MOST_MESHED
                                  })
    ##############
    t0 = time.perf_counter()
    pf_metadata = powsybl.pypowsybl.loadflow.run_ac(pp_net, lf_params)
    solve_time = time.perf_counter() - t0

    # Step 4: format results in gridfm column schema (ID-based bus assignment)
    pf_res = powsybl.get_pf_res(pp_net, solve_time, pf_metadata, p2g_maps)
    pf_data = pf_post_processing(
        scenario_index,
        gfm_net_pf,
        pf_res,
        res_dc=None,
        include_dc_res=False,
    )

    return pp_net, pf_data

# ---------------------------------------------------------------------------
# Public: run Dynawo simulation
# ---------------------------------------------------------------------------

def run_dynawo_simulation(
        pp_net: pp.network.Network,
        dynawo_mapping: DynawoMappings,
        parameters: pp.dynamic.Parameters,
        drop_duplicate_timestep=True,
        ):
    """Apply Dynawo mappings to a balanced pypowsybl network and run the simulation.

    Args
    ----
    pp_net :
        Balanced pypowsybl network (output of ``compute_balanced_static_state_dynawo``).
    dynamic_mappings : DynawoMappings
        Validated Dynawo-ready mapping DataFrames (models, events, variables).
    parameters :
        ``pypowsybl.dynamic.Parameters`` object (from ``get_dynawo_simulation_parameters``).
    drop_duplicate_timestep :
        Whether drop duplicate timestep in the output timeseries, True by default.

    Returns
    -------
    DynamicResults
        Solver-agnostic container with a Zarr-shaped array
        ``(n_variables, n_timesteps)`` and the solver status report string.
    """
    # Setup
    sim = pp.dynamic.Simulation()
    report_node = pp.report.ReportNode()

    # Run simulation
    dyn_res = sim.run(
        pp_net,
        dynawo_mapping.dynamic_model_mapping,
        dynawo_mapping.event_mapping,
        dynawo_mapping.variable_mapping,
        parameters=parameters,
        report_node=report_node)
    
    # Format results
    formated_dyn_res = _format_dynamic_res(dyn_res, drop_duplicate_timestep)

    return DynamicResults(formated_dyn_res, report_node.to_json())


# ---------------------------------------------------------------------------
# Private: format raw Dynawo output
# ---------------------------------------------------------------------------


def _format_dynamic_res(dyn_res: Any, drop_duplicate_timestpe):
    """Convert raw pypowsybl.dynamic simulation output into DynamicResults.

    Dynawo would compute a timestep several times upon system changes 
    and keeps tracks of values that it associates to the same timestep.
    By toggling drop_duplicate_timestep only the last timestep is kept.
    
    Args
    ----
    sim_result :
        Raw object returned by ``pypowsybl.dynamic.run_simulation``.
    dynamic_mappings : DynawoMappings
        Used to determine expected variable order.

    Returns
    -------
    DynamicResults
    """
    # TODO: validate with Youssouf, following code didn't run on my side
    # curves = dyn_res.curves()

    # if curves is None or (hasattr(curves, "empty") and curves.empty):
    #    # No time-series data (simulation failed or no variables requested)
    #    data_array = np.empty((0, 0), dtype=np.float64)

    # variable_cols = list(curves.columns)
    # data_array = curves[variable_cols].to_numpy(dtype=np.float64).T
    # shape: (n_variables, n_timesteps)

    # # Wrap in a Zarr in-memory array so the caller can write it to a store
    # store = zarr.MemoryStore()
    # root = zarr.open_group(store, mode="w")
    # root.create_dataset(
    #     "curves",
    #     data=data_array,
    #     chunks=(data_array.shape[0], min(data_array.shape[1], 1000))
    #     if data_array.size > 0
    #     else True,
    #     dtype="float64",
    #     compressor=None,  # compression applied when writing to persistent store
    # )

    # return root['curves']
    df = dyn_res.curves()
    if drop_duplicate_timestpe:
        return df[~df.index.duplicated(keep='last')]
    return df
