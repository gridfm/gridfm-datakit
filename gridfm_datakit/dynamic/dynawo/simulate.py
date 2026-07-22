"""
Dynawo simulation functions.

Provides two public functions:

- compute_balanced_static_state_dynawo  : OPF -> update powsybl network -> AC-PF
- run_dynawo_simulation                 : run Dynawo via pypowsybl.dynamic using DynawoMappings and simulation parameters

Contains a helper function _format_dynamic_res that formats the raw Dynawo outputs obtained via pypowsybl.dynamic.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, Tuple

# pypowsybl is an optional dependency; guard the import so this module loads
# without it. Type hints are lazy strings (`from __future__ import annotations`)
# and every runtime use of `pp` is inside a function that runs only with
# pypowsybl present.
try:
    import pypowsybl as pp
except ImportError:  # pragma: no cover - optional dependency
    pp = None

from gridfm_datakit.dynamic import DynamicResults
from gridfm_datakit.dynamic.dynawo import DynawoMappings
from gridfm_datakit.network import Network
from gridfm_datakit import powsybl
from gridfm_datakit.process.solvers import run_opf
from gridfm_datakit.process.process_network import pf_preprocessing, pf_post_processing
from gridfm_datakit.process.solver_output import solver_capture
from gridfm_datakit.utils.param_handler import NestedNamespace

# ---------------------------------------------------------------------------
# Public: balanced static state
# ---------------------------------------------------------------------------


def compute_balanced_static_state_dynawo(
    pp_net: pp.network.Network,
    gfm_net: Network,
    julia: Any,
    p2g_maps,
    scenario_index: int = 0,
    lf_params: Any = None,
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
    lf_params:
        ``pypowsybl.loadflow.Parameters`` for step 3 (from
        ``get_dynawo_loadflow_parameters``). Defaults to the dynamic-appropriate
        settings in ``LOADFLOW_PARAMETERS_DEFAULTS`` when None.

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

    # Step 3: run AC-PF via pypowsybl OpenLoadFlow.
    # These parameters deliberately differ from the static pipeline's
    # get_default_lf_params(): Dynawo initialises its synchronous machines from
    # this solution, so the slack has to sit on a machine that carries a dynamic
    # model. See get_dynawo_loadflow_parameters.
    if lf_params is None:
        from gridfm_datakit.dynamic.dynawo import get_dynawo_loadflow_parameters

        lf_params = get_dynawo_loadflow_parameters(NestedNamespace())
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
        Solver-agnostic container holding the curves as a pandas DataFrame
        indexed by time — shape **(n_timesteps, n_variables)** — plus the solver
        status report string and the final state values. The transpose to
        (n_variables, n_timesteps) happens only when writing the Zarr store.
    """
    # Setup
    sim = pp.dynamic.Simulation()
    report_node = pp.report.ReportNode()

    # Run simulation. Dynawo writes prolific native output (OpenModelica banners,
    # solver iterations) straight to fd 1/2; route it through the process-wide
    # log router's "dynawo" channel so it obeys the same verbosity/file policy as
    # the OPF/PF solvers. No-op when no router is installed (e.g. ad-hoc scripts).
    with solver_capture("dynawo"):
        dyn_res = sim.run(
            pp_net,
            dynawo_mapping.dynamic_model_mapping,
            dynawo_mapping.event_mapping,
            dynawo_mapping.variable_mapping,
            parameters=parameters,
            report_node=report_node,
        )

    # Dynawo does NOT raise when the simulation fails: sim.run() returns a result
    # whose status is FAILURE and whose curves are empty or truncated. Left
    # unchecked, a diverged/aborted run would be stored as a valid trajectory (or
    # blow up the Zarr writer with a zero-width array). Raise instead: the caller
    # logs it and drops the sample, exactly like an OPF/PF divergence.
    if dyn_res.status().name != "SUCCESS":
        raise RuntimeError(
            f"Dynawo simulation failed ({dyn_res.status().name}): {dyn_res.status_text()}",
        )

    # Format results. FinalStateValue rows of the variables table are returned
    # separately from the curves and stored as a per-sample scalar table, so they
    # are carried through here rather than dropped.
    formated_dyn_res = _format_dynamic_res(dyn_res, drop_duplicate_timestep)

    return DynamicResults(
        formated_dyn_res,
        report_node.to_json(),
        final_state_values=dyn_res.final_state_values(),
    )


# ---------------------------------------------------------------------------
# Private: format raw Dynawo output
# ---------------------------------------------------------------------------


def _format_dynamic_res(dyn_res: Any, drop_duplicate_timestep):
    """Convert raw pypowsybl.dynamic simulation output into a curves DataFrame.

    Dynawo may compute a given timestep several times upon system changes and
    keeps track of every value it associates to that same timestep. When
    ``drop_duplicate_timestep`` is True only the last value per timestep is kept.

    Args
    ----
    dyn_res :
        Raw object returned by ``pypowsybl.dynamic.Simulation.run``.
    drop_duplicate_timestep : bool
        Whether to keep only the last value for each duplicated timestep.

    Returns
    -------
    pandas.DataFrame
        Curves indexed by time, shape (n_timesteps, n_variables).
    """
    df = dyn_res.curves()
    if drop_duplicate_timestep:
        return df[~df.index.duplicated(keep="last")]
    return df
