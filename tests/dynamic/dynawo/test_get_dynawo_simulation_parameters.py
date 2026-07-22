from pathlib import Path

import pytest

# pypowsybl is an optional dependency; guard the import so collection never
# fails when it is absent (the tests below are skipped anyway).
try:
    import pypowsybl as pp
except ImportError:  # pragma: no cover - optional dependency
    pp = None

from gridfm_datakit.dynamic.dynawo.api import is_dynawo_available
from gridfm_datakit.utils.param_handler import NestedNamespace

# Calls Simulation.run(), so it needs a local Dynawo installation on top of pypowsybl.
pytestmark = pytest.mark.skipif(
    is_dynawo_available() is False,
    reason="Dynawo backend unavailable (needs pypowsybl + a local Dynawo installation)",
)

# Coverage note: this module only benchmarks a full parameter set end-to-end
# against reference curves. Unit-level checks on the config block (missing and
# unsupported keys) live in tests/dynamic/test_solver_parameters.py, which needs
# no Dynawo installation.


@pytest.fixture(scope="module")
def config():
    parameter_file_path = (
        Path(__file__).parent
        / "benchmark_data/ieee14/ieee14_GeneratorDisconnections/IEEE14.par"
    )
    return NestedNamespace(
        dynamic=NestedNamespace(
            solver_parameters=NestedNamespace(
                start_time=0.0,
                stop_time=500.0,
                parameters_file=parameter_file_path,
                network_parameters_file=parameter_file_path,
                network_parameters_id="Network",
                solver_type="SIM",
                solver_parameters_file=parameter_file_path,
                solver_parameters_id="SimplifiedSolver",
            ),
        ),
    )


def test_benchmark_get_dynawo_simulation_parameters(
    config,
    pp_net_ieee14,
    model_mapping_ieee14,
    event_mapping_ieee14,
    variable_mapping_ieee14,
    df_ref_curves_ieee14,
):
    from gridfm_datakit.dynamic.dynawo import get_dynawo_simulation_parameters

    simulation_parameters = get_dynawo_simulation_parameters(config)

    sim = pp.dynamic.Simulation()
    report_node = pp.report.ReportNode()
    res = sim.run(
        network=pp_net_ieee14,
        model_mapping=model_mapping_ieee14,
        event_mapping=event_mapping_ieee14,
        timeseries_mapping=variable_mapping_ieee14,
        parameters=simulation_parameters,
        report_node=report_node,
    )
    assert _validate_output_curves_against_ref(res, df_ref_curves_ieee14)


def _validate_output_curves_against_ref(res, df_ref):
    df_res = (
        res.curves()
        .reset_index(drop=True)
        .rename(
            columns={
                "_GEN____1_SM_generator_efdPu_value": "GEN____1_SM_generator_efdPu_value",
                "_GEN____1_SM_voltageRegulator_EfdMaxPu": "GEN____1_SM_voltageRegulator_EfdMaxPu",
                "_GEN____3_SM_generator_UPu": "GEN____3_SM_generator_UPu",
                "_GEN____3_SM_generator_efdPu_value": "GEN____3_SM_generator_efdPu_value",
                "_GEN____3_SM_voltageRegulator_EfdMaxPu": "GEN____3_SM_voltageRegulator_EfdMaxPu",
            },
        )
    )
    df_ref = df_ref.reset_index(drop=True)
    df_ref = df_ref[df_res.columns]
    return df_res.equals(df_ref)
