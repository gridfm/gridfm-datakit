import pytest
from gridfm_datakit.powsybl.api import is_powsybl_available

pytestmark = pytest.mark.skipif(
    is_powsybl_available() is False,
    reason="pypowsybl is not installed. Install with: pip install gridfm-datakit[powsybl]",
)

import pypowsybl as pp

# TODO: add basic tests on top of the benchmarking 

def test_benchmark_get_dynawo_simulation_parameters(config_ieee14,
                                                    pp_net_ieee14,
                                                    model_mapping_ieee14,
                                                    event_mapping_ieee14,
                                                    variable_mapping_ieee14,
                                                    df_ref_curves_ieee14,
                                                    ):
    from gridfm_datakit.generate import _setup_environment
    from gridfm_datakit.dynamic.dynawo import get_dynawo_simulation_parameters

    args, _, _, _ = _setup_environment(config_ieee14)

    simulation_parameters = get_dynawo_simulation_parameters(args)
    
    sim = pp.dynamic.Simulation()
    report_node = pp.report.ReportNode()
    res = sim.run(
        network=pp_net_ieee14,
        model_mapping=model_mapping_ieee14,
        event_mapping=event_mapping_ieee14,
        timeseries_mapping=variable_mapping_ieee14,
        parameters= simulation_parameters,
        report_node=report_node,
    )
    assert _validate_output_curves_against_ref(res, df_ref_curves_ieee14)

def _validate_output_curves_against_ref(res, df_ref):
    df_res = res.curves().reset_index(drop=True).rename(columns={'_GEN____1_SM_generator_efdPu_value': 'GEN____1_SM_generator_efdPu_value',
                                                                '_GEN____1_SM_voltageRegulator_EfdMaxPu': 'GEN____1_SM_voltageRegulator_EfdMaxPu',
                                                                '_GEN____3_SM_generator_UPu':'GEN____3_SM_generator_UPu',
                                                                '_GEN____3_SM_generator_efdPu_value': 'GEN____3_SM_generator_efdPu_value',
                                                                '_GEN____3_SM_voltageRegulator_EfdMaxPu': 'GEN____3_SM_voltageRegulator_EfdMaxPu'
                                                                })
    df_ref = df_ref.reset_index(drop=True)
    df_ref = df_ref[df_res.columns]
    return df_res.equals(df_ref)