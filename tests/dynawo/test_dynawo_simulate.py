import pytest
from gridfm_datakit.powsybl.api import is_powsybl_available

pytestmark = pytest.mark.skipif(
    is_powsybl_available() is False,
    reason="pypowsybl is not installed. Install with: pip install gridfm-datakit[powsybl]",
)


def test_benchmark_ieee14_run_dynawo_simulation(pp_net_ieee14,
                                                model_mapping_ieee14,
                                                event_mapping_ieee14,
                                                variable_mapping_ieee14,
                                                param_ieee14,
                                                df_ref_curves_ieee14,
                                                ):
    from gridfm_datakit.dynamic.dynawo.simulate import run_dynawo_simulation
    from gridfm_datakit.dynamic.dynawo import DynawoMappings

    dynamic_results = run_dynawo_simulation(
        pp_net=pp_net_ieee14,
        dynawo_mapping=DynawoMappings(
            dynamic_model_mapping=model_mapping_ieee14,
            event_mapping=event_mapping_ieee14,
            variable_mapping=variable_mapping_ieee14,
            ),
        parameters=param_ieee14
        )
    assert _validate_output_curves_against_ref(dynamic_results.dynamic_results, df_ref_curves_ieee14)

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