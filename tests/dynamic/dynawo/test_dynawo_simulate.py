import pytest
from gridfm_datakit.powsybl.api import is_powsybl_available

pytestmark = pytest.mark.skipif(
    is_powsybl_available() is False,
    reason="pypowsybl is not installed. Install with: pip install gridfm-datakit[powsybl]",
)

import pypowsybl as pp

@pytest.fixture(scope='session')
def julia():
    from gridfm_datakit.generate import init_julia
    return init_julia(200)

@pytest.fixture(scope='function')
def pp_net_ieee9():
    return pp.network.create_ieee9()

def test_compute_balanced_static_state_dynawo_output_formats(
        pp_net_ieee9,
        julia
        ):
    from gridfm_datakit.dynamic.dynawo.simulate import compute_balanced_static_state_dynawo
    from gridfm_datakit.powsybl import from_powsybl, build_p2g_maps
    gfm_net = from_powsybl(pp_net_ieee9)
    p2g_maps = build_p2g_maps(gfm_net, pp_net_ieee9)
    pp_net, pf_data = compute_balanced_static_state_dynawo(pp_net_ieee9, from_powsybl(pp_net_ieee9), julia, p2g_maps,)
    assert type(pp_net) == pp.network.Network and type(pf_data) == dict

# Test run_dynawo_simulation using ieee14 benchmark

def test_benchmark_ieee14_run_dynawo_simulation(pp_net_ieee14,
                                                model_mapping_ieee14,
                                                event_mapping_ieee14,
                                                variable_mapping_ieee14,
                                                param_ieee14,
                                                df_ref_curves_ieee14,
                                                ):
    from gridfm_datakit.dynamic.dynawo.simulate import run_dynawo_simulation
    from gridfm_datakit.dynamic.dynawo import DynawoMappings

    drop_duplicate_timestep = True
    dynamic_results = run_dynawo_simulation(
        pp_net=pp_net_ieee14,
        dynawo_mapping=DynawoMappings(
            dynamic_model_mapping=model_mapping_ieee14,
            event_mapping=event_mapping_ieee14,
            variable_mapping=variable_mapping_ieee14,
            ),
        parameters=param_ieee14,
        drop_duplicate_timestep=drop_duplicate_timestep,
        )
    assert _validate_res_against_ref(dynamic_results, df_ref_curves_ieee14, drop_duplicate_timestep)

def _validate_res_against_ref(res, df_ref, drop_duplicate_timestep):
    df_res = res.dynamic_results.reset_index(drop=True).rename(columns={'_GEN____1_SM_generator_efdPu_value': 'GEN____1_SM_generator_efdPu_value',
                                                            '_GEN____1_SM_voltageRegulator_EfdMaxPu': 'GEN____1_SM_voltageRegulator_EfdMaxPu',
                                                            '_GEN____3_SM_generator_UPu':'GEN____3_SM_generator_UPu',
                                                            '_GEN____3_SM_generator_efdPu_value': 'GEN____3_SM_generator_efdPu_value',
                                                            '_GEN____3_SM_voltageRegulator_EfdMaxPu': 'GEN____3_SM_voltageRegulator_EfdMaxPu'
                                                            })
    if drop_duplicate_timestep:
        df_ref = df_ref.set_index('time')
        df_ref = df_ref[~df_ref.index.duplicated(keep='last')]
    df_ref = df_ref.reset_index(drop=True)
    df_ref = df_ref[df_res.columns]
    return df_res.equals(df_ref)