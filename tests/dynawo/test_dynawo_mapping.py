import pytest
from gridfm_datakit.powsybl.api import is_powsybl_available

pytestmark = pytest.mark.skipif(
    is_powsybl_available() is False,
    reason="pypowsybl is not installed. Install with: pip install gridfm-datakit[powsybl]",
)

import pandas as pd
from pathlib import Path
import pypowsybl as pp


STATIC_ELEMENT_MODELS_PATH = Path(__file__).parent/'data/ieee14/ieee14_GeneratorDisconnections/model_mapping.csv'
AUTOMATION_SYSTEMS_PATH = Path(__file__).parent/'data/ieee14/ieee14_GeneratorDisconnections/automation_systems.csv'
EVENTS_PATH = Path(__file__).parent/'data/ieee14/ieee14_GeneratorDisconnections/event_mapping.csv'
VARIABLES_PATH = Path(__file__).parent/'data/ieee14/ieee14_GeneratorDisconnections/variable_mapping.csv'

# Tests

def test_baseline(pp_net_ieee14,
                  model_mapping_ieee14,
                  event_mapping_ieee14,
                  variable_mapping_ieee14,
                  param_ieee14,
                  df_ref_curves_ieee14,
                  ):
    sim = pp.dynamic.Simulation()
    report_node = pp.report.ReportNode()
    res = sim.run(
        network=pp_net_ieee14,
        model_mapping=model_mapping_ieee14,
        event_mapping=event_mapping_ieee14,
        timeseries_mapping=variable_mapping_ieee14,
        parameters=param_ieee14,
        report_node=report_node
    )
    assert _validate_output_curves_against_ref(res, df_ref_curves_ieee14)

def test_model_mapping(pp_net_ieee14,
                       event_mapping_ieee14,
                       variable_mapping_ieee14,
                       param_ieee14,
                       df_ref_curves_ieee14):
    from gridfm_datakit.dynamic.dynawo import _map_dynamic_models_dynawo

    sim = pp.dynamic.Simulation()
    report_node = pp.report.ReportNode()

    static_element_models = pd.read_csv(STATIC_ELEMENT_MODELS_PATH, sep=';')
    automation_systems = pd.read_csv(AUTOMATION_SYSTEMS_PATH, sep=';')
    model_mapping = _map_dynamic_models_dynawo(dynamic_models=[static_element_models, automation_systems])
        
    res = sim.run(
        network=pp_net_ieee14,
        model_mapping=model_mapping,
        event_mapping=event_mapping_ieee14,
        timeseries_mapping=variable_mapping_ieee14,
        parameters=param_ieee14,
        report_node=report_node
        )
    assert _validate_output_curves_against_ref(res, df_ref_curves_ieee14)

def test_event_mapping(pp_net_ieee14,
                       model_mapping_ieee14,
                       variable_mapping_ieee14,
                       param_ieee14,
                       df_ref_curves_ieee14
                       ):
    from gridfm_datakit.dynamic.dynawo import _map_events_dynawo

    events = pd.read_csv(EVENTS_PATH, sep=';')
    event_mapping = _map_events_dynawo(events)

    sim = pp.dynamic.Simulation()
    report_node = pp.report.ReportNode()

    res = sim.run(
        network=pp_net_ieee14,
        model_mapping=model_mapping_ieee14,
        event_mapping=event_mapping,
        timeseries_mapping=variable_mapping_ieee14,
        parameters=param_ieee14,
        report_node=report_node
        )
    
    assert _validate_output_curves_against_ref(res, df_ref_curves_ieee14)

def test_variable_mapping(pp_net_ieee14,
                          model_mapping_ieee14,
                          event_mapping_ieee14,
                          param_ieee14,
                          df_ref_curves_ieee14):
    from gridfm_datakit.dynamic.dynawo import _map_variables_dynawo

    variables = pd.read_csv(VARIABLES_PATH, sep=';')
    variable_mapping = _map_variables_dynawo(variables)

    sim = pp.dynamic.Simulation()
    report_node = pp.report.ReportNode()

    res = sim.run(
        network=pp_net_ieee14,
        model_mapping=model_mapping_ieee14,
        event_mapping=event_mapping_ieee14,
        timeseries_mapping=variable_mapping,
        parameters=param_ieee14,
        report_node=report_node
        )
    
    assert _validate_output_curves_against_ref(res, df_ref_curves_ieee14)

def test_generate_dynawo_mapping(pp_net_ieee14,
                                 param_ieee14,
                                 df_ref_curves_ieee14):
    
    from gridfm_datakit.dynamic import DynamicInputs
    from gridfm_datakit.dynamic.dynawo import generate_dynawo_mappings

    sim = pp.dynamic.Simulation()
    report_node = pp.report.ReportNode()
    dynamic_inputs = DynamicInputs(
        dynamic_models=[pd.read_csv(STATIC_ELEMENT_MODELS_PATH, sep=';'),
                        pd.read_csv(AUTOMATION_SYSTEMS_PATH, sep=';')
                        ],
        events=pd.read_csv(EVENTS_PATH, sep=';'),
        variables=pd.read_csv(VARIABLES_PATH, sep=';')
    )
    dynawo_mapping = generate_dynawo_mappings(dynamic_inputs)

    res = sim.run(
        network=pp_net_ieee14,
        model_mapping=dynawo_mapping.dynamic_model_mapping,
        event_mapping=dynawo_mapping.event_mapping,
        timeseries_mapping=dynawo_mapping.variable_mapping,
        parameters=param_ieee14,
        report_node=report_node
        )
    assert _validate_output_curves_against_ref(res, df_ref_curves_ieee14)

# utils

def _validate_output_curves_against_ref(res, df_ref_curves_ieee14):
    df_res = res.curves().reset_index(drop=True).rename(columns={'_GEN____1_SM_generator_efdPu_value': 'GEN____1_SM_generator_efdPu_value',
                                                                '_GEN____1_SM_voltageRegulator_EfdMaxPu': 'GEN____1_SM_voltageRegulator_EfdMaxPu',
                                                                '_GEN____3_SM_generator_UPu':'GEN____3_SM_generator_UPu',
                                                                '_GEN____3_SM_generator_efdPu_value': 'GEN____3_SM_generator_efdPu_value',
                                                                '_GEN____3_SM_voltageRegulator_EfdMaxPu': 'GEN____3_SM_voltageRegulator_EfdMaxPu'
                                                                })
    df_ref = df_ref_curves_ieee14.reset_index(drop=True)
    df_ref = df_ref[df_res.columns]
    return df_res.equals(df_ref)
