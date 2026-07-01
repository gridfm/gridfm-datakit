import pytest
from gridfm_datakit.powsybl.api import is_powsybl_available

pytestmark = pytest.mark.skipif(
    is_powsybl_available() is False,
    reason="pypowsybl is not installed. Install with: pip install gridfm-datakit[powsybl]",
)

import os
import pandas as pd
import pypowsybl as pp


# ---------------------------------------------------------------------------
# Benchmark case dataset
# ---------------------------------------------------------------------------
BENCHMARK_DATASET = {
        "df_static_element_dynamic_models": pd.DataFrame.from_records(
            columns=['category_name', 'static_id', 'parameter_set_id', 'model_name'],
            data=[
                ('SynchronousGenerator', '_GEN____1_SM', 'Generator1', 'GeneratorSynchronousFourWindingsProportionalRegulations'),
                ('SynchronousGenerator', '_GEN____2_SM', 'Generator2', 'GeneratorSynchronousFourWindingsProportionalRegulations'),
                ('SynchronousGenerator', '_GEN____3_SM', 'Generator3', 'GeneratorSynchronousFourWindingsProportionalRegulations'),
                ('SynchronousGenerator', '_GEN____6_SM', 'Generator6', 'GeneratorSynchronousThreeWindingsProportionalRegulations'),
                ('SynchronousGenerator', '_GEN____8_SM', 'Generator8', 'GeneratorSynchronousThreeWindingsProportionalRegulations'),
                ('LoadTwoTransformersTapChangers', '_LOAD___2_EC', 'GenericLoadTwoTransfos', 'LoadTwoTransformersTapChangers'),
                ('LoadTwoTransformersTapChangers', '_LOAD___3_EC', 'GenericLoadTwoTransfos', 'LoadTwoTransformersTapChangers'),
                ('LoadTwoTransformersTapChangers', '_LOAD___4_EC', 'GenericLoadTwoTransfos', 'LoadTwoTransformersTapChangers'),
                ('LoadTwoTransformersTapChangers', '_LOAD___5_EC', 'GenericLoadTwoTransfos', 'LoadTwoTransformersTapChangers'),
                ('LoadOneTransformerTapChanger', '_LOAD___6_EC', 'GenericLoadOneTransfo', 'LoadOneTransformerTapChanger'),
                ('LoadOneTransformerTapChanger', '_LOAD___9_EC', 'GenericLoadOneTransfo', 'LoadOneTransformerTapChanger'),
                ('LoadOneTransformerTapChanger', '_LOAD__10_EC', 'GenericLoadOneTransfo', 'LoadOneTransformerTapChanger'),
                ('LoadOneTransformerTapChanger', '_LOAD__11_EC', 'GenericLoadOneTransfo', 'LoadOneTransformerTapChanger'),
                ('LoadOneTransformerTapChanger', '_LOAD__12_EC', 'GenericLoadOneTransfo', 'LoadOneTransformerTapChanger'),
                ('LoadOneTransformerTapChanger', '_LOAD__13_EC', 'GenericLoadOneTransfo', 'LoadOneTransformerTapChanger'),
                ('LoadOneTransformerTapChanger', '_LOAD__14_EC', 'GenericLoadOneTransfo', 'LoadOneTransformerTapChanger'),
            ]
        ),
        
        "df_automation_systems": pd.DataFrame.from_records(
            columns=['category_name', 'dynamic_model_id', 'parameter_set_id', 'params', 'model_name'],
            data=[
                ('UnderVoltageAutomationSystem', 'UVA', 'UnderVoltageAutomatonGenerator3', 'generator=_GEN____3_SM;', 'UnderVoltage'),
            ]
        ),

        "df_events": pd.DataFrame.from_records(
            columns=['event_name', 'static_id', 'start_time', 'params'],
            data=[
                ('Disconnect', '_GEN____2_SM', 50, 'disconnect_only=;'),
            ]   
        ),

        "df_variables": pd.DataFrame.from_records(
            columns=['type', 'model_id', 'variables'],
            data=[
                ('Curve', '_BUS____2_TN', 'U_value'),
                ('Curve', 'UVA', 'underVoltageAutomaton_UMinPu'),
                ('Curve', '_GEN____1_SM', 'generator_efdPu_value'),
            ]
        )
    }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path: str, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='function')
def static_element_model_path(tmp_path):
    path = os.path.join(tmp_path, "static_element_dynamic_models.csv")
    _write_csv(
        path,
        BENCHMARK_DATASET['df_static_element_dynamic_models'],
        )
    return path

@pytest.fixture(scope='function')
def automation_systems_path(tmp_path):
    path = os.path.join(tmp_path, "automation_systems.csv")
    _write_csv(
        path,
        BENCHMARK_DATASET['df_automation_systems'],
        )
    return path

@pytest.fixture(scope='function')
def events_path(tmp_path):
    path = os.path.join(tmp_path, "events.csv")
    _write_csv(
        path,
        BENCHMARK_DATASET['df_events'],
        )
    return path

@pytest.fixture(scope='function')
def variables_path(tmp_path):
    path = os.path.join(tmp_path, "variables.csv")
    _write_csv(
        path,
        BENCHMARK_DATASET['df_variables'],
        )
    return path

# ---------------------------------------------------------------------------
# Unit tests for the mappers
# ---------------------------------------------------------------------------
# pypowsybl.dynamic.xxxMapping objects don't support native comparison between two instances,
# The tests are performed as follows. 
# For each input of the dynamic simulation: 
#   1. Only the input produced by the tested mapper is replaced by the mapper while
#   the other inputs are kept as those of the baseline. 
#   2. Run dynamic simulation
#   3. Compared outcomes against benchmark

# start by testing the baseline, which takes the inputs given directly by the conftest file
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

# test dynamic model mapper
def test_dynamic_model_mapping(pp_net_ieee14,
                       event_mapping_ieee14,
                       variable_mapping_ieee14,
                       param_ieee14,
                       df_ref_curves_ieee14,
                       static_element_model_path,
                       automation_systems_path):
    from gridfm_datakit.dynamic.dynawo import _map_dynamic_models_dynawo

    sim = pp.dynamic.Simulation()
    report_node = pp.report.ReportNode()

    static_element_models = pd.read_csv(static_element_model_path)
    automation_systems = pd.read_csv(automation_systems_path)
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

# test event mapper
def test_event_mapping(pp_net_ieee14,
                       model_mapping_ieee14,
                       variable_mapping_ieee14,
                       param_ieee14,
                       df_ref_curves_ieee14,
                       events_path,
                       ):
    from gridfm_datakit.dynamic.dynawo import _map_events_dynawo

    events = pd.read_csv(events_path)
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

# test variable mapper
def test_variable_mapping(pp_net_ieee14,
                          model_mapping_ieee14,
                          event_mapping_ieee14,
                          param_ieee14,
                          df_ref_curves_ieee14,
                          variables_path):
    from gridfm_datakit.dynamic.dynawo import _map_variables_dynawo

    variables = pd.read_csv(variables_path)
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

# test full mapping process through generate_dynawo_mapping
def test_generate_dynawo_mapping(pp_net_ieee14,
                                 param_ieee14,
                                 df_ref_curves_ieee14,
                                 static_element_model_path,
                                 automation_systems_path,
                                 events_path,
                                 variables_path
                                 ):
    
    from gridfm_datakit.dynamic import DynamicInputs
    from gridfm_datakit.dynamic.dynawo import generate_dynawo_mappings

    sim = pp.dynamic.Simulation()
    report_node = pp.report.ReportNode()
    dynamic_inputs = DynamicInputs(
        dynamic_models=[pd.read_csv(static_element_model_path),
                        pd.read_csv(automation_systems_path)
                        ],
        events=pd.read_csv(events_path),
        variables=pd.read_csv(variables_path)
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
