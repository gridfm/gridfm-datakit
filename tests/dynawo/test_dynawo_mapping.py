import pytest
from gridfm_datakit.powsybl.api import is_powsybl_available

pytestmark = pytest.mark.skipif(
    is_powsybl_available() is False,
    reason="pypowsybl is not installed. Install with: pip install gridfm-datakit[powsybl]",
)

import pandas as pd
from pathlib import Path
import pypowsybl as pp
from gridfm_datakit.powsybl import load_net

# Test data
# simulation parameters

PARAM = pp.dynamic.Parameters(
    start_time=0,
    stop_time=500,
    provider_parameters={
    'parametersFile': str(Path(__file__).parent/'ieee14/ieee14_GeneratorDisconnections/IEEE14.par'),
    'network.parametersFile': str(Path(__file__).parent/'ieee14/ieee14_GeneratorDisconnections/IEEE14.par'),
    'network.parametersId': 'Network',
    'solver.type': 'SIM',
    'solver.parametersFile': str(Path(__file__).parent/'ieee14/ieee14_GeneratorDisconnections/IEEE14.par'),
    'solver.parametersId': 'SimplifiedSolver',
})

# Dynamic models
DF_GENERATOR_MAPPING= pd.DataFrame.from_records(
    index='static_id',
    columns=['static_id', 'parameter_set_id', 'model_name'],
    data=[
        ('_GEN____1_SM', 'Generator1', 'GeneratorSynchronousFourWindingsProportionalRegulations'),
        ('_GEN____2_SM', 'Generator2', 'GeneratorSynchronousFourWindingsProportionalRegulations'),
        ('_GEN____3_SM', 'Generator3', 'GeneratorSynchronousFourWindingsProportionalRegulations'),
        ('_GEN____6_SM', 'Generator6', 'GeneratorSynchronousThreeWindingsProportionalRegulations'),
        ('_GEN____8_SM', 'Generator8', 'GeneratorSynchronousThreeWindingsProportionalRegulations'),
        ]
)

DF_LOAD_ONE_TRANSFO_MAPPING = pd.DataFrame.from_records(
    index='static_id',
    columns=['static_id', 'parameter_set_id', 'model_name'],
    data=[
        ('_LOAD___6_EC', 'GenericLoadOneTransfo', 'LoadOneTransformerTapChanger'),
        ('_LOAD___9_EC', 'GenericLoadOneTransfo', 'LoadOneTransformerTapChanger'),
        ('_LOAD__10_EC', 'GenericLoadOneTransfo', 'LoadOneTransformerTapChanger'),
        ('_LOAD__11_EC', 'GenericLoadOneTransfo', 'LoadOneTransformerTapChanger'),
        ('_LOAD__12_EC', 'GenericLoadOneTransfo', 'LoadOneTransformerTapChanger'),
        ('_LOAD__13_EC', 'GenericLoadOneTransfo', 'LoadOneTransformerTapChanger'),
        ('_LOAD__14_EC', 'GenericLoadOneTransfo', 'LoadOneTransformerTapChanger'),
        ]
)

DF_LOAD_TWO_TRANSFO_MAPPING = pd.DataFrame.from_records(
    index='static_id',
    columns=['static_id', 'parameter_set_id', 'model_name'],
    data=[
        ('_LOAD___2_EC', 'GenericLoadTwoTransfos', 'LoadTwoTransformersTapChangers'),
        ('_LOAD___3_EC', 'GenericLoadTwoTransfos', 'LoadTwoTransformersTapChangers'),
        ('_LOAD___4_EC', 'GenericLoadTwoTransfos', 'LoadTwoTransformersTapChangers'),
        ('_LOAD___5_EC', 'GenericLoadTwoTransfos', 'LoadTwoTransformersTapChangers'),
        ]
)

# Automation systems
DF_AUTOMATION_SYSTEMS_MAPPING = pd.DataFrame.from_records(
    index='dynamic_model_id',
    columns=['dynamic_model_id', 'parameter_set_id', 'generator', 'model_name'],
    data=[
        ('UVA', 'UnderVoltageAutomatonGenerator3', '_GEN____3_SM', 'UnderVoltage'),
        ]
)

# Events
DF_EVENT_MAPPING = pd.DataFrame.from_records(
    index='static_id',
    columns=['static_id', 'start_time'],
    data=[
        ('_GEN____2_SM', 50),
        ]
)

# Benchmark data
DF_REF_CURVES = pd.read_csv(
    Path(__file__).parent/'ieee14/ieee14_GeneratorDisconnections/ref_output_curves.csv',
    sep=';'
)

# Fixtures 

@pytest.fixture(scope="function")
def pp_net():
    loaded_net = load_net(str(Path(__file__).parent/'ieee14/ieee14_GeneratorDisconnections/IEEE14.iidm'))
    return loaded_net.pp_net

@pytest.fixture(scope="module")
def model_mapping():
    model_mapping = pp.dynamic.ModelMapping()
    model_mapping.add_dynamic_model(category_name='SynchronousGenerator', df=DF_GENERATOR_MAPPING)
    model_mapping.add_dynamic_model(category_name='LoadOneTransformerTapChanger', df=DF_LOAD_ONE_TRANSFO_MAPPING)
    model_mapping.add_dynamic_model(category_name='LoadTwoTransformersTapChangers', df=DF_LOAD_TWO_TRANSFO_MAPPING)
    model_mapping.add_under_voltage_automation_system(df=DF_AUTOMATION_SYSTEMS_MAPPING)
    return model_mapping

@pytest.fixture(scope="module")
def event_mapping():
    event_mapping = pp.dynamic.EventMapping()
    event_mapping.add_event_model(event_name='Disconnect', df=DF_EVENT_MAPPING)
    return event_mapping

@pytest.fixture(scope="module")
def variable_mapping():
    variable_mapping = pp.dynamic.OutputVariableMapping()
    variable_mapping.add_curves(model_id='_BUS____2_TN', variables='U_value')
    variable_mapping.add_curves(model_id='_GEN____3_SM', variables='generator_UPu')
    return variable_mapping

# Tests

def test_baseline(pp_net,
                  model_mapping,
                  event_mapping,
                  variable_mapping,
                  ):
    sim = pp.dynamic.Simulation()
    report_node = pp.report.ReportNode()
    res = sim.run(
        network=pp_net,
        model_mapping=model_mapping,
        event_mapping=event_mapping,
        timeseries_mapping=variable_mapping,
        parameters=PARAM,
        report_node=report_node
    )
    assert validate_output_curves_against_ref(res)

def test_model_mapping(pp_net,
                       event_mapping,
                       variable_mapping):
    from gridfm_datakit.dynamic.dynawo import _map_dynamic_models_dynawo

    sim = pp.dynamic.Simulation()
    report_node = pp.report.ReportNode()

    static_element_models = pd.read_csv(Path(__file__).parent/'ieee14/ieee14_GeneratorDisconnections/model_mapping.csv', sep=';')
    automation_systems = pd.read_csv(Path(__file__).parent/'ieee14/ieee14_GeneratorDisconnections/automation_systems.csv', sep=';')
    model_mapping = _map_dynamic_models_dynawo(dynamic_models=[static_element_models, automation_systems])
        
    res = sim.run(
        network=pp_net,
        model_mapping=model_mapping,
        event_mapping=event_mapping,
        timeseries_mapping=variable_mapping,
        parameters=PARAM,
        report_node=report_node
        )
    assert validate_output_curves_against_ref(res)

def test_event_mapping(pp_net,
                       model_mapping,
                       variable_mapping):
    from gridfm_datakit.dynamic.dynawo import _map_events_dynawo

    events = pd.read_csv(Path(__file__).parent/'ieee14/ieee14_GeneratorDisconnections/event_mapping.csv', sep=';')
    event_mapping = _map_events_dynawo(events)

    sim = pp.dynamic.Simulation()
    report_node = pp.report.ReportNode()

    res = sim.run(
        network=pp_net,
        model_mapping=model_mapping,
        event_mapping=event_mapping,
        timeseries_mapping=variable_mapping,
        parameters=PARAM,
        report_node=report_node
        )
    
    assert validate_output_curves_against_ref(res)

def test_variable_mapping(pp_net,
                          model_mapping,
                          event_mapping):
    from gridfm_datakit.dynamic.dynawo import _map_variables_dynawo

    variables = pd.read_csv(Path(__file__).parent/'ieee14/ieee14_GeneratorDisconnections/variable_mapping.csv', sep=';')
    variable_mapping = _map_variables_dynawo(variables)

    sim = pp.dynamic.Simulation()
    report_node = pp.report.ReportNode()

    res = sim.run(
        network=pp_net,
        model_mapping=model_mapping,
        event_mapping=event_mapping,
        timeseries_mapping=variable_mapping,
        parameters=PARAM,
        report_node=report_node
        )
    
    assert validate_output_curves_against_ref(res)

def test_generate_dynawo_mapping(pp_net):
    
    from gridfm_datakit.dynamic import DynamicInputs
    from gridfm_datakit.dynamic.dynawo import generate_dynawo_mappings

    sim = pp.dynamic.Simulation()
    report_node = pp.report.ReportNode()
    dynamic_inputs = DynamicInputs(
        dynamic_models=[pd.read_csv(Path(__file__).parent/'ieee14/ieee14_GeneratorDisconnections/model_mapping.csv', sep=';'),
                        pd.read_csv(Path(__file__).parent/'ieee14/ieee14_GeneratorDisconnections/automation_systems.csv', sep=';')
                        ],
        events=pd.read_csv(Path(__file__).parent/'ieee14/ieee14_GeneratorDisconnections/event_mapping.csv', sep=';'),
        variables=pd.read_csv(Path(__file__).parent/'ieee14/ieee14_GeneratorDisconnections/variable_mapping.csv', sep=';')
    )
    dynawo_mapping = generate_dynawo_mappings(dynamic_inputs)

    res = sim.run(
        network=pp_net,
        model_mapping=dynawo_mapping.dynamic_model_mapping,
        event_mapping=dynawo_mapping.event_mapping,
        timeseries_mapping=dynawo_mapping.variable_mapping,
        parameters=PARAM,
        report_node=report_node
        )
    assert validate_output_curves_against_ref(res)

# utils

def validate_output_curves_against_ref(res):
    df_res = res.curves().reset_index(drop=True).rename(columns={'_GEN____1_SM_generator_efdPu_value': 'GEN____1_SM_generator_efdPu_value',
                                                                '_GEN____1_SM_voltageRegulator_EfdMaxPu': 'GEN____1_SM_voltageRegulator_EfdMaxPu',
                                                                '_GEN____3_SM_generator_UPu':'GEN____3_SM_generator_UPu',
                                                                '_GEN____3_SM_generator_efdPu_value': 'GEN____3_SM_generator_efdPu_value',
                                                                '_GEN____3_SM_voltageRegulator_EfdMaxPu': 'GEN____3_SM_voltageRegulator_EfdMaxPu'
                                                                })
    df_ref = DF_REF_CURVES.reset_index(drop=True)
    df_ref = df_ref[df_res.columns]
    return df_res.equals(df_ref)
