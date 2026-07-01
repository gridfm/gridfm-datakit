"""
Shared fixtures for dynawo module tests.
"""

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


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

PATH_NETWORK_IEEE14 = Path(__file__).parent/'benchmark_data/ieee14/ieee14_GeneratorDisconnections/IEEE14.iidm'
PARAMETERS_PATH = Path(__file__).parent/'benchmark_data/ieee14/ieee14_GeneratorDisconnections/IEEE14.par'
REF_OUTPUT_CURVES_PATH = Path(__file__).parent/'benchmark_data/ieee14/ieee14_GeneratorDisconnections/ref_output_curves.csv'

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# solver parameters

@pytest.fixture(scope="module")
def param_ieee14():
    return pp.dynamic.Parameters(
        start_time=0,
        stop_time=500,
        provider_parameters={
        'parametersFile': str(PARAMETERS_PATH),
        'network.parametersFile': str(PARAMETERS_PATH),
        'network.parametersId': 'Network',
        'solver.type': 'SIM',
        'solver.parametersFile': str(PARAMETERS_PATH),
        'solver.parametersId': 'SimplifiedSolver',
        }
        )

# network

@pytest.fixture(scope="function")
def pp_net_ieee14():
    loaded_net = load_net(str(PATH_NETWORK_IEEE14))
    return loaded_net.pp_net

# dynamic model mapping
@pytest.fixture(scope="module")
def model_mapping_ieee14():
    
    model_mapping = pp.dynamic.ModelMapping()

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
    model_mapping.add_dynamic_model(category_name='SynchronousGenerator', df=DF_GENERATOR_MAPPING)

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
    model_mapping.add_dynamic_model(category_name='LoadOneTransformerTapChanger', df=DF_LOAD_ONE_TRANSFO_MAPPING)

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
    model_mapping.add_dynamic_model(category_name='LoadTwoTransformersTapChangers', df=DF_LOAD_TWO_TRANSFO_MAPPING)

    DF_AUTOMATION_SYSTEMS_MAPPING = pd.DataFrame.from_records(
    index='dynamic_model_id',
    columns=['dynamic_model_id', 'parameter_set_id', 'generator', 'model_name'],
    data=[
        ('UVA', 'UnderVoltageAutomatonGenerator3', '_GEN____3_SM', 'UnderVoltage'),
        ]
    )
    model_mapping.add_under_voltage_automation_system(df=DF_AUTOMATION_SYSTEMS_MAPPING)

    return model_mapping

# event mapping
@pytest.fixture(scope="module")
def event_mapping_ieee14():
    event_mapping = pp.dynamic.EventMapping()

    DF_EVENT_MAPPING = pd.DataFrame.from_records(
        index='static_id',
        columns=['static_id', 'start_time'],
        data=[
            ('_GEN____2_SM', 50),
            ]
    )
    event_mapping.add_event_model(event_name='Disconnect', df=DF_EVENT_MAPPING)
    return event_mapping

# variable mapping
@pytest.fixture(scope="module")
def variable_mapping_ieee14():
    variable_mapping = pp.dynamic.OutputVariableMapping()
    variable_mapping.add_curves(model_id='_BUS____2_TN', variables='U_value')
    variable_mapping.add_curves(model_id='_GEN____3_SM', variables='generator_UPu')
    return variable_mapping

# reference outputs
@pytest.fixture(scope="module")
def df_ref_curves_ieee14():
    return pd.read_csv(
        REF_OUTPUT_CURVES_PATH,
        sep=';'
    )