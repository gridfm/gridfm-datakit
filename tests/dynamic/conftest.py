"""
Shared fixtures for dynamic module tests.
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


# File paths
PATH_NETWORK_IEEE14 = Path(__file__).parent/'data/ieee14/ieee14_GeneratorDisconnections/IEEE14.iidm'
PATH_CONFIG_IEEE14 = Path(__file__).parent/'data/config/config_test_dynawo_IEEE14.yaml'
PARAMETERS_PATH = Path(__file__).parent/'data/ieee14/ieee14_GeneratorDisconnections/IEEE14.par'
REF_OUTPUT_CURVES_PATH = Path(__file__).parent/'data/ieee14/ieee14_GeneratorDisconnections/ref_output_curves.csv'


@pytest.fixture(scope="module")
def benchmark_dataset():
    return {
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
            columns=['event_nmae', 'static_id', 'start_time', 'params'],
            data=[
                ('Disconnect', '_GEN____2_SM', 50, ''),
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

@pytest.fixture(scope="module")
def minimal_dataset():
    return {
        "df_static_element_dynamic_models": pd.DataFrame.from_records(
            columns=['category_name', 'static_id', 'parameter_set_id', 'model_name'],
            data=[
                ('SynchronousGenerator', '_GEN____1_SM', 'Generator1', 'GeneratorSynchronousFourWindingsProportionalRegulations'),
                ('SynchronousGenerator', '_GEN____6_SM', 'Generator6', 'GeneratorSynchronousThreeWindingsProportionalRegulations'),
                ('LoadTwoTransformersTapChangers', '_LOAD___2_EC', 'GenericLoadTwoTransfos', 'LoadTwoTransformersTapChangers'),
                ('LoadOneTransformerTapChanger', '_LOAD___6_EC', 'GenericLoadOneTransfo', 'LoadOneTransformerTapChanger'),
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
                ('Disconnect', '_GEN____2_SM', 50, ''),
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

@pytest.fixture(scope="module")
def config_ieee14():
    import yaml

    with open(PATH_CONFIG_IEEE14) as f:
        config = yaml.safe_load(f)
    return config


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

@pytest.fixture(scope="function")
def pp_net_ieee14():
    loaded_net = load_net(str(PATH_NETWORK_IEEE14))
    return loaded_net.pp_net

@pytest.fixture(scope="module")
def df_ref_curves_ieee14():
    return pd.read_csv(
        REF_OUTPUT_CURVES_PATH,
        sep=';'
        )