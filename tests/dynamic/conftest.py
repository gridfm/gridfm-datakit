"""
Shared fixtures for dynamic module tests.
"""

import pytest
from gridfm_datakit.powsybl.api import is_powsybl_available

pytestmark = pytest.mark.skipif(
    is_powsybl_available() is False,
    reason="pypowsybl is not installed. Install with: pip install gridfm-datakit[powsybl]",
)

import os
import pandas as pd
from pathlib import Path
import pypowsybl as pp
from gridfm_datakit.powsybl import load_net
from gridfm_datakit.utils.param_handler import NestedNamespace


# File paths
PATH_NETWORK_IEEE14 = str(Path(__file__).parent/'dynawo/benchmark_data/ieee14/ieee14_GeneratorDisconnections/IEEE14.iidm')
PARAMETERS_PATH = str(Path(__file__).parent/'dynawo'/'benchmark_data/ieee14/ieee14_GeneratorDisconnections/IEEE14.par')
REF_OUTPUT_CURVES_PATH = str(Path(__file__).parent/'dynawo/benchmark_data/ieee14/ieee14_GeneratorDisconnections/ref_output_curves.csv')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path: str, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def _make_config(tmp_dir: str, 
                 dataset
) -> NestedNamespace:
    """Build a valid config pointing at CSVs in tmp_dir."""

    df_static_element_dynamic_models = dataset['df_static_element_dynamic_models']
    df_automation_systems = dataset['df_automation_systems']
    df_events = dataset['df_events']
    df_variables = dataset['df_variables']

    static_element_dynamic_models_path = os.path.join(tmp_dir, "static_element_dynamic_models.csv")
    automation_systems_path = os.path.join(tmp_dir, "automation_systems.csv")
    events_path = os.path.join(tmp_dir, "events.csv")
    variables_path = os.path.join(tmp_dir, "variables.csv")

    _write_csv(static_element_dynamic_models_path, df_static_element_dynamic_models)
    _write_csv(automation_systems_path, df_automation_systems)
    _write_csv(events_path, df_events)
    _write_csv(variables_path, df_variables)

    output_dir = os.path.join(tmp_dir)

    return NestedNamespace(
        network=NestedNamespace(
            name='IEEE14',
            reader='powsybl',
            source='file',
            file=PATH_NETWORK_IEEE14,
        ),
        load=NestedNamespace(
            generator='agg_load_profile',
            agg_profile='default',
            scenarios=1,
            sigma=0.2,
            change_reactive_power='true',
            global_range=0.4,
            max_scaling_factor=4.0,
            step_size=0.05,
            start_scaling_factor=0.8,
        ),
        dynamic=NestedNamespace(
            dynamic_solver="dynawo",
            input_files=NestedNamespace(
                static_element_dynamic_models_file=static_element_dynamic_models_path,
                automation_systems_file=automation_systems_path,
                events_file=events_path,
                variables_file=variables_path,
            ),
            solver_parameters=NestedNamespace(
                start_time=0.0,
                stop_time=500.0,
                parameters_file=PARAMETERS_PATH,
                network_parameters_file=PARAMETERS_PATH,
                network_parameters_id='Network',
                solver_type='SIM',
                solver_parameters_file=PARAMETERS_PATH,
                solver_parameters_id='SimplifiedSolver',
            ),
            output_dir=output_dir,
        ),
        settings=NestedNamespace(
            num_processes=1,
            data_dir=output_dir,
            large_chunk_size=5,
            overwrite='true',
            mode='pf',
            include_dc_res='false',
            enable_solver_logs='false',
            pf_fast='false',
            dcpf_fast='false',
            max_iter=200,
            pf_solver='powsybl',
            seed=49455
        )
    )

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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

@pytest.fixture(scope='function')
def config_ieee14(tmp_path, benchmark_dataset):
    return _make_config(tmp_path, benchmark_dataset)

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