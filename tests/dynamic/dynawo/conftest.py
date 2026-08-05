"""
Shared fixtures for dynawo module tests.
"""

from pathlib import Path

import pandas as pd
import pytest

# pypowsybl is an optional dependency; guard the import so collecting this
# conftest never fails when it is absent (the tests below are skipped anyway).
try:
    import pypowsybl as pp
except ImportError:  # pragma: no cover - optional dependency
    pp = None

from gridfm_datakit.powsybl import load_net

# No pytestmark: pytest ignores it in conftest. See markers.py.

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

PATH_NETWORK_IEEE14 = (
    Path(__file__).parent
    / "benchmark_data/ieee14/ieee14_GeneratorDisconnections/IEEE14.iidm"
)
PARAMETERS_PATH = (
    Path(__file__).parent
    / "benchmark_data/ieee14/ieee14_GeneratorDisconnections/IEEE14.par"
)
UNIT_TEST_PARAMETERS_PATH = Path(__file__).parent / "unit_test_data/unit_tests.par"
REF_OUTPUT_CURVES_PATH = (
    Path(__file__).parent
    / "benchmark_data/ieee14/ieee14_GeneratorDisconnections/ref_output_curves.csv"
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# benchmark dataset
@pytest.fixture(scope="function")
def benchmark_dataset():
    return {
        "df_static_element_dynamic_models": pd.DataFrame.from_records(
            columns=["category_name", "static_id", "parameter_set_id", "model_name"],
            data=[
                (
                    "SynchronousGenerator",
                    "_GEN____1_SM",
                    "Generator1",
                    "GeneratorSynchronousFourWindingsProportionalRegulations",
                ),
                (
                    "SynchronousGenerator",
                    "_GEN____2_SM",
                    "Generator2",
                    "GeneratorSynchronousFourWindingsProportionalRegulations",
                ),
                (
                    "SynchronousGenerator",
                    "_GEN____3_SM",
                    "Generator3",
                    "GeneratorSynchronousFourWindingsProportionalRegulations",
                ),
                (
                    "SynchronousGenerator",
                    "_GEN____6_SM",
                    "Generator6",
                    "GeneratorSynchronousThreeWindingsProportionalRegulations",
                ),
                (
                    "SynchronousGenerator",
                    "_GEN____8_SM",
                    "Generator8",
                    "GeneratorSynchronousThreeWindingsProportionalRegulations",
                ),
                (
                    "LoadTwoTransformersTapChangers",
                    "_LOAD___2_EC",
                    "GenericLoadTwoTransfos",
                    "LoadTwoTransformersTapChangers",
                ),
                (
                    "LoadTwoTransformersTapChangers",
                    "_LOAD___3_EC",
                    "GenericLoadTwoTransfos",
                    "LoadTwoTransformersTapChangers",
                ),
                (
                    "LoadTwoTransformersTapChangers",
                    "_LOAD___4_EC",
                    "GenericLoadTwoTransfos",
                    "LoadTwoTransformersTapChangers",
                ),
                (
                    "LoadTwoTransformersTapChangers",
                    "_LOAD___5_EC",
                    "GenericLoadTwoTransfos",
                    "LoadTwoTransformersTapChangers",
                ),
                (
                    "LoadOneTransformerTapChanger",
                    "_LOAD___6_EC",
                    "GenericLoadOneTransfo",
                    "LoadOneTransformerTapChanger",
                ),
                (
                    "LoadOneTransformerTapChanger",
                    "_LOAD___9_EC",
                    "GenericLoadOneTransfo",
                    "LoadOneTransformerTapChanger",
                ),
                (
                    "LoadOneTransformerTapChanger",
                    "_LOAD__10_EC",
                    "GenericLoadOneTransfo",
                    "LoadOneTransformerTapChanger",
                ),
                (
                    "LoadOneTransformerTapChanger",
                    "_LOAD__11_EC",
                    "GenericLoadOneTransfo",
                    "LoadOneTransformerTapChanger",
                ),
                (
                    "LoadOneTransformerTapChanger",
                    "_LOAD__12_EC",
                    "GenericLoadOneTransfo",
                    "LoadOneTransformerTapChanger",
                ),
                (
                    "LoadOneTransformerTapChanger",
                    "_LOAD__13_EC",
                    "GenericLoadOneTransfo",
                    "LoadOneTransformerTapChanger",
                ),
                (
                    "LoadOneTransformerTapChanger",
                    "_LOAD__14_EC",
                    "GenericLoadOneTransfo",
                    "LoadOneTransformerTapChanger",
                ),
            ],
        ),
        "df_automation_systems": pd.DataFrame.from_records(
            columns=[
                "category_name",
                "dynamic_model_id",
                "parameter_set_id",
                "params",
                "model_name",
            ],
            data=[
                (
                    "UnderVoltageAutomationSystem",
                    "UVA",
                    "UnderVoltageAutomatonGenerator3",
                    "generator=_GEN____3_SM;",
                    "UnderVoltage",
                ),
            ],
        ),
        "df_events": pd.DataFrame.from_records(
            columns=["event_name", "static_id", "start_time", "params"],
            data=[
                ("Disconnect", "_GEN____2_SM", 50, "disconnect_only=;"),
            ],
        ),
        "df_variables": pd.DataFrame.from_records(
            columns=["type", "model_id", "variables"],
            data=[
                ("Curve", "_BUS____2_TN", "U_value"),
                ("Curve", "UVA", "underVoltageAutomaton_UMinPu"),
                ("Curve", "_GEN____1_SM", "generator_efdPu_value"),
            ],
        ),
    }


# solver parameters
@pytest.fixture(scope="module")
def param_ieee14():
    return pp.dynamic.Parameters(
        start_time=0,
        stop_time=500,
        provider_parameters={
            "parametersFile": str(PARAMETERS_PATH),
            "network.parametersFile": str(PARAMETERS_PATH),
            "network.parametersId": "Network",
            "solver.type": "SIM",
            "solver.parametersFile": str(PARAMETERS_PATH),
            "solver.parametersId": "SimplifiedSolver",
        },
    )


@pytest.fixture(scope="module")
def unit_test_param():
    return pp.dynamic.Parameters(
        start_time=0,
        stop_time=10,
        provider_parameters={
            "parametersFile": str(UNIT_TEST_PARAMETERS_PATH),
            "network.parametersFile": str(UNIT_TEST_PARAMETERS_PATH),
            "network.parametersId": "Network",
            "solver.type": "SIM",
            "solver.parametersFile": str(UNIT_TEST_PARAMETERS_PATH),
            "solver.parametersId": "SimplifiedSolver",
        },
    )


# network
@pytest.fixture(scope="function")
def pp_net_ieee14():
    loaded_net = load_net(str(PATH_NETWORK_IEEE14))
    return loaded_net.pp_net


# dynamic model mapping
@pytest.fixture(scope="function")
def model_mapping_ieee14():
    model_mapping = pp.dynamic.ModelMapping()

    df_generator_mapping = pd.DataFrame.from_records(
        index="static_id",
        columns=["static_id", "parameter_set_id", "model_name"],
        data=[
            (
                "_GEN____1_SM",
                "Generator1",
                "GeneratorSynchronousFourWindingsProportionalRegulations",
            ),
            (
                "_GEN____2_SM",
                "Generator2",
                "GeneratorSynchronousFourWindingsProportionalRegulations",
            ),
            (
                "_GEN____3_SM",
                "Generator3",
                "GeneratorSynchronousFourWindingsProportionalRegulations",
            ),
            (
                "_GEN____6_SM",
                "Generator6",
                "GeneratorSynchronousThreeWindingsProportionalRegulations",
            ),
            (
                "_GEN____8_SM",
                "Generator8",
                "GeneratorSynchronousThreeWindingsProportionalRegulations",
            ),
        ],
    )
    model_mapping.add_dynamic_model(
        category_name="SynchronousGenerator",
        df=df_generator_mapping,
    )

    df_load_one_transfo_mapping = pd.DataFrame.from_records(
        index="static_id",
        columns=["static_id", "parameter_set_id", "model_name"],
        data=[
            ("_LOAD___6_EC", "GenericLoadOneTransfo", "LoadOneTransformerTapChanger"),
            ("_LOAD___9_EC", "GenericLoadOneTransfo", "LoadOneTransformerTapChanger"),
            ("_LOAD__10_EC", "GenericLoadOneTransfo", "LoadOneTransformerTapChanger"),
            ("_LOAD__11_EC", "GenericLoadOneTransfo", "LoadOneTransformerTapChanger"),
            ("_LOAD__12_EC", "GenericLoadOneTransfo", "LoadOneTransformerTapChanger"),
            ("_LOAD__13_EC", "GenericLoadOneTransfo", "LoadOneTransformerTapChanger"),
            ("_LOAD__14_EC", "GenericLoadOneTransfo", "LoadOneTransformerTapChanger"),
        ],
    )
    model_mapping.add_dynamic_model(
        category_name="LoadOneTransformerTapChanger",
        df=df_load_one_transfo_mapping,
    )

    df_two_load_transfo_mapping = pd.DataFrame.from_records(
        index="static_id",
        columns=["static_id", "parameter_set_id", "model_name"],
        data=[
            (
                "_LOAD___2_EC",
                "GenericLoadTwoTransfos",
                "LoadTwoTransformersTapChangers",
            ),
            (
                "_LOAD___3_EC",
                "GenericLoadTwoTransfos",
                "LoadTwoTransformersTapChangers",
            ),
            (
                "_LOAD___4_EC",
                "GenericLoadTwoTransfos",
                "LoadTwoTransformersTapChangers",
            ),
            (
                "_LOAD___5_EC",
                "GenericLoadTwoTransfos",
                "LoadTwoTransformersTapChangers",
            ),
        ],
    )
    model_mapping.add_dynamic_model(
        category_name="LoadTwoTransformersTapChangers",
        df=df_two_load_transfo_mapping,
    )

    df_automation_systems_mapping = pd.DataFrame.from_records(
        index="dynamic_model_id",
        columns=["dynamic_model_id", "parameter_set_id", "generator", "model_name"],
        data=[
            ("UVA", "UnderVoltageAutomatonGenerator3", "_GEN____3_SM", "UnderVoltage"),
        ],
    )
    model_mapping.add_under_voltage_automation_system(df=df_automation_systems_mapping)

    return model_mapping


# event mapping
@pytest.fixture(scope="function")
def event_mapping_ieee14():
    event_mapping = pp.dynamic.EventMapping()

    df_event_mapping = pd.DataFrame.from_records(
        index="static_id",
        columns=["static_id", "start_time"],
        data=[
            ("_GEN____2_SM", 50),
        ],
    )
    event_mapping.add_event_model(event_name="Disconnect", df=df_event_mapping)
    return event_mapping


# variable mapping
@pytest.fixture(scope="function")
def variable_mapping_ieee14():
    variable_mapping = pp.dynamic.OutputVariableMapping()
    variable_mapping.add_curves(model_id="_BUS____2_TN", variables="U_value")
    variable_mapping.add_curves(model_id="_GEN____3_SM", variables="generator_UPu")
    return variable_mapping


# reference outputs
@pytest.fixture(scope="function")
def df_ref_curves_ieee14():
    return pd.read_csv(REF_OUTPUT_CURVES_PATH, sep=";")
