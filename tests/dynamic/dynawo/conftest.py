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

# No pytestmark: pytest ignores it in conftest. See markers.py.

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

UNIT_TEST_PARAMETERS_PATH = Path(__file__).parent / "unit_test_data/unit_tests.par"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
