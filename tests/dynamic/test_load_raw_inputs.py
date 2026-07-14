"""
Tests for gridfm_datakit.dynamic.load_raw_inputs.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from gridfm_datakit.dynamic import (
    STATIC_ELEMENT_DYNAMIC_MODELS_REQUIRED_COLS,
    AUTOMATION_SYSTEMS_REQUIRED_COLS,
    EVENTS_REQUIRED_COLS,
    VARIABLES_REQUIRED_COLS,
    DynamicInputs,
    load_raw_inputs,
)
from gridfm_datakit.utils.param_handler import NestedNamespace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(path: str, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def _make_config(tmp_dir: str, dataset) -> NestedNamespace:
    """Build a valid config pointing at CSVs in tmp_dir."""

    df_static_element_dynamic_models = dataset["df_static_element_dynamic_models"]
    df_automation_systems = dataset["df_automation_systems"]
    df_events = dataset["df_events"]
    df_variables = dataset["df_variables"]

    static_element_dynamic_models_path = os.path.join(
        tmp_dir,
        "static_element_dynamic_models.csv",
    )
    automation_systems_path = os.path.join(tmp_dir, "automation_systems.csv")
    events_path = os.path.join(tmp_dir, "events.csv")
    variables_path = os.path.join(tmp_dir, "variables.csv")

    _write_csv(static_element_dynamic_models_path, df_static_element_dynamic_models)
    _write_csv(automation_systems_path, df_automation_systems)
    _write_csv(events_path, df_events)
    _write_csv(variables_path, df_variables)

    return NestedNamespace(
        dynamic=NestedNamespace(
            dynamic_solver="dynawo",
            input_files=NestedNamespace(
                static_element_dynamic_models_file=static_element_dynamic_models_path,
                automation_systems_file=automation_systems_path,
                events_file=events_path,
                variables_file=variables_path,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadCsvFormat:
    def test_returns_dynamic_inputs(self, tmp_path, minimal_dataset):
        config = _make_config(str(tmp_path), minimal_dataset)
        result = load_raw_inputs(config)
        assert isinstance(result, DynamicInputs)

    def test_dynamic_models_shape(self, tmp_path, minimal_dataset):
        config = _make_config(str(tmp_path), minimal_dataset)
        result = load_raw_inputs(config)
        assert len(result.dynamic_models) == 2
        assert len(result.dynamic_models[0]) == 4
        assert len(result.dynamic_models[1]) == 1

    def test_events_shape(self, tmp_path, minimal_dataset):
        config = _make_config(str(tmp_path), minimal_dataset)
        result = load_raw_inputs(config)
        assert len(result.events) == 1

    def test_variables_shape(self, tmp_path, minimal_dataset):
        config = _make_config(str(tmp_path), minimal_dataset)
        result = load_raw_inputs(config)
        assert len(result.variables) == 3

    def test_all_required_model_columns_present(self, tmp_path, minimal_dataset):
        config = _make_config(str(tmp_path), minimal_dataset)
        result = load_raw_inputs(config)
        for col in STATIC_ELEMENT_DYNAMIC_MODELS_REQUIRED_COLS:
            assert col in result.dynamic_models[0].columns
        for col in AUTOMATION_SYSTEMS_REQUIRED_COLS:
            assert col in result.dynamic_models[1].columns

    def test_all_required_event_columns_present(self, tmp_path, minimal_dataset):
        config = _make_config(str(tmp_path), minimal_dataset)
        result = load_raw_inputs(config)
        for col in EVENTS_REQUIRED_COLS:
            assert col in result.events.columns

    def test_all_required_variable_columns_present(self, tmp_path, minimal_dataset):
        config = _make_config(str(tmp_path), minimal_dataset)
        result = load_raw_inputs(config)
        for col in VARIABLES_REQUIRED_COLS:
            assert col in result.variables.columns


# class TestLoadParquetFormat:
#     def test_loads_parquet(self, tmp_path):
#         """load_raw_inputs also accepts Parquet files."""
#         models_df = pd.DataFrame(
#             {
#                 "static_id": ["G1"],
#                 "dynamic_model_id": ["SomeModel"],
#                 "parameter_set_id": ["p0"],
#             },
#         )
#         events_df = pd.DataFrame(
#             {
#                 "static_id": ["L1"],
#                 "event_model_id": ["EventX"],
#                 "parameter_set_id": ["e0"],
#             },
#         )
#         variables_df = pd.DataFrame({"dynamic_model_id": ["G1"], "variable": ["omega"]})

#         models_path = str(tmp_path / "models.parquet")
#         events_path = str(tmp_path / "events.parquet")
#         variables_path = str(tmp_path / "variables.parquet")

#         models_df.to_parquet(models_path, index=False)
#         events_df.to_parquet(events_path, index=False)
#         variables_df.to_parquet(variables_path, index=False)

#         config = NestedNamespace(
#             dynamic=NestedNamespace(
#                 dynamic_solver="dynawo",
#                 dynamic_models_file=models_path,
#                 events_file=events_path,
#                 variables_file=variables_path,
#             ),
#         )
#         result = load_raw_inputs(config)
#         assert isinstance(result, DynamicInputs)
#         assert len(result.dynamic_models) == 1


class TestValidateValues:
    """The tables' *values* are validated, not just their columns.

    Each value below is later used as a key into a Dynawo mapping dict, or as a
    branch in _map_variables_dynawo. Unvalidated, a typo either raises a bare
    KeyError inside a worker process (failing every chunk with no hint which key
    was wrong) or — for the variables 'type' — is silently ignored, producing a
    simulation that monitors nothing and only dies later in the Zarr writer.
    """

    def _config(self, tmp_path, dataset, **overrides):
        dataset = {k: v.copy() for k, v in dataset.items()}
        dataset.update(overrides)
        return _make_config(str(tmp_path), dataset)

    def test_unknown_event_name_raises(self, tmp_path, minimal_dataset):
        events = minimal_dataset["df_events"].copy()
        events.loc[0, "event_name"] = "Disconect"  # typo
        config = self._config(tmp_path, minimal_dataset, df_events=events)
        with pytest.raises(ValueError, match=r"unsupported event_name \['Disconect'\]"):
            load_raw_inputs(config)

    def test_unknown_automation_category_raises(self, tmp_path, minimal_dataset):
        systems = minimal_dataset["df_automation_systems"].copy()
        systems.loc[0, "category_name"] = "UnderVoltage"  # model_name, not a category
        config = self._config(tmp_path, minimal_dataset, df_automation_systems=systems)
        with pytest.raises(ValueError, match="unsupported category_name"):
            load_raw_inputs(config)

    def test_unknown_variable_type_raises(self, tmp_path, minimal_dataset):
        """Silently dropped before: no curves -> ZeroDivisionError in the Zarr writer."""
        variables = minimal_dataset["df_variables"].copy()
        variables["type"] = "curve"  # lowercase typo
        config = self._config(tmp_path, minimal_dataset, df_variables=variables)
        with pytest.raises(ValueError, match=r"unsupported type \['curve'\]"):
            load_raw_inputs(config)

    def test_no_curve_rows_raises(self, tmp_path, minimal_dataset):
        """The dynamic store is built from Curve rows; with none, curves come back empty."""
        variables = minimal_dataset["df_variables"].copy()
        variables["type"] = "FinalStateValue"
        config = self._config(tmp_path, minimal_dataset, df_variables=variables)
        with pytest.raises(ValueError, match="no row of type 'Curve'"):
            load_raw_inputs(config)

    def test_valid_dataset_still_loads(self, tmp_path, minimal_dataset):
        assert isinstance(
            load_raw_inputs(_make_config(str(tmp_path), minimal_dataset)),
            DynamicInputs,
        )


class TestLoadRawInputsErrors:
    def test_missing_file_raises(self, tmp_path):
        config = NestedNamespace(
            dynamic=NestedNamespace(
                dynamic_solver="dynawo",
                input_files=NestedNamespace(
                    static_element_dynamic_models_file=str(
                        tmp_path / "nonexistent.csv",
                    ),
                    automation_systems_file=str(tmp_path / "automation_systems.csv"),
                    events_file=str(tmp_path / "events.csv"),
                    variables_file=str(tmp_path / "variables.csv"),
                ),
            ),
        )
        with pytest.raises(FileNotFoundError, match="nonexistent.csv"):
            load_raw_inputs(config)

    def test_missing_column_raises_value_error(self, tmp_path):
        """Missing required column must raise ValueError with descriptive message."""
        # models.csv missing 'parameter_set_id'
        static_element_dynamic_models_df = pd.DataFrame(
            {
                "category_name": ["SynchronousGenerator"],
                "static_id": ["G1"],
                # parameter_set_id intentionally missing,
                "model_name": [
                    "GeneratorSynchronousFourWindingsProportionalRegulations",
                ],
            },
        )

        automation_systems_df = pd.DataFrame(
            {
                "category_name": ["UnderVoltageAutomationSystem"],
                "dynamic_model": "UVA",
                "parameter_set_id": "G1UVA",
                "params": "generator=G1;",
                "model_name": "UnderVoltage",
            },
        )
        events_df = pd.DataFrame(
            {
                "event_name": ["Disconnect"],
                "static_id": ["G1"],
                "start_time": ["Ev"],
                "params": [""],
            },
        )
        variables_df = pd.DataFrame(
            {"type": "Curve", "model_id": ["G1"], "variables": ["v"]},
        )

        static_element_dynamic_models_df.to_csv(
            str(tmp_path / "models.csv"),
            index=False,
        )
        automation_systems_df.to_csv(
            str(tmp_path / "automation_systems.csv"),
            index=False,
        )
        events_df.to_csv(str(tmp_path / "events.csv"), index=False)
        variables_df.to_csv(str(tmp_path / "variables.csv"), index=False)

        config = NestedNamespace(
            dynamic=NestedNamespace(
                dynamic_solver="dynawo",
                input_files=NestedNamespace(
                    static_element_dynamic_models_file=str(tmp_path / "models.csv"),
                    automation_systems_file=str(tmp_path / "automation_systems.csv"),
                    events_file=str(tmp_path / "events.csv"),
                    variables_file=str(tmp_path / "variables.csv"),
                ),
            ),
        )
        with pytest.raises(ValueError, match="parameter_set_id"):
            load_raw_inputs(config)
