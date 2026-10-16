"""Tests for dynamic generation configuration validation."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from gridfm_datakit.config import validate_dynamic_config
from gridfm_datakit.dynamic import load_raw_inputs
from gridfm_datakit.dynamic import generate_dynamic as gd
from gridfm_datakit.dynamic.generate_dynamic import generate_dynamic_data
from gridfm_datakit.utils.param_handler import NestedNamespace


_MAX_NUMPY_SEED = 2**32 - 1
_DISTRIBUTED_SEED_STRIDE = 20_000


def _default_config() -> dict[str, Any]:
    """Return an independent copy of the shipped dynamic example config."""
    with Path("scripts/dynamic_example/config.yaml").open() as stream:
        return yaml.safe_load(stream)


def _create_existing_output(config: dict[str, Any]) -> Path:
    """Create an output marker that must survive configuration failures."""
    base_path = Path(config["settings"]["data_dir"]) / config["network"]["name"] / "raw"
    base_path.mkdir(parents=True)
    marker = base_path / "marker.txt"
    marker.write_text("must survive validation failure")
    return marker


def test_validate_dynamic_config_accepts_shipped_example() -> None:
    validated = validate_dynamic_config(_default_config())

    assert validated["dynamic"]["dynamic_solver"] == "dynawo"
    assert validated["dynamic"]["logging"] == {
        "verbosity": "info",
        "save_reports": True,
    }
    assert validated["settings"]["opf_formulation"] == "polar"


def test_validate_dynamic_config_preserves_validate_alias() -> None:
    config = _default_config()
    config["dynamic"]["validate"] = True

    validated = validate_dynamic_config(config)

    assert validated["dynamic"]["validate"] is True
    assert "run_validation" not in validated["dynamic"]


def test_validate_dynamic_config_supplies_execution_and_perturbation_defaults() -> None:
    config = _default_config()
    del config["settings"]["num_processes"]
    del config["settings"]["large_chunk_size"]

    validated = validate_dynamic_config(config)

    assert validated["settings"]["num_processes"] == 1
    assert validated["settings"]["large_chunk_size"] == 1_000
    assert validated["generation_perturbation"] == {"type": "none"}
    assert validated["admittance_perturbation"] == {"type": "none"}


def test_validate_dynamic_config_requires_every_input_table() -> None:
    config = _default_config()
    del config["dynamic"]["input_files"]["events_file"]

    with pytest.raises(
        ValueError,
        match=r"dynamic\.input_files\.events_file: Field required",
    ):
        validate_dynamic_config(config)


def test_validate_dynamic_config_rejects_unknown_solver_parameter() -> None:
    config = _default_config()
    config["dynamic"]["solver_parameters"]["solver_typo"] = "SIM"

    with pytest.raises(
        ValueError,
        match=r"dynamic\.solver_parameters\.solver_typo: Extra inputs are not permitted",
    ):
        validate_dynamic_config(config)


@pytest.mark.parametrize("field", ["solver_type", "precision"])
def test_validate_dynamic_config_accepts_none_solver_parameter(field: str) -> None:
    config = _default_config()
    config["dynamic"]["solver_parameters"][field] = "none"

    validated = validate_dynamic_config(config)

    assert validated["dynamic"]["solver_parameters"][field] == "none"


def test_validate_dynamic_config_rejects_unknown_loadflow_parameter() -> None:
    config = _default_config()
    config["dynamic"]["loadflow_parameters"] = {"distributed_slak": True}

    with pytest.raises(
        ValueError,
        match=r"dynamic\.loadflow_parameters\.distributed_slak: Extra inputs are not permitted",
    ):
        validate_dynamic_config(config)


@pytest.mark.parametrize(
    ("start_time", "stop_time"),
    [(1.0, 1.0), (2.0, 1.0)],
    ids=["empty", "backwards"],
)
def test_validate_dynamic_config_rejects_invalid_simulation_window(
    start_time: float,
    stop_time: float,
) -> None:
    config = _default_config()
    config["dynamic"]["solver_parameters"].update(
        start_time=start_time,
        stop_time=stop_time,
    )

    with pytest.raises(
        ValueError,
        match=r"dynamic\.solver_parameters: stop_time must be greater than start_time",
    ):
        validate_dynamic_config(config)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_validate_dynamic_config_rejects_non_finite_solver_time(value: float) -> None:
    config = _default_config()
    config["dynamic"]["solver_parameters"]["stop_time"] = value

    with pytest.raises(
        ValueError,
        match=r"dynamic\.solver_parameters\.stop_time: Input should be a finite number",
    ):
        validate_dynamic_config(config)


@pytest.mark.parametrize(
    ("seed", "match"),
    [
        (-1, r"settings\.seed: Input should be greater than or equal to 0"),
        (
            2**32,
            r"settings\.seed: Input should be less than or equal to 4294967295",
        ),
        (
            (_MAX_NUMPY_SEED - 6) // _DISTRIBUTED_SEED_STRIDE + 1,
            r"configuration: settings\.seed and load\.scenarios produce a derived random seed",
        ),
    ],
)
def test_validate_dynamic_config_rejects_invalid_seed(seed: int, match: str) -> None:
    config = _default_config()
    config["settings"]["seed"] = seed

    with pytest.raises(ValueError, match=match):
        validate_dynamic_config(config)


def test_validate_dynamic_config_rejects_incompatible_reader() -> None:
    config = _default_config()
    config["network"]["reader"] = "native"
    config["network"]["network_dir"] = "grids"

    with pytest.raises(
        ValueError,
        match=(
            r"configuration: dynamic\.dynamic_solver='dynawo' is incompatible "
            r"with network\.reader='native'; missing capabilities:"
        ),
    ):
        validate_dynamic_config(config)


@pytest.mark.parametrize("input_kind", ["yaml", "dict", "namespace"])
def test_dynamic_generation_validates_before_overwrite(
    tmp_path: Path,
    input_kind: str,
) -> None:
    config = _default_config()
    config["settings"]["data_dir"] = str(tmp_path / "data")
    del config["dynamic"]["input_files"]["events_file"]
    marker = _create_existing_output(config)

    if input_kind == "yaml":
        config_input = tmp_path / "invalid.yaml"
        config_input.write_text(yaml.safe_dump(config))
    elif input_kind == "namespace":
        config_input = NestedNamespace(**deepcopy(config))
    else:
        config_input = deepcopy(config)

    with pytest.raises(
        ValueError,
        match=r"dynamic\.input_files\.events_file: Field required",
    ):
        generate_dynamic_data(config_input)

    assert marker.read_text() == "must survive validation failure"


def test_dynamic_generation_runs_schema_validation_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    validate = gd.validate_dynamic_config

    def _validate(config):
        nonlocal calls
        calls += 1
        return validate(config)

    def _stop_after_validation(_args):
        raise RuntimeError("stop after schema validation")

    monkeypatch.setattr(gd, "validate_dynamic_config", _validate)
    monkeypatch.setattr(
        "gridfm_datakit.dynamic.dynawo.api.check_dynawo_available",
        lambda: None,
    )
    monkeypatch.setattr(gd, "load_raw_inputs", _stop_after_validation)

    with pytest.raises(RuntimeError, match="stop after schema validation"):
        generate_dynamic_data(_default_config())

    assert calls == 1


def test_dynamic_input_tables_are_checked_before_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _default_config()
    config["settings"]["data_dir"] = str(tmp_path / "data")
    for key in config["dynamic"]["input_files"]:
        config["dynamic"]["input_files"][key] = str(tmp_path / f"missing-{key}.csv")
    marker = _create_existing_output(config)

    monkeypatch.setattr(
        "gridfm_datakit.dynamic.dynawo.api.check_dynawo_available",
        lambda: None,
    )

    with pytest.raises(FileNotFoundError, match="Dynamic input file not found"):
        generate_dynamic_data(config)

    assert marker.read_text() == "must survive validation failure"


@pytest.mark.parametrize(
    ("event_time", "match"),
    [
        ("NaN", "must contain only finite values"),
        ("inf", "must contain only finite values"),
        ("-inf", "must contain only finite values"),
        ("-0.1", "must be within dynamic.solver_parameters window"),
        ("500.1", "must be within dynamic.solver_parameters window"),
    ],
    ids=["nan", "positive-infinity", "negative-infinity", "before", "after"],
)
def test_dynamic_event_times_are_checked_before_overwrite(
    config_ieee14: NestedNamespace,
    monkeypatch: pytest.MonkeyPatch,
    event_time: str,
    match: str,
) -> None:
    events_path = Path(config_ieee14.dynamic.input_files.events_file)
    events = pd.read_csv(events_path)
    events["start_time"] = event_time
    events.to_csv(events_path, index=False)
    marker = _create_existing_output(config_ieee14.to_dict())

    monkeypatch.setattr(
        "gridfm_datakit.dynamic.dynawo.api.check_dynawo_available",
        lambda: None,
    )

    with pytest.raises(ValueError, match=match):
        generate_dynamic_data(config_ieee14)

    assert marker.read_text() == "must survive validation failure"


def test_dynamic_event_times_accept_both_window_boundaries(
    config_ieee14: NestedNamespace,
) -> None:
    dynamic_inputs = load_raw_inputs(config_ieee14)
    event = dynamic_inputs.events.iloc[[0]].copy()
    start_time = config_ieee14.dynamic.solver_parameters.start_time
    stop_time = config_ieee14.dynamic.solver_parameters.stop_time
    dynamic_inputs.events = pd.concat([event, event], ignore_index=True)
    dynamic_inputs.events["start_time"] = [start_time, stop_time]

    gd._validate_event_time_window(config_ieee14, dynamic_inputs)


@pytest.mark.parametrize("case", ["unknown-event-param", "unsupported-automation"])
def test_dynamic_params_are_checked_before_overwrite(
    config_ieee14: NestedNamespace,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    if case == "unknown-event-param":
        table_path = Path(config_ieee14.dynamic.input_files.events_file)
        table = pd.read_csv(table_path)
        table.loc[0, "event_name"] = "ActivePowerVariation"
        table.loc[0, "params"] = "delta_p=0.2;typo=0"
        match = "unexpected parameter"
    else:
        table_path = Path(
            config_ieee14.dynamic.input_files.automation_systems_file,
        )
        table = pd.read_csv(table_path)
        table.loc[0, "category_name"] = "TapChangerBlocking"
        table.loc[0, "params"] = (
            "rfo_df=one;mp1_df=one;mp2_df=one;mp3_df=one;mp4_df=one;mp5_df=one"
        )
        match = "cannot be configured from the flat params column"
    table.to_csv(table_path, index=False)
    marker = _create_existing_output(config_ieee14.to_dict())

    monkeypatch.setattr(
        "gridfm_datakit.dynamic.dynawo.api.check_dynawo_available",
        lambda: None,
    )

    with pytest.raises(ValueError, match=match):
        generate_dynamic_data(config_ieee14)

    assert marker.read_text() == "must survive validation failure"


@pytest.mark.parametrize(
    ("input_file", "column"),
    [
        ("static_element_dynamic_models_file", "static_id"),
        ("automation_systems_file", "dynamic_model_id"),
        ("events_file", "static_id"),
        ("variables_file", "model_id"),
    ],
)
def test_empty_dynamic_input_values_are_checked_before_overwrite(
    config_ieee14: NestedNamespace,
    monkeypatch: pytest.MonkeyPatch,
    input_file: str,
    column: str,
) -> None:
    table_path = Path(getattr(config_ieee14.dynamic.input_files, input_file))
    table = pd.read_csv(table_path)
    table.loc[0, column] = "   "
    table.to_csv(table_path, index=False)
    marker = _create_existing_output(config_ieee14.to_dict())

    monkeypatch.setattr(
        "gridfm_datakit.dynamic.dynawo.api.check_dynawo_available",
        lambda: None,
    )

    with pytest.raises(
        ValueError,
        match=rf"required values must not be empty: .*{column} row\(s\) \[0\]",
    ):
        generate_dynamic_data(config_ieee14)

    assert marker.read_text() == "must survive validation failure"
