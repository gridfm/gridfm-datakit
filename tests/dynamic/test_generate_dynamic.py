"""End-to-end tests for the dynamic generation pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from markers import needs_dynawo

from gridfm_datakit.dynamic import generate_dynamic as gd
from gridfm_datakit.utils.param_handler import NestedNamespace

# No module-level pytestmark: only the tests that simulate carry @needs_dynawo.


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------


@needs_dynawo
def test_generate_dynamic(config_ieee14):
    from gridfm_datakit.dynamic.generate_dynamic import generate_dynamic_data

    file_paths = generate_dynamic_data(config_ieee14)
    assert Path(file_paths["dynamic_results"]).exists()
    assert Path(file_paths["metadata"]).exists()


@needs_dynawo
def test_generated_data_passes_static_validation(config_ieee14):
    """The dynamic pipeline's PF snapshot must satisfy the same physics checks
    (Y-bus consistency, Kirchhoff balance, generator limits, ...) as the static
    pipeline's output."""
    from gridfm_datakit.dynamic.generate_dynamic import generate_dynamic_data
    from gridfm_datakit.validation import validate_dynamic_data

    file_paths = generate_dynamic_data(config_ieee14)
    assert validate_dynamic_data(file_paths, mode="pf", sn_mva=100.0)


@needs_dynawo
def test_datasets_without_event_index_pass_static_validation(config_ieee14, tmp_path):
    import pandas as pd

    from gridfm_datakit.dynamic.generate_dynamic import generate_dynamic_data
    from gridfm_datakit.validation import validate_dynamic_data

    config_ieee14.dynamic.solver_parameters.stop_time = 60.0
    file_paths = generate_dynamic_data(config_ieee14)

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    legacy_paths = {}
    for key in ("bus_data", "branch_data", "gen_data", "y_bus_data", "runtime_data"):
        legacy_paths[key] = str(legacy / f"{key}.parquet")
        pd.read_parquet(file_paths[key]).drop(columns="event_index").to_parquet(
            legacy_paths[key],
        )
    assert validate_dynamic_data(legacy_paths, mode="pf", sn_mva=100.0)


@needs_dynawo
def test_curves_carry_a_time_axis_in_seconds(config_ieee14):
    """Curves are unusable as labels without knowing which instant each column is."""
    import numpy as np
    import zarr

    from gridfm_datakit.dynamic.generate_dynamic import generate_dynamic_data

    file_paths = generate_dynamic_data(config_ieee14)
    store = zarr.open(file_paths["dynamic_results"], mode="r")
    curves = np.asarray(store["curves"])
    time = np.asarray(store["time"])

    # one time value per curve column, per scenario
    assert time.shape == (curves.shape[0], curves.shape[2])
    # and it is the config's simulation window, in seconds
    sp = config_ieee14.dynamic.solver_parameters
    assert np.nanmin(time) == sp.start_time
    assert np.nanmax(time) == sp.stop_time


@needs_dynawo
def test_final_state_values_reach_the_output(config_ieee14):
    """A FinalStateValue row in the variables table produces a keyed table."""
    import pandas as pd

    from gridfm_datakit.dynamic.generate_dynamic import generate_dynamic_data

    file_paths = generate_dynamic_data(config_ieee14)

    path = Path(file_paths["final_state_values"])
    assert path.is_file()
    frame = pd.read_parquet(path)
    assert list(frame.columns[:3]) == [
        "scenario_index",
        "perturbation_index",
        "event_index",
    ]

    metadata = json.loads(Path(file_paths["metadata"]).read_text())
    names = metadata["final_state_value_names"]
    assert names, "the run monitors a FinalStateValue row, so names must be recorded"
    assert list(frame.columns[3:]) == names
    assert len(frame) == metadata["n_samples"]
    assert frame[names].notna().all().all()


@needs_dynawo
def test_several_final_state_values_each_get_a_column(config_ieee14_multi_fsv):
    """One monitored variable cannot tell one column per variable from
    one column, last one wins."""
    import pandas as pd

    from gridfm_datakit.dynamic.generate_dynamic import generate_dynamic_data

    file_paths = generate_dynamic_data(config_ieee14_multi_fsv)

    frame = pd.read_parquet(file_paths["final_state_values"])
    metadata = json.loads(Path(file_paths["metadata"]).read_text())
    names = metadata["final_state_value_names"]

    assert len(names) == 3, names
    key = ["scenario_index", "perturbation_index", "event_index"]
    assert list(frame.columns) == key + names
    assert frame[names].notna().all().all()
    # distinct models, so not one number repeated
    assert frame[names].iloc[0].nunique() > 1

    # each column names the model it came from
    for model_id in ("_GEN____1_SM", "_GEN____3_SM", "_GEN____6_SM"):
        assert any(model_id in name for name in names), (model_id, names)


@needs_dynawo
def test_events_are_recorded_for_every_sample(config_ieee14):
    import pandas as pd

    events = pd.read_csv(config_ieee14.dynamic.input_files.events_file)

    file_paths = gd.generate_dynamic_data(config_ieee14)
    metadata = json.loads(Path(file_paths["metadata"]).read_text())

    key = ["scenario_index", "perturbation_index", "event_index"]
    recorded = pd.read_parquet(file_paths["events"])
    bus = pd.read_parquet(file_paths["bus_data"])
    assert len(recorded) == metadata["n_samples"] * len(events)
    assert set(map(tuple, recorded[key].to_numpy())) == set(
        map(tuple, bus[key].to_numpy()),
    )


@needs_dynawo
def test_random_events_are_drawn_per_event_variant(config_ieee14):
    import pandas as pd
    import pypowsybl.network as pn

    config_ieee14.dynamic.solver_parameters.stop_time = 60.0
    del config_ieee14.dynamic.input_files.events_file
    config_ieee14.dynamic.event_perturbation = NestedNamespace(
        type="random",
        n_event_variants=3,
        scenarios=[
            {
                "name": "generator_trip",
                "start_time": {"distribution": "uniform", "low": 20, "high": 50},
                "events": [
                    {
                        "type": "Disconnect",
                        "target": {"element": "generator", "distance": [0, 2]},
                    },
                ],
            },
        ],
    )

    file_paths = gd.generate_dynamic_data(config_ieee14)
    metadata = json.loads(Path(file_paths["metadata"]).read_text())
    error_log = Path(file_paths["error_log"])
    assert metadata["n_samples"] == 3, error_log.read_text()

    events = pd.read_parquet(file_paths["events"])
    assert events["event_index"].tolist() == [0, 1, 2]
    assert (events["scenario"] == "generator_trip").all()
    generators = set(pn.load(config_ieee14.network.file).get_generators().index)
    assert set(events["static_id"]) <= generators
    assert len(set(zip(events["static_id"], events["start_time"]))) == 3
    assert events["start_time"].between(20, 50, inclusive="left").all()

    bus = pd.read_parquet(file_paths["bus_data"])
    samples = [
        frame.drop(columns="event_index").reset_index(drop=True)
        for _, frame in bus.groupby("event_index")
    ]
    assert len(samples) == 3
    assert all(sample.equals(samples[0]) for sample in samples)


@needs_dynawo
def test_node_faults_are_drawn_per_event_variant(config_ieee14):
    import pandas as pd
    import pypowsybl.network as pn

    config_ieee14.dynamic.solver_parameters.stop_time = 60.0
    del config_ieee14.dynamic.input_files.events_file
    config_ieee14.dynamic.event_perturbation = NestedNamespace(
        type="random",
        n_event_variants=2,
        scenarios=[
            {
                "name": "node_fault",
                "start_time": {"distribution": "uniform", "low": 20, "high": 50},
                "events": [
                    {
                        "type": "NodeFault",
                        "target": {"element": "bus", "distance": [0, 1]},
                        "params": {
                            "fault_time": 0.1,
                            "r_pu": 0,
                            "x_pu": {
                                "distribution": "uniform",
                                "low": 0.01,
                                "high": 0.1,
                            },
                        },
                    },
                ],
            },
        ],
    )

    file_paths = gd.generate_dynamic_data(config_ieee14)
    metadata = json.loads(Path(file_paths["metadata"]).read_text())
    error_log = Path(file_paths["error_log"])
    assert metadata["n_samples"] == 2, error_log.read_text()

    events = pd.read_parquet(file_paths["events"])
    assert (events["event_name"] == "NodeFault").all()
    buses = set(pn.load(config_ieee14.network.file).get_bus_breaker_view_buses().index)
    assert set(events["static_id"]) <= buses
    x_pu = [
        float(dict(item.split("=") for item in params.split(";"))["x_pu"])
        for params in events["params"]
    ]
    assert all(0.01 <= value < 0.1 for value in x_pu)


@needs_dynawo
def test_multi_event_scenarios_are_drawn_per_event_variant(config_ieee14):
    import pandas as pd

    config_ieee14.dynamic.solver_parameters.stop_time = 60.0
    del config_ieee14.dynamic.input_files.events_file
    config_ieee14.dynamic.event_perturbation = NestedNamespace(
        type="random",
        n_event_variants=2,
        scenarios=[
            {
                "name": "trip_then_fault",
                "start_time": {"distribution": "uniform", "low": 20, "high": 50},
                "events": [
                    {
                        "type": "Disconnect",
                        "target": {"element": "generator", "distance": [0, 2]},
                    },
                    {
                        "type": "NodeFault",
                        "target": {"element": "bus", "distance": [0, 1]},
                        "delay": {"distribution": "uniform", "low": 0.1, "high": 0.3},
                        "params": {"fault_time": 0.1, "r_pu": 0, "x_pu": 0.05},
                    },
                ],
            },
        ],
    )

    file_paths = gd.generate_dynamic_data(config_ieee14)
    metadata = json.loads(Path(file_paths["metadata"]).read_text())
    error_log = Path(file_paths["error_log"])
    assert metadata["n_samples"] == 2, error_log.read_text()

    events = pd.read_parquet(file_paths["events"])
    assert len(events) == 4
    for _, group in events.groupby("event_index"):
        assert group["event_name"].tolist() == ["Disconnect", "NodeFault"]
        start, later = group["start_time"]
        assert 0.1 <= later - start < 0.3


@needs_dynawo
def test_fixed_event_targets_are_simulated(config_ieee14):
    import pandas as pd

    config_ieee14.dynamic.solver_parameters.stop_time = 60.0
    del config_ieee14.dynamic.input_files.events_file
    config_ieee14.dynamic.event_perturbation = NestedNamespace(
        type="random",
        n_event_variants=2,
        scenarios=[
            {
                "name": "fixed_trip",
                "start_time": {"distribution": "uniform", "low": 20, "high": 50},
                "events": [
                    {"type": "Disconnect", "target": {"static_id": "_GEN____2_SM"}},
                ],
            },
        ],
    )

    file_paths = gd.generate_dynamic_data(config_ieee14)
    metadata = json.loads(Path(file_paths["metadata"]).read_text())
    error_log = Path(file_paths["error_log"])
    assert metadata["n_samples"] == 2, error_log.read_text()

    events = pd.read_parquet(file_paths["events"])
    assert len(events) == 2
    assert (events["static_id"] == "_GEN____2_SM").all()
    assert events["start_time"].nunique() == 2


@needs_dynawo
@pytest.mark.parametrize(
    "static_id",
    [
        pytest.param("_BUS___10_TN", id="bus"),
        pytest.param("_BUS___10-BUS___11-1_AC", id="line"),
        pytest.param("_BUS____4-BUS____7-1_PT", id="transformer"),
    ],
)
def test_disconnecting_a_bus_line_or_transformer_moves_the_curves(
    config_ieee14,
    static_id,
):
    import numpy as np
    import pandas as pd
    import zarr

    config_ieee14.dynamic.solver_parameters.stop_time = 60.0
    del config_ieee14.dynamic.input_files.events_file
    config_ieee14.dynamic.event_perturbation = NestedNamespace(type="none")
    file_paths = gd.generate_dynamic_data(config_ieee14)
    base = np.asarray(zarr.open(file_paths["dynamic_results"], mode="r")["curves"])

    config_ieee14.dynamic.event_perturbation = NestedNamespace(
        type="random",
        n_event_variants=1,
        scenarios=[
            {
                "name": "fixed_disconnect",
                "start_time": 20,
                "events": [
                    {"type": "Disconnect", "target": {"static_id": static_id}},
                ],
            },
        ],
    )
    file_paths = gd.generate_dynamic_data(config_ieee14)
    metadata = json.loads(Path(file_paths["metadata"]).read_text())
    error_log = Path(file_paths["error_log"]).read_text()
    assert metadata["n_samples"] == 1, error_log
    assert metadata["n_samples_with_curves"] == 1
    assert not re.search(r"^\[dynamic\]", error_log, re.MULTILINE), error_log

    events = pd.read_parquet(file_paths["events"])
    assert len(events) == 1
    assert events["event_name"].iloc[0] == "Disconnect"
    assert events["static_id"].iloc[0] == static_id
    assert events["start_time"].iloc[0] == 20

    curves = np.asarray(zarr.open(file_paths["dynamic_results"], mode="r")["curves"])
    assert curves.shape != base.shape or not np.allclose(
        curves,
        base,
        equal_nan=True,
    ), static_id


@needs_dynawo
def test_no_events_writes_no_event_table(config_ieee14):
    del config_ieee14.dynamic.input_files.events_file
    config_ieee14.dynamic.event_perturbation = NestedNamespace(type="none")

    file_paths = gd.generate_dynamic_data(config_ieee14)
    metadata = json.loads(Path(file_paths["metadata"]).read_text())

    assert metadata["n_samples"] == 1
    assert "events" not in file_paths


def test_a_malformed_event_block_raises_before_the_output_is_set_up(
    config_ieee14,
    monkeypatch,
):
    def _fail(args):
        raise AssertionError("the output was set up before the inputs were read")

    monkeypatch.setattr(gd, "_validate_dynamic_config", lambda args: None)
    monkeypatch.setattr(gd, "_setup_environment", _fail)
    config_ieee14.dynamic.event_perturbation = NestedNamespace(type="sometimes")
    with pytest.raises(ValueError, match=r"dynamic\.event_perturbation\.type"):
        gd.generate_dynamic_data(config_ieee14)


def test_an_unknown_fixed_target_raises_before_the_output_is_set_up(
    config_ieee14,
    monkeypatch,
):
    def _fail(args):
        raise AssertionError("the output was set up before the targets were checked")

    monkeypatch.setattr(gd, "_validate_dynamic_config", lambda args: None)
    monkeypatch.setattr(gd, "_setup_environment", _fail)
    del config_ieee14.dynamic.input_files.events_file
    config_ieee14.dynamic.event_perturbation = NestedNamespace(
        type="random",
        scenarios=[
            {
                "name": "fixed_trip",
                "start_time": 1.0,
                "events": [
                    {"type": "Disconnect", "target": {"static_id": "_NO_SUCH_GEN"}},
                ],
            },
        ],
    )
    with pytest.raises(ValueError, match=r"_NO_SUCH_GEN"):
        gd.generate_dynamic_data(config_ieee14)


@needs_dynawo
def test_validate_flag_runs_the_validation_suite(config_ieee14, monkeypatch):
    config_ieee14.dynamic.validate = True
    seen = {}

    def _spy(file_paths, mode="pf", sn_mva=100.0):
        seen["file_paths"] = file_paths
        seen["mode"] = mode
        return True

    monkeypatch.setattr(
        "gridfm_datakit.validation.validate_dynamic_data",
        _spy,
    )
    file_paths = gd.generate_dynamic_data(config_ieee14)

    assert seen["file_paths"] is file_paths
    assert seen["mode"] == config_ieee14.settings.mode


@needs_dynawo
def test_validation_is_off_by_default(config_ieee14, monkeypatch):
    def _fail(*_args, **_kwargs):
        raise AssertionError("validation must not run unless dynamic.validate is set")

    monkeypatch.setattr("gridfm_datakit.validation.validate_dynamic_data", _fail)
    gd.generate_dynamic_data(config_ieee14)


@needs_dynawo
def test_topology_perturbation_expands_scenarios_into_samples(config_ieee14):
    """Every requested sample (scenarios x variants) is written or logged as failed."""
    import pandas as pd

    n_scenarios, n_variants = 2, 2
    config_ieee14.load.scenarios = n_scenarios
    config_ieee14.topology_perturbation = NestedNamespace(
        type="random",
        k=1,
        n_topology_variants=n_variants,
        elements=["branch"],
    )
    # four Dynawo runs: keep them short, but past the t=50 s event
    config_ieee14.dynamic.solver_parameters.stop_time = 60.0

    file_paths = gd.generate_dynamic_data(config_ieee14)
    metadata = json.loads(Path(file_paths["metadata"]).read_text())

    requested = n_scenarios * n_variants
    error_log = Path(file_paths["error_log"]).read_text()
    # a scenario failing before it expands has no known sample count
    assert not re.findall(
        r"^\[dynamic\] scenario \d+ failed",
        error_log,
        flags=re.MULTILINE,
    ), error_log
    failures = re.findall(
        r"^\[dynamic\] scenario \d+ perturbation \d+ (event \d+ )?failed",
        error_log,
        flags=re.MULTILINE,
    )
    assert metadata["n_samples"] + len(failures) == requested

    keys = set(
        zip(metadata["static_scenario_index"], metadata["static_perturbation_index"]),
    )
    assert len(keys) == metadata["n_samples"]
    assert {scenario for scenario, _ in keys} <= set(range(n_scenarios))
    assert {perturbation for _, perturbation in keys} <= set(range(n_variants))

    bus = pd.read_parquet(file_paths["bus_data"])
    assert set(zip(bus["scenario_index"], bus["perturbation_index"])) == keys


def _valid_config_skeleton() -> dict:
    """The smallest config _validate_dynamic_config accepts, as a plain dict."""
    return {
        "network": {"name": "IEEE14", "reader": "powsybl"},
        "load": {"scenarios": 1},
        "dynamic": {"dynamic_solver": "dynawo"},
        "settings": {"data_dir": "out"},
    }


class TestConfigValidation:
    @staticmethod
    def _validate(overrides: dict) -> None:
        config = _valid_config_skeleton()
        for block, values in overrides.items():
            if values is None:
                config.pop(block, None)
            else:
                config.setdefault(block, {}).update(values)
        gd._validate_dynamic_config(NestedNamespace(**config))

    def test_native_reader_raises(self):
        with pytest.raises(ValueError, match="require network.reader='powsybl'"):
            self._validate({"network": {"reader": "native"}})

    def test_missing_network_block_raises_about_the_reader(self):
        with pytest.raises(ValueError, match="require network.reader='powsybl'"):
            self._validate({"network": None})

    def test_missing_dynamic_block_raises(self):
        with pytest.raises(ValueError, match="missing the 'dynamic:' block"):
            self._validate({"dynamic": None})

    def test_missing_dynamic_solver_raises(self):
        with pytest.raises(ValueError, match="missing dynamic.dynamic_solver"):
            self._validate({"dynamic": {"dynamic_solver": None}})

    def test_removed_output_dir_key_raises(self):
        with pytest.raises(ValueError, match="dynamic.output_dir has been removed"):
            self._validate({"dynamic": {"output_dir": "somewhere"}})

    @pytest.mark.parametrize("scenarios", [0, -1, None])
    def test_non_positive_scenario_count_raises(self, scenarios):
        with pytest.raises(ValueError, match="load.scenarios must be >= 1"):
            self._validate({"load": {"scenarios": scenarios}})

    def test_missing_load_block_raises(self):
        with pytest.raises(ValueError, match="load.scenarios must be >= 1"):
            self._validate({"load": None})

    @needs_dynawo
    def test_a_valid_config_passes(self):
        self._validate({})  # must not raise

    def test_an_unknown_solver_skips_the_dynawo_install_check(self, monkeypatch):
        def _fail():
            raise AssertionError("must not check for Dynawo on another solver")

        monkeypatch.setattr(
            "gridfm_datakit.dynamic.dynawo.api.check_dynawo_available",
            _fail,
        )
        self._validate({"dynamic": {"dynamic_solver": "some_other_solver"}})


class _StopPipeline(Exception):
    """The config was accepted; stop before anything simulates."""


class TestAcceptedConfigForms:
    @pytest.fixture
    def accepted(self, monkeypatch):
        """Capture the normalised config, then bail out."""
        seen = {}

        def _spy(args):
            seen["args"] = args
            raise _StopPipeline

        monkeypatch.setattr(gd, "_validate_dynamic_config", _spy)
        return seen

    @pytest.fixture
    def config_file(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(_valid_config_skeleton()))
        return path

    def test_path_string(self, accepted, config_file):
        with pytest.raises(_StopPipeline):
            gd.generate_dynamic_data(str(config_file))
        assert accepted["args"].dynamic.dynamic_solver == "dynawo"
        assert accepted["args"].load.scenarios == 1

    def test_path_object(self, accepted, config_file):
        """A Path used to fail with an AttributeError on the path object."""
        with pytest.raises(_StopPipeline):
            gd.generate_dynamic_data(config_file)
        assert accepted["args"].dynamic.dynamic_solver == "dynawo"

    def test_dict(self, accepted):
        with pytest.raises(_StopPipeline):
            gd.generate_dynamic_data(_valid_config_skeleton())
        assert isinstance(accepted["args"], NestedNamespace)
        assert accepted["args"].network.reader == "powsybl"

    def test_nested_namespace_is_used_as_is(self, accepted):
        config = NestedNamespace(**_valid_config_skeleton())
        with pytest.raises(_StopPipeline):
            gd.generate_dynamic_data(config)
        assert accepted["args"] is config

    def test_every_form_yields_the_same_config(self, monkeypatch, config_file):
        seen = []

        def _spy(args):
            seen.append(args.to_dict())
            raise _StopPipeline

        monkeypatch.setattr(gd, "_validate_dynamic_config", _spy)
        for form in (str(config_file), config_file, _valid_config_skeleton()):
            with pytest.raises(_StopPipeline):
                gd.generate_dynamic_data(form)
        assert seen[0] == seen[1] == seen[2] == _valid_config_skeleton()

    @pytest.mark.parametrize("config", [42, ["network"], None, object()])
    def test_unsupported_type_names_the_accepted_forms(self, config):
        with pytest.raises(TypeError, match="str or os.PathLike"):
            gd.generate_dynamic_data(config)

    def test_yaml_that_is_not_a_mapping_raises(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("- just\n- a list\n")
        with pytest.raises(ValueError, match="must contain a YAML mapping"):
            gd.generate_dynamic_data(path)

    def test_missing_config_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            gd.generate_dynamic_data(tmp_path / "nope.yaml")
