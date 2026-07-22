"""End-to-end tests for the dynamic generation pipeline.

Coverage note: these exercise the happy path (a run completes, its static
snapshot passes the shared validation suite, curves carry a time axis). Output
shape, padding and join keys are covered in test_output_format.py; input loading
in test_load_raw_inputs.py; solver/load-flow config rejection in
test_solver_parameters.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from markers import needs_dynawo

# Full end-to-end pipeline: OPF via Julia, then Dynawo.
pytestmark = needs_dynawo


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------


def test_generate_dynamic(config_ieee14):
    from gridfm_datakit.dynamic.generate_dynamic import generate_dynamic_data

    file_paths = generate_dynamic_data(config_ieee14)
    assert Path(file_paths["dynamic_results"]).exists()
    assert Path(file_paths["metadata"]).exists()


def test_generated_data_passes_static_validation(config_ieee14):
    """The dynamic pipeline's PF snapshot must satisfy the same physics checks
    (Y-bus consistency, Kirchhoff balance, generator limits, ...) as the static
    pipeline's output."""
    from gridfm_datakit.dynamic.generate_dynamic import generate_dynamic_data
    from gridfm_datakit.validation import validate_dynamic_data

    file_paths = generate_dynamic_data(config_ieee14)
    assert validate_dynamic_data(file_paths, mode="pf", sn_mva=100.0)


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


def test_final_state_values_reach_the_output(config_ieee14):
    """A FinalStateValue row in the variables table produces a keyed table.

    Guards the gap where these were mapped and simulated but never persisted: the
    run consumed them, Dynawo computed them, and nothing came out.
    """
    import pandas as pd

    from gridfm_datakit.dynamic.generate_dynamic import generate_dynamic_data

    file_paths = generate_dynamic_data(config_ieee14)

    path = Path(file_paths["final_state_values"])
    assert path.is_file()
    frame = pd.read_parquet(path)
    assert list(frame.columns[:2]) == ["scenario_index", "perturbation_index"]

    metadata = json.loads(Path(file_paths["metadata"]).read_text())
    names = metadata["final_state_value_names"]
    assert names, "the run monitors a FinalStateValue row, so names must be recorded"
    assert list(frame.columns[2:]) == names
    # one row per sample, and the values are real numbers rather than NaN padding
    assert len(frame) == metadata["n_samples"]
    assert frame[names].notna().all().all()


def test_validate_flag_runs_the_validation_suite(config_ieee14, monkeypatch):
    """dynamic.validate: true runs the shared checks over the static snapshot."""
    from gridfm_datakit.dynamic import generate_dynamic as gd

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


def test_validation_is_off_by_default(config_ieee14, monkeypatch):
    from gridfm_datakit.dynamic import generate_dynamic as gd

    def _fail(*_args, **_kwargs):
        raise AssertionError("validation must not run unless dynamic.validate is set")

    monkeypatch.setattr("gridfm_datakit.validation.validate_dynamic_data", _fail)
    gd.generate_dynamic_data(config_ieee14)
