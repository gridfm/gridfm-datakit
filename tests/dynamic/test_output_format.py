"""Tests for the dynamic output format written by ``_save_generated_data``.

These exercise the output contract directly with synthetic results, so they run
without pypowsybl/Dynawo/Julia:

- Zarr curves have axis order (n_scenarios, n_variables, n_timesteps) (arch. §8).
- Scenarios with different timestep counts are NaN-padded to the longest run.
- Static (Parquet) and dynamic (Zarr) outputs are joinable by scenario_index,
  even when a scenario contributes to only one of them.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gridfm_datakit.dynamic import DynamicResults
from gridfm_datakit.dynamic.generate_dynamic import _save_generated_data
from gridfm_datakit.utils.param_handler import NestedNamespace

zarr = pytest.importorskip("zarr")


def _result(
    scenario_index,
    *,
    perturbation_index=0,
    static=True,
    dynamic=True,
    n_timesteps=5,
    n_variables=3,
):
    pf = None
    if static:
        # Same keys as pf_post_processing's output — the contract the dynamic
        # saver consumes. Y_bus / runtime included so the fixture stays faithful
        # to it (both are exported, at parity with the static pipeline).
        pf = {
            "bus": np.full((4, 5), scenario_index, dtype=float),
            "gen": np.full((2, 4), scenario_index, dtype=float),
            "branch": np.full((6, 6), scenario_index, dtype=float),
            "Y_bus": np.full((7, 5), scenario_index, dtype=float),
            "runtime": np.full((1, 2), scenario_index, dtype=float),
        }
    dr = None
    if dynamic:
        cols = [f"v{i}" for i in range(n_variables)]
        df = pd.DataFrame(
            np.full((n_timesteps, n_variables), scenario_index, dtype=float),
            columns=cols,
        )
        df.index.name = "time"
        dr = DynamicResults(dynamic_results=df, report="{}")
    return {
        "pf_data": pf,
        "dynamic_results": dr,
        "scenario_index": scenario_index,
        "perturbation_index": perturbation_index,
    }


def _save(results, tmp_path):
    out = tmp_path / "dyn"
    out.mkdir(parents=True)
    file_paths = {}
    _save_generated_data(
        all_results=results,
        output_dir=out,
        file_paths=file_paths,
        config=NestedNamespace(dummy=1),
        seed=0,
    )
    curves_group = zarr.open(str(out / "dynamic_results.zarr"), mode="r")
    metadata = json.loads((out / "metadata.json").read_text())
    return out, curves_group, metadata


def test_zarr_axis_order_is_scenarios_variables_timesteps(tmp_path):
    out, grp, meta = _save([_result(0), _result(1)], tmp_path)
    assert grp["curves"].shape == (2, 3, 5)  # (n_scenarios, n_variables, n_timesteps)
    assert meta["n_variables"] == 3
    assert meta["n_timesteps"] == 5
    # curves are transposed from the (n_timesteps, n_variables) DataFrame
    assert np.all(grp["curves"][0] == 0.0)
    assert np.all(grp["curves"][1] == 1.0)


def test_ragged_scenarios_are_nan_padded(tmp_path):
    out, grp, meta = _save(
        [_result(0, n_timesteps=5), _result(1, n_timesteps=3)],
        tmp_path,
    )
    assert grp["curves"].shape == (2, 3, 5)
    assert meta["timesteps_per_scenario"] == [5, 3]
    # the short scenario's tail is NaN, its valid region is not
    assert np.isnan(grp["curves"][1, :, 3:]).all()
    assert not np.isnan(grp["curves"][1, :, :3]).any()


def test_parquet_rows_carry_scenario_index(tmp_path):
    out, _, _ = _save([_result(7), _result(9)], tmp_path)
    bus = pd.read_parquet(out / "bus_data.parquet")
    assert "scenario_index" in bus.columns
    # 4 bus rows per scenario, tagged with the right scenario
    assert bus["scenario_index"].tolist() == [7, 7, 7, 7, 9, 9, 9, 9]


def test_all_scenarios_failing_raises_instead_of_keeping_stale_outputs(tmp_path):
    """An empty result set must not silently leave a previous run's files in place."""
    out = tmp_path / "dyn"
    out.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="every scenario failed"):
        _save_generated_data(
            all_results=[],
            output_dir=out,
            file_paths={},
            config=NestedNamespace(dummy=1),
            seed=0,
        )


def test_variable_count_mismatch_raises(tmp_path):
    """Samples monitoring different variables cannot share one Zarr store."""
    out = tmp_path / "dyn"
    out.mkdir(parents=True)
    with pytest.raises(ValueError, match="disagree on the number of variables"):
        _save_generated_data(
            all_results=[_result(0, n_variables=3), _result(1, n_variables=2)],
            output_dir=out,
            file_paths={},
            config=NestedNamespace(dummy=1),
            seed=0,
        )


def test_static_snapshot_is_at_parity_with_static_pipeline(tmp_path):
    """Y-bus and runtime are exported too, like generate._save_generated_data."""
    out, _, _ = _save([_result(7), _result(9)], tmp_path)

    y_bus = pd.read_parquet(out / "y_bus_data.parquet")
    assert list(y_bus.columns) == ["scenario_index", "perturbation_index"] + [
        "load_scenario_idx",
        "index1",
        "index2",
        "G",
        "B",
    ]
    assert y_bus["scenario_index"].tolist() == [7] * 7 + [9] * 7

    runtime = pd.read_parquet(out / "runtime_data.parquet")
    assert list(runtime.columns) == [
        "scenario_index",
        "perturbation_index",
        "load_scenario_idx",
        "ac",
    ]
    assert runtime["scenario_index"].tolist() == [7, 9]


def test_topology_perturbations_labeled_by_composite_key(tmp_path):
    # one scenario expanded into two topology perturbations
    results = [
        _result(0, perturbation_index=0),
        _result(0, perturbation_index=1),
    ]
    out, grp, meta = _save(results, tmp_path)

    bus = pd.read_parquet(out / "bus_data.parquet")
    assert {"scenario_index", "perturbation_index"} <= set(bus.columns)
    static_keys = sorted(set(zip(bus["scenario_index"], bus["perturbation_index"])))
    dyn_keys = sorted(
        zip(
            np.asarray(grp["scenario_index"]).tolist(),
            np.asarray(grp["perturbation_index"]).tolist(),
        ),
    )
    assert static_keys == [(0, 0), (0, 1)]
    assert dyn_keys == [(0, 0), (0, 1)]
    assert meta["dynamic_perturbation_index"] == [0, 1]
    # the two perturbations are distinct slices of the same scenario
    assert grp["curves"].shape[0] == 2


def test_features_and_labels_joinable_when_membership_differs(tmp_path):
    # scenario 7: both; 9: static only; 11: dynamic only
    results = [
        _result(7),
        _result(9, dynamic=False),
        _result(11, static=False),
    ]
    out, grp, meta = _save(results, tmp_path)

    static_ids = set(pd.read_parquet(out / "bus_data.parquet")["scenario_index"])
    dynamic_ids = set(np.asarray(grp["scenario_index"]).tolist())

    assert static_ids == {7, 9}
    assert dynamic_ids == {7, 11}
    assert meta["static_scenario_index"] == [7, 9]
    assert meta["dynamic_scenario_index"] == [7, 11]
    # only scenario 7 is joinable across both modalities
    assert static_ids & dynamic_ids == {7}
    # and its curve slice is addressable via the scenario_index coordinate
    i7 = np.asarray(grp["scenario_index"]).tolist().index(7)
    assert np.all(grp["curves"][i7] == 7.0)
