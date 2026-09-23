"""Regression tests for distributed generation orchestration."""

import numpy as np

import gridfm_datakit.process.process_network as process_network
from gridfm_datakit.generate import _split_range


class RecordingQueue:
    """Minimal queue seam used to verify progress accounting."""

    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def test_split_range_matches_numpy_array_split():
    """The allocation-free bounds preserve the prior chunk boundaries."""
    for stop, n_parts in ((1, 1), (10, 3), (1001, 2), (10_000, 16)):
        bounds = _split_range(0, stop, n_parts)
        expected = np.array_split(np.arange(stop), n_parts)
        assert bounds == [(int(part[0]), int(part[-1]) + 1) for part in expected]


def test_chunk_slice_keeps_global_scenario_indices(monkeypatch, tmp_path):
    """Sliced task tensors still write the original global scenario indices."""
    seen = []

    def fake_pf_mode(
        net,
        scenarios,
        scenario_index,
        topology_generator,
        generation_generator,
        admittance_generator,
        local_processed_data,
        error_log_file,
        include_dc_res,
        pf_fast,
        dcpf_fast,
        jl,
        pf_solver="powermodel",
        *,
        meta=None,
        scenario_data_index=None,
    ):
        seen.append((scenario_index, scenario_data_index))
        local_processed_data.append(scenario_index)
        return local_processed_data

    monkeypatch.setattr(process_network, "process_scenario_pf_mode", fake_pf_mode)
    monkeypatch.setattr(process_network, "_worker_jl", object())
    progress = RecordingQueue()
    scenarios = np.zeros((2, 3, 2))

    error, traceback_text, processed = process_network.process_scenario_chunk(
        "pf",
        0,
        3,
        scenarios,
        object(),
        progress,
        object(),
        object(),
        object(),
        str(tmp_path / "error.log"),
        False,
        True,
        True,
        None,
        10,
        42,
        scenario_index_offset=7,
    )

    assert error is None
    assert traceback_text is None
    assert processed == [7, 8, 9]
    assert seen == [(7, 0), (8, 1), (9, 2)]
    assert progress.items == [1, 1, 1]


def test_chunk_error_backfills_only_unreported_progress(monkeypatch, tmp_path):
    """A mid-chunk failure must not leak progress ticks into the next chunk."""
    calls = 0

    def failing_pf_mode(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("deliberate failure")
        return args[6]

    monkeypatch.setattr(process_network, "process_scenario_pf_mode", failing_pf_mode)
    monkeypatch.setattr(process_network, "_worker_jl", object())
    progress = RecordingQueue()

    error, _, processed = process_network.process_scenario_chunk(
        "pf",
        0,
        3,
        np.zeros((1, 3, 2)),
        object(),
        progress,
        object(),
        object(),
        object(),
        str(tmp_path / "error.log"),
        False,
        True,
        True,
        None,
        10,
        42,
    )

    assert isinstance(error, RuntimeError)
    assert processed is None
    assert progress.items == [1, 1, 1]
