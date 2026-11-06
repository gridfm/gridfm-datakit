"""Timeout tests with real spawned pools and inexpensive simulated scenarios."""

import multiprocessing
import os
from pathlib import Path
from queue import Queue
import time
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from tqdm import tqdm
import yaml

from gridfm_datakit.config import validate_static_config
from gridfm_datakit.checkpoint import GenerationCheckpoint
from gridfm_datakit.generate import _setup_environment
from gridfm_datakit.process import process_network
from gridfm_datakit.process.scenario_timeout import (
    ScenarioTimeout,
    ScenarioTimeoutPool,
    _collect_timed_results,
)
from gridfm_datakit.utils.random_seed import _DEFAULT_SEED_POLICY


_OPTIONS = {}


def _fake_scenario(*args: Any, **kwargs: Any) -> list:
    """Simulate native hangs, ordinary exceptions, and process crashes by index."""
    index = args[2]
    if index in _OPTIONS.get("hang", []):
        time.sleep(5)
    if index in _OPTIONS.get("error", []):
        raise ValueError("scenario failed normally")
    if index in _OPTIONS.get("crash", []):
        os._exit(23)
    args[6].append((index, float(np.random.random())))
    return args[6]


def _initialize_fake_worker(
    options: dict, queue: Any, timeout: float, skipped: set[int],
) -> None:
    """Install the real scenario loop with a simulated solver in a spawned child."""
    global _OPTIONS
    _OPTIONS = options
    time.sleep(options.get("startup_delay", 0))
    process_network._initialize_scenario_worker(
        "opf",
        None,
        None,
        None,
        None,
        options["error_log"],
        False,
        True,
        True,
        None,
        10,
        42,
        "powermodel",
        {},
        progress_queue=queue,
        scenario_timeout_sec=timeout,
        skipped_scenarios=skipped,
    )
    process_network._worker_jl = object()
    process_network.process_scenario_opf_mode = _fake_scenario


@pytest.mark.timeout(45)
def test_hung_scenarios_are_skipped_without_changing_other_draws(
    tmp_path: Path,
) -> None:
    """Kill offenders, recreate pools, and preserve the draws of surviving indices."""
    previous = {process.pid for process in multiprocessing.active_children()}
    options = {"hang": [6, 9], "error_log": str(tmp_path / "error.log")}
    reports = []
    with (tmp_path / "progress.log").open("w") as log, tqdm(total=6, file=log) as bar:
        with ScenarioTimeoutPool(
            multiprocessing.get_context("spawn"),
            2,
            _initialize_fake_worker,
            (options,),
            process_network._process_scenario_worker,
            0.5,
            set(),
            reports.append,
        ) as pool:
            results = pool.collect(
                [(5, np.zeros((1, 3, 2))), (8, np.zeros((1, 3, 2)))], bar,
            )
            assert pool.skipped_scenarios == {6, 9}
        assert bar.n == 6
    actual = [record for start in sorted(results) for record in results[start][4]]
    expected = [
        (
            index,
            float(
                np.random.RandomState(
                    _DEFAULT_SEED_POLICY.distributed_seed(42, index),
                ).random_sample(),
            ),
        )
        for index in (5, 7, 8, 10)
    ]
    assert actual == expected
    assert {error.scenario_index for error in reports} == {6, 9}
    assert not (
        {process.pid for process in multiprocessing.active_children()} - previous
    )


@pytest.mark.timeout(45)
def test_all_timeouts_and_persisted_skips_finish_without_hanging(
    tmp_path: Path,
) -> None:
    """Terminate finite retries even when every scenario is skipped."""
    options = {"hang": [0, 1], "error_log": str(tmp_path / "error.log")}
    reports = []
    with (tmp_path / "progress.log").open("w") as log, tqdm(total=3, file=log) as bar:
        with ScenarioTimeoutPool(
            multiprocessing.get_context("spawn"),
            1,
            _initialize_fake_worker,
            (options,),
            process_network._process_scenario_worker,
            0.5,
            {2},
            reports.append,
        ) as pool:
            results = pool.collect([(0, np.zeros((1, 3, 2)))], bar)
            assert pool.skipped_scenarios == {0, 1, 2}
            assert pool.pool is None
        assert bar.n == 3
    assert results[0][4] == []
    assert {error.scenario_index for error in reports} == {0, 1}


@pytest.mark.timeout(45)
@pytest.mark.parametrize("failure", ["error", "crash"])
def test_worker_failures_are_not_silently_skipped(tmp_path: Path, failure: str) -> None:
    """Propagate normal exceptions and hard worker exits instead of retrying them."""
    options = {failure: [0], "error_log": str(tmp_path / "error.log")}
    reports = []
    previous = {process.pid for process in multiprocessing.active_children()}
    with (tmp_path / "progress.log").open("w") as log, tqdm(total=1, file=log) as bar:
        with pytest.raises(RuntimeError, match="failed|exited unexpectedly"):
            with ScenarioTimeoutPool(
                multiprocessing.get_context("spawn"),
                1,
                _initialize_fake_worker,
                (options,),
                process_network._process_scenario_worker,
                5.0,
                set(),
                reports.append,
            ) as pool:
                pool.collect([(0, np.zeros((1, 1, 2)))], bar)
    assert reports == []
    assert not (
        {process.pid for process in multiprocessing.active_children()} - previous
    )


def test_done_events_are_drained_before_deadline_checks() -> None:
    """A completed scenario is not timed out because its event waited in the queue."""
    queue = Queue()
    started = time.monotonic() - 10
    queue.put(("start", 123, 7, started))
    queue.put(("done", 123, 7, started + 0.1))
    result = Mock()
    result.ready.return_value = True
    result.get.return_value = (7, 1, None, None, [])
    pool = Mock()
    pool._pool = []
    pool.apply_async.return_value = result
    bar = Mock()
    assert (
        _collect_timed_results(
            pool, [(7, np.zeros((1, 1, 2)))], queue, bar, 1.0, Mock(),
        )[7][0]
        == 7
    )
    bar.update.assert_called_once_with(1)


def test_finished_over_budget_scenario_still_times_out() -> None:
    """Enforce the worker's measured duration even if start/done arrive together."""
    queue = Queue()
    started = time.monotonic() - 10
    queue.put(("start", 123, 7, started))
    queue.put(("done", 123, 7, started + 2.0))
    pool = Mock()
    pool._pool = []
    with pytest.raises(ScenarioTimeout) as caught:
        _collect_timed_results(
            pool, [(7, np.zeros((1, 1, 2)))], queue, Mock(), 1.0, Mock(),
        )
    assert caught.value.scenario_index == 7
    assert caught.value.elapsed == 2.0


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), "600", True])
def test_invalid_timeout_preserves_output(tmp_path: Path, timeout: Any) -> None:
    """Reject invalid limits before overwriting a dataset."""
    config = yaml.safe_load(Path("scripts/config/default.yaml").read_text())
    config["settings"].update(
        data_dir=str(tmp_path), is_multiprocessing=True, scenario_timeout_sec=timeout,
    )
    raw = tmp_path / config["network"]["name"] / "raw"
    raw.mkdir(parents=True)
    marker = raw / "keep.txt"
    marker.write_text("existing data")
    with pytest.raises(ValueError):
        _setup_environment(config, allow_checkpointing=True)
    assert marker.read_text() == "existing data"


def test_timeout_requires_pool_backend() -> None:
    """Reject executor timeouts instead of silently ignoring the limit."""
    config = yaml.safe_load(Path("scripts/config/default.yaml").read_text())
    config["settings"].update(is_multiprocessing=False, scenario_timeout_sec=600.0)
    with pytest.raises(ValueError, match="requires is_multiprocessing"):
        validate_static_config(config)


@pytest.mark.timeout(45)
def test_worker_initialization_is_outside_scenario_deadline(tmp_path: Path) -> None:
    """Do not misclassify slow initialization as a hung load scenario."""
    options = {"startup_delay": 0.4, "error_log": str(tmp_path / "error.log")}
    reports = []
    with (tmp_path / "progress.log").open("w") as log, tqdm(total=1, file=log) as bar:
        with ScenarioTimeoutPool(
            multiprocessing.get_context("spawn"),
            1,
            _initialize_fake_worker,
            (options,),
            process_network._process_scenario_worker,
            0.15,
            set(),
            reports.append,
        ) as pool:
            results = pool.collect([(0, np.zeros((1, 1, 2)))], bar)
    assert results[0][4][0][0] == 0
    assert reports == []


def test_skipped_scenarios_are_durable_before_chunk_commit(tmp_path: Path) -> None:
    """Persist skips immediately and retain them when the chunk later commits."""
    config = yaml.safe_load(Path("scripts/config/default.yaml").read_text())
    config["settings"].update(
        data_dir=str(tmp_path),
        is_multiprocessing=True,
        scenario_timeout_sec=1.0,
        enable_checkpointing=True,
    )
    args, root, _, seed = _setup_environment(config, allow_checkpointing=True)
    checkpoint = GenerationCheckpoint.create(Path(root), args.to_dict(), seed)
    checkpoint.record_skip(2)
    checkpoint.record_skip(2)
    checkpoint.record_skip(4)
    assert checkpoint.state.next_chunk == 0
    assert checkpoint.state.saved_scenarios == 0
    config["settings"]["overwrite"] = False
    loaded = GenerationCheckpoint.load(Path(root), validate_static_config(config))
    assert loaded.state.skipped_scenarios == [2, 4]
    loaded.commit(1)
    assert GenerationCheckpoint.load(
        Path(root), validate_static_config(config),
    ).state.skipped_scenarios == [2, 4]
