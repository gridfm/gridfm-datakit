"""Regression tests for Pool selection, progress, ordering, and cleanup."""

from concurrent.futures import Future
import importlib
from pathlib import Path
from queue import Queue
from typing import Any
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import yaml

from gridfm_datakit.config import validate_static_config
from gridfm_datakit.generate import _collect_pool_results, _setup_environment
from gridfm_datakit.process import process_network


@pytest.fixture
def config(tmp_path: Path) -> dict[str, Any]:
    """Return a seeded configuration with multiple uneven large chunks."""
    config = yaml.safe_load(
        Path("scripts/config/default.yaml").read_text(encoding="utf-8"),
    )
    config["settings"].update(
        data_dir=str(tmp_path / "data"),
        seed=42,
        num_processes=2,
        large_chunk_size=3,
    )
    config["load"]["scenarios"] = 5
    return config


def test_multiprocessing_defaults_to_existing_backend(config: dict[str, Any]) -> None:
    """Omitting the new setting keeps ProcessPoolExecutor behavior."""
    del config["settings"]["process_backend"]
    assert (
        validate_static_config(config)["settings"]["process_backend"]
        == "process_pool_executor"
    )


def test_invalid_backend_setting_preserves_output(config: dict[str, Any]) -> None:
    """Reject an invalid backend name before removing existing output."""
    config["settings"]["process_backend"] = "true"
    raw = Path(config["settings"]["data_dir"]) / config["network"]["name"] / "raw"
    raw.mkdir(parents=True)
    marker = raw / "keep.txt"
    marker.write_text("existing output", encoding="utf-8")
    with pytest.raises(
        ValueError,
        match=r"settings\.process_backend: Input should be 'process_pool_executor' or 'multiprocessing_pool'",
    ):
        _setup_environment(config)
    assert marker.read_text(encoding="utf-8") == "existing output"


def test_pool_results_drain_progress_and_preserve_offsets() -> None:
    """Out-of-order task completion retains offsets and drains every tick."""
    queue = Queue()
    for _ in range(3):
        queue.put(1)
    first = Mock()
    first.ready.side_effect = [False, True]
    first.get.return_value = (7, 2, None, None, ["first"])
    second = Mock()
    second.ready.return_value = True
    second.get.return_value = (9, 1, None, None, ["second"])
    pool = Mock()
    pool.apply_async.side_effect = [first, second]
    bar = Mock()

    results = _collect_pool_results(
        pool,
        [(7, np.zeros((1, 2, 2))), (9, np.zeros((1, 1, 2)))],
        queue,
        bar,
    )

    assert list(results) == [9, 7]
    assert results[7][4] == ["first"]
    assert queue.empty()
    assert sum(call.args[0] for call in bar.update.call_args_list) == 3


@pytest.mark.parametrize("reported_error", [False, True], ids=["raised", "returned"])
def test_pool_worker_errors_are_propagated(reported_error: bool) -> None:
    """Propagate both transport errors and failures returned by workers."""
    error = ValueError("worker failed")
    result = Mock()
    result.ready.return_value = True
    if reported_error:
        result.get.return_value = (0, 1, error, "worker traceback", None)
    else:
        result.get.side_effect = error
    pool = Mock()
    pool.apply_async.return_value = result
    with pytest.raises(RuntimeError, match="failed") as caught:
        _collect_pool_results(pool, [(0, np.zeros((1, 1, 2)))], Queue(), Mock())
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("mode", ["pf", "opf"])
def test_pool_matches_executor_and_cleans_up(
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    workers: int,
    mode: str,
) -> None:
    """Preserve seeds, ordering, and solver options across both backends."""
    generate = importlib.import_module("gridfm_datakit.generate")
    config["settings"].update(
        num_processes=workers,
        mode=mode,
        opf_formulation="rectangular",
    )
    metadata = {"pp_net": object(), "network_path": "network.m", "mapping_p2g": {}}
    monkeypatch.setattr(
        generate,
        "_prepare_network_and_scenarios",
        lambda *_: (object(), np.zeros((1, 5, 2)), metadata),
    )
    for name in (
        "initialize_topology_generator",
        "initialize_generation_generator",
        "initialize_admittance_generator",
    ):
        monkeypatch.setattr(generate, name, lambda *_: None)
    monkeypatch.setattr(generate, "write_ram_usage_distributed", lambda *_: None)
    monkeypatch.setattr(process_network, "_worker_context", None)
    monkeypatch.setattr(process_network, "_worker_jl", None)
    initialize = Mock(return_value=object())
    monkeypatch.setattr(process_network, "init_julia", initialize)

    def process(*args: Any, **kwargs: Any) -> list:
        """Record global indices and draws from the actual chunk seed policy."""
        data = args[6]
        data.append((args[2], float(np.random.random())))
        return data

    monkeypatch.setattr(process_network, "process_scenario_pf_mode", process)
    monkeypatch.setattr(process_network, "process_scenario_opf_mode", process)
    context = MagicMock()
    manager = context.Manager.return_value.__enter__.return_value
    progress = Queue()
    manager.Queue.return_value = progress
    pool_context = MagicMock()
    pool = pool_context.__enter__.return_value

    def make_pool(*, processes: int, initializer: Any, initargs: tuple) -> Any:
        """Install pool worker state without spawning during unit tests."""
        assert processes == workers
        assert "pp_net" not in initargs[13]
        initializer(*initargs)
        return pool_context

    def apply_async(fn: Any, args: tuple) -> Mock:
        """Run a task and return the Pool asynchronous-result interface."""
        result = Mock()
        result.ready.return_value = True
        result.get.return_value = fn(*args)
        return result

    context.Pool.side_effect = make_pool
    pool.apply_async.side_effect = apply_async
    monkeypatch.setattr(generate.multiprocessing, "get_context", lambda method: context)

    class Executor:
        """Execute futures locally using the same worker state as a real pool."""

        def __init__(self, *, initializer: Any, initargs: tuple, **kwargs: Any) -> None:
            """Initialize worker state once per generation run."""
            initializer(*initargs)

        def __enter__(self) -> "Executor":
            """Return the executor."""
            return self

        def __exit__(self, *args: Any) -> None:
            """Complete context-manager cleanup."""
            pass

        def submit(self, fn: Any, task: Any) -> Future:
            """Return the completed task as a future."""
            future = Future()
            future.set_result(fn(task))
            return future

    monkeypatch.setattr(generate, "ProcessPoolExecutor", Executor)
    observed = []
    monkeypatch.setattr(
        generate,
        "_save_generated_data",
        lambda net, data, *args: observed.extend(data),
    )
    config["settings"]["process_backend"] = "process_pool_executor"
    generate.generate_power_flow_data_distributed(config)
    expected = observed[:]
    observed.clear()
    initialize.reset_mock()
    config["settings"]["process_backend"] = "multiprocessing_pool"
    generate.generate_power_flow_data_distributed(config)

    assert observed == expected
    assert [item[0] for item in observed] == list(range(5))
    assert initialize.call_count == 1
    assert initialize.call_args.kwargs["opf_formulation"] == "rectangular"
    assert initialize.call_args.kwargs["coinhsl"] == config["settings"]["coinhsl"]
    assert pool_context.__exit__.call_count == 1
    assert context.Manager.return_value.__exit__.call_count == 1
    assert progress.empty()

    pool.apply_async.side_effect = RuntimeError("cannot submit task")
    with pytest.raises(RuntimeError, match="cannot submit task"):
        generate.generate_power_flow_data_distributed(config)
    assert pool_context.__exit__.call_count == 2
    assert context.Manager.return_value.__exit__.call_count == 2

    pool.apply_async.side_effect = apply_async
    initialize.side_effect = RuntimeError("solver initialization failed")
    with pytest.raises(RuntimeError, match="solver initialization failed"):
        generate.generate_power_flow_data_distributed(config)
    assert pool_context.__exit__.call_count == 3
    assert context.Manager.return_value.__exit__.call_count == 3
