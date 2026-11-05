"""CoinHSL validation and forwarding without requiring Julia or an HSL license."""

import importlib
import json
import sys
from concurrent.futures import Future
from contextlib import nullcontext
from multiprocessing.context import BaseContext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from gridfm_datakit.config import validate_static_config
from gridfm_datakit.generate import _setup_environment
from gridfm_datakit.process import process_network


@pytest.fixture
def config(tmp_path: Path) -> dict[str, Any]:
    """Return a small static configuration writing to a temporary directory."""
    config = yaml.safe_load(Path("scripts/config/default.yaml").read_text(encoding="utf-8"))
    config["settings"].update(data_dir=str(tmp_path / "data"), seed=42)
    config["load"]["scenarios"] = 2
    return config


@pytest.fixture
def coinhsl(tmp_path: Path) -> dict[str, Any]:
    """Return enabled options pointing to a placeholder library file."""
    # This file stands in for an installed library; these tests do not load it.
    library = tmp_path / "coinhsl $library.so"
    library.touch()
    return {"enabled": True, "linear_solver": "ma57", "hsllib": str(library)}


def test_coinhsl_defaults_are_disabled(config: dict[str, Any]) -> None:
    """Omitted options preserve Ipopt's default linear solver."""
    del config["settings"]["coinhsl"]
    assert validate_static_config(config)["settings"]["coinhsl"] == {
        "enabled": False,
        "linear_solver": "ma57",
        "hsllib": "",
    }


@pytest.mark.parametrize("solver", ["ma27", "ma57", "ma77", "ma86", "ma97"])
def test_coinhsl_accepts_supported_solvers(
    config: dict[str, Any], coinhsl: dict[str, Any], solver: str,
) -> None:
    """Accept each supported CoinHSL solver name."""
    config["settings"]["coinhsl"] = {**coinhsl, "linear_solver": solver}
    assert validate_static_config(config)["settings"]["coinhsl"]["linear_solver"] == solver


@pytest.mark.parametrize(
    "options,match",
    [
        ({"enabled": True}, "hsllib is required"),
        ({"enabled": True, "hsllib": "missing.so"}, "library not found"),
        ({"enabled": True, "hsllib": "."}, "library not found"),
        ({"linear_solver": "unknown"}, "linear_solver"),
        ({"enabled": "true"}, "valid boolean"),
        ({"extra_option": True}, "Extra inputs"),
    ],
)
def test_invalid_coinhsl_preserves_output(
    config: dict[str, Any], options: dict[str, Any], match: str,
) -> None:
    """Reject invalid options before deleting existing output."""
    config["settings"]["coinhsl"] = options
    output = Path(config["settings"]["data_dir"]) / config["network"]["name"] / "raw"
    output.mkdir(parents=True)
    marker = output / "keep.txt"
    marker.write_text("existing output")
    with pytest.raises(ValueError, match=match):
        _setup_environment(config)
    assert marker.read_text() == "existing output"


def test_disabled_coinhsl_does_not_require_library(config: dict[str, Any]) -> None:
    """Ignore an unavailable library path while CoinHSL is disabled."""
    config["settings"]["coinhsl"] = {"enabled": False, "hsllib": "missing.so"}
    assert validate_static_config(config)["settings"]["coinhsl"]["enabled"] is False


@pytest.mark.parametrize("options", [False, [], "", 0])
def test_init_julia_rejects_malformed_coinhsl(options: Any) -> None:
    """Reject malformed direct API inputs before importing or starting Julia."""
    with pytest.raises(ValueError, match="valid dictionary"):
        process_network.init_julia(50, coinhsl=options)


def test_coinhsl_resolves_relative_library(
    config: dict[str, Any], coinhsl: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve a relative library path before passing it to workers."""
    library = Path(coinhsl["hsllib"])
    monkeypatch.chdir(library.parent)
    config["settings"]["coinhsl"] = {**coinhsl, "hsllib": library.name}
    assert validate_static_config(config)["settings"]["coinhsl"]["hsllib"] == str(library)


@pytest.mark.parametrize("enabled", [False, True])
def test_init_julia_configures_all_ipopt_solvers(
    monkeypatch: pytest.MonkeyPatch, coinhsl: dict[str, Any], enabled: bool,
) -> None:
    """Apply enabled options to all Ipopt definitions and escape Julia paths."""
    julia = Mock()
    monkeypatch.setitem(sys.modules, "juliacall", SimpleNamespace(Main=julia))
    monkeypatch.setitem(
        sys.modules,
        "juliapkg.state",
        SimpleNamespace(STATE={}),
    )
    # Avoid modifying process-wide output routing or redirecting test output.
    router = Mock()
    router.channel.return_value.capture.side_effect = nullcontext
    monkeypatch.setattr(process_network, "build_router", lambda _: router)
    monkeypatch.setattr(process_network, "set_active_router", lambda _: None)

    options = {**coinhsl, "enabled": enabled}
    assert process_network.init_julia(50, coinhsl=options) is julia
    source = "\n".join(call.args[0] for call in julia.seval.call_args_list)
    assert source.count('"linear_solver" => "ma57"') == (5 if enabled else 0)
    literal = json.dumps(coinhsl["hsllib"], ensure_ascii=False).replace("$", "\\$")
    assert source.count(f'"hsllib" => {literal}') == (5 if enabled else 0)


@pytest.mark.parametrize("workers", [0, 1, 2], ids=["sequential", "one-worker", "pool"])
def test_generation_forwards_coinhsl(
    config: dict[str, Any], coinhsl: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch, workers: int,
) -> None:
    """Forward options through sequential, local-worker, and pool generation."""
    generate = importlib.import_module("gridfm_datakit.generate")
    config["settings"].update(coinhsl=coinhsl, mode="opf", num_processes=max(workers, 1))
    net = SimpleNamespace()
    scenarios = np.zeros((1, 2, 2))
    monkeypatch.setattr(
        generate,
        "_prepare_network_and_scenarios",
        lambda *_: (net, scenarios, {}),
    )
    for name in (
        "initialize_topology_generator",
        "initialize_generation_generator",
        "initialize_admittance_generator",
    ):
        monkeypatch.setattr(generate, name, lambda *_: None)
    monkeypatch.setattr(generate, "_save_generated_data", lambda *_: None)
    monkeypatch.setattr(generate, "write_ram_usage_distributed", lambda *_: None)
    init = Mock(return_value=object())
    process = Mock(return_value=[])
    monkeypatch.setattr(generate, "init_julia", init)
    monkeypatch.setattr(process_network, "init_julia", init)
    monkeypatch.setattr(generate, "process_scenario_opf_mode", process)
    monkeypatch.setattr(process_network, "process_scenario_opf_mode", process)
    monkeypatch.setattr(process_network, "_worker_context", None)
    monkeypatch.setattr(process_network, "_worker_jl", None)

    class Executor:
        """Execute pool tasks locally so initialization is observable."""

        def __init__(
            self, *, max_workers: int, mp_context: BaseContext,
            initializer: Callable[..., None], initargs: tuple[Any, ...],
        ) -> None:
            """Initialize worker state using the same arguments as a spawned pool."""
            assert max_workers == 2
            assert mp_context.get_start_method() == "spawn"
            initializer(*initargs)

        def __enter__(self) -> "Executor":
            """Return the executor for context-manager use."""
            return self

        def __exit__(self, *_: Any) -> None:
            """Finish without suppressing exceptions."""
            pass

        def submit(self, fn: Callable[..., Any], task: Any) -> Future[Any]:
            """Execute a task and return its completed future."""
            future = Future()
            future.set_result(fn(task))
            return future

    monkeypatch.setattr(generate, "ProcessPoolExecutor", Executor)
    if workers:
        generate.generate_power_flow_data_distributed(config)
    else:
        generate.generate_power_flow_data(config)
    init.assert_called_once()
    assert init.call_args.kwargs["coinhsl"] == coinhsl
    assert process.call_count == 2


def test_load_scaling_forwards_coinhsl(
    coinhsl: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Forward options from load generation to the scaling worker's Julia instance."""
    load = importlib.import_module("gridfm_datakit.perturbations.load_perturbation")
    julia = Mock()
    julia.run_opf.return_value = {"termination_status": "LOCALLY_SOLVED"}
    init = Mock(return_value=julia)
    monkeypatch.setattr(load, "init_julia", init)

    class Pool:
        """Run the load-scaling worker synchronously for inspection."""

        def __init__(self, processes: int) -> None:
            """Check that load-scaling uses a single worker."""
            assert processes == 1

        def apply_async(self, fn: Callable[..., Any], args: tuple[Any, ...]) -> SimpleNamespace:
            """Return a result object wrapping the worker's computed value."""
            value = fn(*args)
            return SimpleNamespace(get=lambda timeout: value)

        def close(self) -> None:
            """Accept the normal pool shutdown call."""
            pass

        def join(self) -> None:
            """Accept the normal pool join call."""
            pass

    monkeypatch.setattr(load, "Pool", Pool)
    net = SimpleNamespace(Pd=np.array([10.0]), Qd=np.array([2.0]), to_mpc=Mock())
    generator = load.LoadScenariosFromAggProfile("default", 0.0, True, 0.0, 1.0, 0.1, 1.0)
    scenarios = generator(net, 2, str(tmp_path / "load.log"), 50, 42, coinhsl=coinhsl)
    assert scenarios.shape == (1, 2, 2)
    assert init.call_args.kwargs["coinhsl"] == coinhsl
