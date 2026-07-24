"""Tests for process_dynamic.py"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from markers import needs_dynawo

from gridfm_datakit.dynamic.dynawo import (
    generate_dynawo_mappings,
    get_dynawo_simulation_parameters,
)
from gridfm_datakit.dynamic import load_raw_inputs
from gridfm_datakit.dynamic import process_dynamic as pdyn
from gridfm_datakit.generate import _setup_environment, _prepare_network_and_scenarios
from gridfm_datakit.powsybl import load_net
from gridfm_datakit.utils.param_handler import NestedNamespace

# No module-level pytestmark: only the tests that simulate carry @needs_dynawo.


@needs_dynawo
def test_process_single_dynamic_simulation(config_ieee14):
    from gridfm_datakit.dynamic.process_dynamic import process_single_dynamic_simulation
    from gridfm_datakit.generate import init_julia

    config = config_ieee14

    args, _, _, _ = _setup_environment(config)
    dynamic_inputs = load_raw_inputs(args)
    dynawo_mappings = generate_dynawo_mappings(dynamic_inputs)
    simulation_parameters = get_dynawo_simulation_parameters(args)
    net = load_net(args.network.file)

    gfm_net = net.gfm_net
    scenarios = np.zeros((len(gfm_net.Qd), 1, 2))
    scenarios[:, 0, 0] = gfm_net.Pd
    scenarios[:, 0, 1] = gfm_net.Qd

    julia = init_julia(200)

    results = process_single_dynamic_simulation(
        pp_net=net.pp_net,
        gfm_net=net.gfm_net,
        scenarios=scenarios,
        scenario_index=0,
        p2g_maps=net.mapping_p2g,
        dynamic_mappings=dynawo_mappings,
        dynamic_solver_params=simulation_parameters,
        dynamic_solver="dynawo",
        julia=julia,
    )
    # With no perturbation generators, a scenario yields exactly one sample
    assert len(results) == 1
    sample = results[0]
    assert set(sample) == {
        "pf_data",
        "dynamic_results",
        "scenario_index",
        "perturbation_index",
    }
    assert sample["scenario_index"] == 0 and sample["perturbation_index"] == 0


@needs_dynawo
def test_process_dynamic_simulation(config_ieee14):
    from gridfm_datakit.dynamic.process_dynamic import process_dynamic_simulations

    config = config_ieee14

    args, _, file_paths, seed = _setup_environment(config)
    gfm_net, scenarios, meta = _prepare_network_and_scenarios(args, file_paths, seed)
    dynamic_inputs = load_raw_inputs(args)

    res_dict = process_dynamic_simulations(
        network_path=str(meta.get("network_path")),
        scenarios=scenarios,
        dynamic_inputs=dynamic_inputs,
        dynamic_solver="dynawo",
        config=args,
        error_log_file=".",
        seed=seed,
    )
    assert len(res_dict) == args.load.scenarios


# The tests below cover the orchestration — chunking, error isolation, variant
# lifecycle — which is solver-independent, so the solvers are stubbed out.


class _FakeNet:
    """Minimal stand-in for the object load_net returns."""

    def __init__(self):
        self.pp_net = _FakePpNet()
        self.gfm_net = _FakeGfmNet()
        self.mapping_p2g = {}


class _FakeGfmNet:
    """A gfm network carrying only the load arrays a scenario overwrites."""

    def __init__(self, n_loads: int = 3):
        self.Pd = np.zeros(n_loads)
        self.Qd = np.zeros(n_loads)


class _FakePpNet:
    """Records the variant calls, so the per-perturbation lifecycle is observable."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.live_variants: set[str] = set()
        self.working = "base"
        self.clone_fails_on: set[str] = set()

    def get_working_variant_id(self):
        return self.working

    def clone_variant(self, source, target):
        self.calls.append(("clone", source, target))
        if target in self.clone_fails_on:
            raise RuntimeError(f"cannot clone {target}")
        self.live_variants.add(target)

    def set_working_variant(self, variant_id):
        self.calls.append(("set", variant_id))
        self.working = variant_id

    def remove_variant(self, variant_id):
        self.calls.append(("remove", variant_id))
        self.live_variants.discard(variant_id)


class _ListTopologyGenerator:
    """Expands one network into a fixed number of perturbed ones."""

    def __init__(self, n_variants: int):
        self.n_variants = n_variants

    def generate(self, net):
        return [copy.deepcopy(net) for _ in range(self.n_variants)]


def _stub_solver_steps(monkeypatch, fail_on=()):
    """Stub the OPF/PF and Dynawo steps; fail_on lists the ones that diverge."""
    seen = []

    def _static_state(**kwargs):
        return None, {"bus": np.zeros((2, 3))}

    def _dynamic(network, mappings, params, solver):
        index = len(seen)
        seen.append(index)
        if index in fail_on:
            raise RuntimeError(f"dynamic simulation {index} diverged")
        return f"curves-{index}"

    monkeypatch.setattr(pdyn, "_compute_balanced_static_state", _static_state)
    monkeypatch.setattr(pdyn, "_run_dynamic_simulation", _dynamic)
    return seen


class TestProcessSingleDynamicSimulation:
    @staticmethod
    def _run(pp_net, topology_generator, error_log_file=None, scenario_index=0):
        scenarios = np.zeros((3, scenario_index + 1, 2))
        return pdyn.process_single_dynamic_simulation(
            pp_net=pp_net,
            gfm_net=_FakeGfmNet(),
            scenarios=scenarios,
            scenario_index=scenario_index,
            p2g_maps={},
            dynamic_mappings=None,
            dynamic_solver_params=None,
            dynamic_solver="dynawo",
            julia=None,
            topology_generator=topology_generator,
            error_log_file=error_log_file,
        )

    def test_one_sample_per_topology_perturbation(self, monkeypatch):
        _stub_solver_steps(monkeypatch)
        pp_net = _FakePpNet()

        results = self._run(pp_net, _ListTopologyGenerator(3))

        assert [r["perturbation_index"] for r in results] == [0, 1, 2]
        assert {r["scenario_index"] for r in results} == {0}

    def test_a_failed_perturbation_does_not_drop_the_scenario(
        self,
        monkeypatch,
        tmp_path,
    ):
        _stub_solver_steps(monkeypatch, fail_on={1})
        pp_net = _FakePpNet()
        error_log = tmp_path / "error.log"

        results = self._run(pp_net, _ListTopologyGenerator(3), str(error_log))

        # the failed sample leaves a hole rather than shifting the others
        assert [r["perturbation_index"] for r in results] == [0, 2]
        assert "scenario 0 perturbation 1 failed" in error_log.read_text()

    def test_every_variant_is_removed_even_when_the_simulation_fails(
        self,
        monkeypatch,
        tmp_path,
    ):
        """A leaked variant would outlive the sample and pollute the chunk."""
        _stub_solver_steps(monkeypatch, fail_on={0, 1})
        pp_net = _FakePpNet()

        self._run(pp_net, _ListTopologyGenerator(2), str(tmp_path / "error.log"))

        assert pp_net.live_variants == set()
        assert pp_net.get_working_variant_id() == "base"

    def test_a_variant_that_cannot_be_created_is_not_removed(
        self,
        monkeypatch,
        tmp_path,
    ):
        """remove_variant on a variant that was never cloned would itself raise."""
        _stub_solver_steps(monkeypatch)
        pp_net = _FakePpNet()
        pp_net.clone_fails_on = {"scenario_0_perturbation_0"}
        error_log = tmp_path / "error.log"

        results = self._run(pp_net, _ListTopologyGenerator(2), str(error_log))

        assert [r["perturbation_index"] for r in results] == [1]
        assert ("remove", "scenario_0_perturbation_0") not in pp_net.calls
        assert "scenario 0 perturbation 0 failed" in error_log.read_text()

    def test_no_topology_generator_yields_exactly_one_sample(self, monkeypatch):
        _stub_solver_steps(monkeypatch)
        results = self._run(_FakePpNet(), None)
        assert len(results) == 1 and results[0]["perturbation_index"] == 0


def _chunk_args(
    start_idx=0,
    end_idx=2,
    dynamic_solver="dynawo",
    error_log_file=None,
    config=None,
):
    """Build the positional tuple _process_dynamic_chunk unpacks."""
    return (
        start_idx,
        end_idx,
        np.zeros((3, max(end_idx, 1), 2)),  # scenarios
        "network.iidm",  # network_path
        None,  # dynamic_inputs
        dynamic_solver,
        error_log_file,
        200,  # max_iter
        None,  # solver_log_dir
        0,  # seed
        config if config is not None else NestedNamespace(dummy=1),
        None,  # topology_generator
        None,  # generation_generator
        None,  # admittance_generator
    )


def _stub_worker_setup(monkeypatch, **overrides):
    """Stub the per-worker initialisation (Julia, network, Dynawo mappings)."""
    defaults = {
        "init_julia": lambda *_args, **_kwargs: "julia",
        "load_net": lambda *_args, **_kwargs: _FakeNet(),
        "generate_dynawo_mappings": lambda *_args, **_kwargs: "mappings",
        "get_dynawo_simulation_parameters": lambda *_args, **_kwargs: "params",
        "get_dynawo_loadflow_parameters": lambda *_args, **_kwargs: "lf_params",
    }
    defaults.update(overrides)
    for name, value in defaults.items():
        monkeypatch.setattr(pdyn, name, value)


class TestProcessDynamicChunk:
    def test_returns_one_entry_per_scenario_in_the_chunk(self, monkeypatch):
        _stub_worker_setup(monkeypatch)
        monkeypatch.setattr(
            pdyn,
            "process_single_dynamic_simulation",
            lambda **kwargs: [{"scenario_index": kwargs["scenario_index"]}],
        )

        results = pdyn._process_dynamic_chunk(_chunk_args(start_idx=2, end_idx=5))

        assert [r["scenario_index"] for r in results] == [2, 3, 4]

    def test_perturbations_are_flattened_into_the_chunk_result(self, monkeypatch):
        """One scenario expands to several samples: extend, not append."""
        _stub_worker_setup(monkeypatch)
        monkeypatch.setattr(
            pdyn,
            "process_single_dynamic_simulation",
            lambda **kwargs: [
                {"scenario_index": kwargs["scenario_index"], "perturbation_index": i}
                for i in range(2)
            ],
        )

        results = pdyn._process_dynamic_chunk(_chunk_args(start_idx=0, end_idx=2))

        assert [(r["scenario_index"], r["perturbation_index"]) for r in results] == [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]

    def test_a_failing_scenario_is_logged_and_the_chunk_continues(
        self,
        monkeypatch,
        tmp_path,
    ):
        _stub_worker_setup(monkeypatch)
        error_log = tmp_path / "error.log"

        def _sometimes(**kwargs):
            if kwargs["scenario_index"] == 0:
                raise RuntimeError("scenario blew up")
            return [{"scenario_index": kwargs["scenario_index"]}]

        monkeypatch.setattr(pdyn, "process_single_dynamic_simulation", _sometimes)

        results = pdyn._process_dynamic_chunk(
            _chunk_args(start_idx=0, end_idx=3, error_log_file=str(error_log)),
        )

        assert [r["scenario_index"] for r in results] == [1, 2]
        log = error_log.read_text()
        assert "scenario 0 failed: scenario blew up" in log
        assert "Traceback" in log  # the traceback is what makes it diagnosable

    def test_a_worker_that_cannot_start_returns_the_exception(self, monkeypatch):
        """The exception travels back to the parent, which logs it."""

        def _boom(*_args, **_kwargs):
            raise RuntimeError("no Julia here")

        _stub_worker_setup(monkeypatch, init_julia=_boom)

        results = pdyn._process_dynamic_chunk(_chunk_args())

        assert len(results) == 1 and isinstance(results[0], RuntimeError)
        assert "no Julia here" in str(results[0])

    def test_unsupported_solver_raises(self):
        with pytest.raises(NotImplementedError, match="is not implemented"):
            pdyn._process_dynamic_chunk(_chunk_args(dynamic_solver="not_a_solver"))

    def test_chunk_seeds_are_derived_from_the_scenario_offset(self, monkeypatch):
        """Two chunks of one run must not draw the same perturbations."""
        _stub_worker_setup(monkeypatch)
        monkeypatch.setattr(
            pdyn,
            "process_single_dynamic_simulation",
            lambda **kwargs: [{"draw": float(np.random.rand())}],
        )

        first = pdyn._process_dynamic_chunk(_chunk_args(start_idx=0, end_idx=1))
        second = pdyn._process_dynamic_chunk(_chunk_args(start_idx=1, end_idx=2))
        first_again = pdyn._process_dynamic_chunk(_chunk_args(start_idx=0, end_idx=1))

        assert first[0]["draw"] != second[0]["draw"]
        assert first[0]["draw"] == first_again[0]["draw"]  # and it is reproducible


class _InlinePool:
    """A Pool that maps in this process: a spawned one re-imports the module and
    would never see a patched _process_dynamic_chunk."""

    def __init__(self, processes=None):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def map(self, func, iterable):
        return [func(item) for item in iterable]


class _InlineContext:
    def Pool(self, processes=None):
        return _InlinePool(processes)


def _config(scenarios: int, large_chunk_size: int, num_processes: int = 1):
    return NestedNamespace(
        load=NestedNamespace(scenarios=scenarios),
        settings=NestedNamespace(
            large_chunk_size=large_chunk_size,
            num_processes=num_processes,
            max_iter=200,
        ),
    )


@pytest.fixture
def inline_pool(monkeypatch):
    """Run the chunk worker inline, on a stub network."""

    class _FakeMultiprocessing:
        @staticmethod
        def get_context(_method):
            return _InlineContext()

    monkeypatch.setattr(pdyn, "multiprocessing", _FakeMultiprocessing)
    monkeypatch.setattr(pdyn, "load_net", lambda *_a, **_kw: _FakeNet())


def _iter_chunks(config, worker, monkeypatch, seed=0):
    monkeypatch.setattr(pdyn, "_process_dynamic_chunk", worker)
    return list(
        pdyn.iter_dynamic_simulations(
            network_path="network.iidm",
            scenarios=np.zeros((3, config.load.scenarios, 2)),
            dynamic_inputs=None,
            dynamic_solver="dynawo",
            config=config,
            error_log_file=None,
            seed=seed,
        ),
    )


@pytest.mark.usefixtures("inline_pool")
class TestIterDynamicSimulations:
    def test_yields_one_list_per_large_chunk(self, monkeypatch):
        """Chunks are yielded as they complete, so the caller can write and drop
        them."""

        def _worker(args):
            start_idx, end_idx = args[0], args[1]
            return [{"scenario_index": i} for i in range(start_idx, end_idx)]

        chunks = _iter_chunks(
            _config(scenarios=5, large_chunk_size=2),
            _worker,
            monkeypatch,
        )

        assert [[r["scenario_index"] for r in chunk] for chunk in chunks] == [
            [0, 1],
            [2, 3],
            [4],
        ]

    def test_every_scenario_is_dispatched_exactly_once(self, monkeypatch):
        """Splitting a chunk across workers must not drop or duplicate a scenario."""
        dispatched = []

        def _worker(args):
            dispatched.extend(range(args[0], args[1]))
            return []

        _iter_chunks(
            _config(scenarios=7, large_chunk_size=3, num_processes=2),
            _worker,
            monkeypatch,
        )

        assert sorted(dispatched) == list(range(7))

    def test_a_worker_level_exception_is_dropped_not_yielded(self, monkeypatch):
        """Yielding it would crash the writer, which expects result dicts."""
        chunks = _iter_chunks(
            _config(scenarios=2, large_chunk_size=2),
            lambda _args: [RuntimeError("worker died")],
            monkeypatch,
        )
        assert chunks == [[]]

    def test_a_per_scenario_exception_does_not_hide_its_siblings(self, monkeypatch):
        chunks = _iter_chunks(
            _config(scenarios=2, large_chunk_size=2),
            lambda _args: [{"scenario_index": 0}, ValueError("bad scenario")],
            monkeypatch,
        )
        assert chunks == [[{"scenario_index": 0}]]

    def test_process_dynamic_simulations_flattens_every_chunk(self, monkeypatch):
        monkeypatch.setattr(
            pdyn,
            "_process_dynamic_chunk",
            lambda args: [{"scenario_index": i} for i in range(args[0], args[1])],
        )
        results = pdyn.process_dynamic_simulations(
            network_path="network.iidm",
            scenarios=np.zeros((3, 5, 2)),
            dynamic_inputs=None,
            dynamic_solver="dynawo",
            config=_config(scenarios=5, large_chunk_size=2),
            error_log_file=None,
            seed=0,
        )
        assert [r["scenario_index"] for r in results] == [0, 1, 2, 3, 4]


def test_log_error_falls_back_to_stdout_when_the_file_cannot_be_written(
    tmp_path,
    capsys,
):
    """The error path must not itself raise."""
    unwritable = tmp_path / "missing_dir" / "error.log"
    pdyn._log_error(str(unwritable), "boom\n")
    assert "boom" in capsys.readouterr().out


def test_log_error_appends_rather_than_truncates(tmp_path):
    error_log = Path(tmp_path / "error.log")
    pdyn._log_error(str(error_log), "first\n")
    pdyn._log_error(str(error_log), "second\n")
    assert error_log.read_text() == "first\nsecond\n"
