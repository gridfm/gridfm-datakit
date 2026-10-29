"""
Tests for load perturbation no-op conditions for aggregated profile generator.
"""

import numpy as np
import pytest
import tempfile
import os
from gridfm_datakit.network import load_net_from_pglib
from gridfm_datakit.perturbations import load_perturbation
from gridfm_datakit.perturbations.load_perturbation import LoadScenariosFromAggProfile


def test_agg_load_no_variation_when_sigma_and_range_zero_change_q_true():
    """With sigma=0 and global_range=0, all scenarios should be identical (p and q)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scenario_log = os.path.join(tmpdir, "scenarios.log")
        net = load_net_from_pglib("case24_ieee_rts")
        gen = LoadScenariosFromAggProfile(
            agg_load_name="default",
            sigma=0.0,
            change_reactive_power=True,
            global_range=0.0,
            max_scaling_factor=1.0,
            step_size=0.1,
            start_scaling_factor=1.0,
        )
        scenarios = gen(
            net,
            n_scenarios=5,
            scenarios_log=scenario_log,
            max_iter=200,
            seed=0,
        )
        # shape: (n_loads, n_scenarios, 2)
        # All scenarios across axis=1 should be equal to the first scenario
        p_first = scenarios[:, 0, 0]
        q_first = scenarios[:, 0, 1]
        assert np.allclose(scenarios[:, :, 0], p_first[:, None])
        assert np.allclose(scenarios[:, :, 1], q_first[:, None])


def test_agg_load_no_variation_when_sigma_and_range_zero_change_q_false():
    """With sigma=0 and global_range=0 and change_reactive_power=False, p varies identically, q equals base."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scenario_log = os.path.join(tmpdir, "scenarios.log")
        net = load_net_from_pglib("case24_ieee_rts")
        q_base = net.Qd.copy()
        gen = LoadScenariosFromAggProfile(
            agg_load_name="default",
            sigma=0.0,
            change_reactive_power=False,
            global_range=0.0,
            max_scaling_factor=1.0,
            step_size=0.1,
            start_scaling_factor=1.0,
        )
        scenarios = gen(
            net,
            n_scenarios=4,
            scenarios_log=scenario_log,
            max_iter=200,
            seed=0,
        )
        # p columns identical across scenarios
        p_first = scenarios[:, 0, 0]
        assert np.allclose(scenarios[:, :, 0], p_first[:, None])
        # q must equal base across all scenarios
        for s in range(scenarios.shape[1]):
            assert np.allclose(scenarios[:, s, 1], q_base)


class _FakeJulia:
    """OPF stand-in that solves the first ``solved`` calls and fails after."""

    def __init__(self, solved):
        self.solved = solved
        self.calls = 0

    def run_opf(self, path):
        self.calls += 1
        status = "LOCALLY_SOLVED" if self.calls <= self.solved else "LOCALLY_INFEASIBLE"
        return {"termination_status": status}


def _search(monkeypatch, solved, start=0.8, step=0.05, max_scaling=4.0):
    fake = _FakeJulia(solved)
    monkeypatch.setattr(load_perturbation, "init_julia", lambda **kwargs: fake)
    net = load_net_from_pglib("case14_ieee")
    args = (net, max_scaling, step, start, True, 200)
    return load_perturbation._find_largest_scaling_factor_worker(args)


def test_scaling_search_returns_last_solved_level(monkeypatch):
    assert _search(monkeypatch, solved=3) == pytest.approx(0.9)


def test_scaling_search_stops_at_max_scaling(monkeypatch):
    assert _search(monkeypatch, 100, 1.0, 0.1, 1.0) == pytest.approx(1.0)


def test_scaling_search_raises_when_start_fails(monkeypatch):
    with pytest.raises(RuntimeError, match="starting u=0.800"):
        _search(monkeypatch, solved=0)
