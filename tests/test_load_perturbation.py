"""
Tests for load perturbation no-op conditions for aggregated profile generator.
"""

import numpy as np
import tempfile
import os
from gridfm_datakit.network import load_net_from_pglib
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
        scenarios = gen(net, n_scenarios=5, scenarios_log=scenario_log, max_iter=200)
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
        scenarios = gen(net, n_scenarios=4, scenarios_log=scenario_log, max_iter=200)
        # p columns identical across scenarios
        p_first = scenarios[:, 0, 0]
        assert np.allclose(scenarios[:, :, 0], p_first[:, None])
        # q must equal base across all scenarios
        for s in range(scenarios.shape[1]):
            assert np.allclose(scenarios[:, s, 1], q_base)
