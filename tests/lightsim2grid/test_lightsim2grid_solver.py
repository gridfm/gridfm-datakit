"""Tests for the lightsim2grid power flow solver (``settings.pf_solver: lightsim2grid``)."""

import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridfm_datakit import generate_power_flow_data
from gridfm_datakit import lightsim2grid as l2g
from gridfm_datakit.network import load_net_from_file
from gridfm_datakit.process.process_network import _solution_arrays

pytestmark = pytest.mark.skipif(
    not l2g.is_lightsim2grid_available(),
    reason="lightsim2grid is not installed. Install with: pip install gridfm-datakit[lightsim2grid]",
)

_GRID = Path(__file__).parents[1] / "powsybl" / "grids" / "ieee14.m"

_BASE_CONFIG = {
    "network": {
        "name": "ieee14",
        "source": "file",
        "network_dir": str(_GRID.parent),
    },
    "load": {
        "generator": "agg_load_profile",
        "agg_profile": "default",
        "scenarios": 5,
        "sigma": 0.2,
        "change_reactive_power": True,
        "global_range": 0.4,
        "max_scaling_factor": 4.0,
        "step_size": 0.05,
        "start_scaling_factor": 0.8,
    },
    "topology_perturbation": {"type": "none"},
    "generation_perturbation": {"type": "cost_permutation"},
    "admittance_perturbation": {"type": "random_perturbation", "sigma": 0.2},
    "settings": {
        "num_processes": 1,
        "large_chunk_size": 5,
        "overwrite": True,
        "mode": "pf",
        "include_dc_res": True,
        "enable_solver_logs": False,
        "pf_fast": True,
        "dcpf_fast": True,
        "max_iter": 200,
        "pf_solver": "lightsim2grid",
        "seed": 42,
    },
}


def test_mapping_and_ac_pf_are_consistent():
    """Every branch/gen/bus is mapped, and the AC PF closes the power balance."""
    net = load_net_from_file(str(_GRID))
    conv = l2g.convert_net(net)
    mapping = conv.mapping_l2g
    all_rows = np.sort(np.concatenate([mapping.line_rows, mapping.trafo_rows]))
    assert np.array_equal(all_rows, np.arange(net.branches.shape[0]))

    res = l2g.run_ls_pf(conv.ls_net, net, mapping)
    flows, gen_pq, bus_vmva = _solution_arrays(res, net)
    assert flows.shape == (len(net.idx_branches_in_service), 4)
    assert gen_pq.shape == (len(net.idx_gens_in_service), 2)
    # losses are positive: pf + pt >= 0 on every branch
    assert np.all(flows[:, 0] + flows[:, 2] > -1e-9)
    assert np.all(np.isfinite(bus_vmva))


def test_diverging_pf_raises():
    net = load_net_from_file(str(_GRID))
    net.buses[:, 2] *= 100  # absurd load: no solution
    conv = l2g.convert_net(net)
    with pytest.raises(ValueError, match="did not converge"):
        l2g.run_ls_pf(conv.ls_net, net, conv.mapping_l2g)


def test_generate_pf_mode(tmp_path):
    config = copy.deepcopy(_BASE_CONFIG)
    config["settings"]["data_dir"] = str(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generate_power_flow_data(config)
    raw = tmp_path / "ieee14" / "raw"
    assert (raw / "error.log").read_text().count("Caught an exception") == 0
    bus = pd.read_parquet(raw / "bus_data.parquet")
    assert bus["scenario"].nunique() == 5
    assert np.all(np.isfinite(bus[["Vm", "Va"]].to_numpy()))


def _perturbed(net, rng, step):
    """A perturbed copy of ``net``: loads, and depending on the step admittances,
    a branch outage, generator set points and a generator outage (never the slack)."""
    from gridfm_datakit.utils.idx_brch import BR_B, BR_R, BR_STATUS, BR_X
    from gridfm_datakit.utils.idx_bus import PD, QD
    from gridfm_datakit.utils.idx_gen import GEN_STATUS, PG, VG

    p = net.copy_for_perturbation()
    f = 1 + 0.1 * rng.uniform(-1, 1, p.buses.shape[0])
    p.buses[:, PD] *= f
    p.buses[:, QD] *= f
    if step % 2:
        for col in (BR_R, BR_X, BR_B):
            p.branches[:, col] *= 1 + 0.2 * rng.uniform(-1, 1, p.branches.shape[0])
    if step % 3 == 0:
        p.branches[rng.integers(p.branches.shape[0]), BR_STATUS] = 0
    if step % 5 == 0:
        p.gens[:, PG] *= 1 + 0.1 * rng.uniform(-1, 1, p.gens.shape[0])
        p.gens[1:, VG] += 0.005
        if step % 10 == 0:
            p.gens[3, GEN_STATUS] = 0
    return p


def test_in_place_update_matches_a_rebuild():
    """Updating the LSGrid in place, across perturbations, must give exactly what a fresh LSGrid gives."""
    net = load_net_from_file(str(_GRID))
    # TODO(lightsim2grid>=X): a released lightsim2grid has no update_powerlines_parameters /
    # update_trafos_parameters yet (see the TODO on the lightsim2grid extra in pyproject.toml), so
    # here every branch-parameter change falls back to a rebuild instead of going in place. Once a
    # release has them, this starts asserting n_rebuilt == 1 automatically.
    supports_in_place = hasattr(
        l2g.to_lightsim2grid(net).ls_net,
        "update_powerlines_parameters",
    )
    rng = np.random.default_rng(0)
    converted = None
    n_compared = n_rebuilt = 0
    for step in range(30):
        p = _perturbed(net, rng, step)
        previous = converted
        converted = l2g.update_lightsim2grid(p, converted)
        n_rebuilt += converted is not previous
        fresh = l2g.to_lightsim2grid(p)
        for dc in (False, True):
            results = []
            for conv in (converted, fresh):
                try:
                    results.append(
                        _solution_arrays(
                            l2g.run_ls_pf(conv.ls_net, p, conv.mapping_l2g, dc=dc),
                            p,
                        ),
                    )
                except ValueError:  # diverged
                    results.append(None)
            assert (results[0] is None) == (results[1] is None)
            if results[0] is None:
                continue
            n_compared += 1
            for updated, rebuilt in zip(*results):
                np.testing.assert_allclose(updated, rebuilt, atol=1e-10, equal_nan=True)
    assert n_compared > 40
    if supports_in_place:
        assert n_rebuilt == 1  # only the initial build: everything else went in place


def test_structural_change_triggers_a_rebuild():
    """A change the LSGrid cannot take in place (here a tap ratio) rebuilds it."""
    from gridfm_datakit.utils.idx_brch import TAP

    net = load_net_from_file(str(_GRID))
    converted = l2g.update_lightsim2grid(net)
    p = net.copy_for_perturbation()
    trafo_row = converted.mapping_l2g.trafo_rows[0]
    p.branches[trafo_row, TAP] *= 1.05
    updated = l2g.update_lightsim2grid(p, converted)
    assert updated is not converted
    ref = l2g.to_lightsim2grid(p)
    a = _solution_arrays(l2g.run_ls_pf(updated.ls_net, p, updated.mapping_l2g), p)
    b = _solution_arrays(l2g.run_ls_pf(ref.ls_net, p, ref.mapping_l2g), p)
    for u, v in zip(a, b):
        np.testing.assert_allclose(u, v, atol=1e-10)


def test_in_place_update_applies_changes_below_lightsim2grid_tolerance():
    """lightsim2grid ignores setter changes <= 1e-7: the update must still be exact,
    otherwise the data would depend on what the worker solved before."""
    from gridfm_datakit.utils.idx_bus import PD
    from gridfm_datakit.utils.idx_gen import PG, VG

    net = load_net_from_file(str(_GRID))
    converted = l2g.update_lightsim2grid(net)
    p = net.copy_for_perturbation()
    p.gens[1:, VG] += 2e-8
    p.gens[:, PG] += 3e-8
    p.buses[:, PD] += 5e-8 * (p.buses[:, PD] != 0)
    updated = l2g.update_lightsim2grid(p, converted)
    assert updated is converted  # went in place
    fresh = l2g.to_lightsim2grid(p)
    for got, expected in zip(
        updated.ls_net.get_generators(),
        fresh.ls_net.get_generators(),
    ):
        assert got.target_vm_pu == expected.target_vm_pu
        assert got.target_p_mw == expected.target_p_mw
    a = _solution_arrays(
        l2g.run_ls_pf(updated.ls_net, p, updated.mapping_l2g, tol=1e-12),
        p,
    )
    b = _solution_arrays(
        l2g.run_ls_pf(fresh.ls_net, p, fresh.mapping_l2g, tol=1e-12),
        p,
    )
    for u, v in zip(a, b):
        np.testing.assert_allclose(u, v, atol=1e-11)
