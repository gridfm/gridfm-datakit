"""Equivalence tests for the in-memory PowerModels data path.

The solvers avoid a MATPOWER file round-trip per solve by parsing the case
once in Julia (``_gfm_init_base``) and pushing only the mutable state per
solve (``_gfm_state``). These tests assert that the resulting data dict is
field-for-field equivalent to a fresh ``PowerModels.parse_file`` of the same
perturbed network, and that solver results agree.
"""

import copy
import os
import tempfile

import numpy as np
import pytest

from gridfm_datakit.network import load_net_from_file
from gridfm_datakit.process.process_network import init_julia
from gridfm_datakit.process.solvers import (
    _julia_pm_data,
    run_opf,
)
from gridfm_datakit.utils.idx_brch import BR_B, BR_R, BR_X
from gridfm_datakit.utils.idx_bus import PD, QD, VA, VM
from gridfm_datakit.utils.idx_cost import COST
from gridfm_datakit.utils.idx_gen import PG, VG

GRID_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gridfm_datakit",
    "grids",
)


@pytest.fixture(scope="module")
def jl():
    return init_julia(max_iter=500, solver_log_dir=None)


@pytest.fixture()
def net():
    return load_net_from_file(
        os.path.join(GRID_DIR, "pglib_opf_case24_ieee_rts.m"),
    )


def perturb_richly(net, seed=0):
    """Apply every kind of mutation the pipeline can produce."""
    rng = np.random.default_rng(seed)
    n_bus = net.buses.shape[0]

    # load scenario: scale all loads, zero one previously loaded bus, and add
    # load to a bus that had none (stresses the synthetic zero-load components)
    net.Pd = net.Pd * rng.uniform(0.5, 1.5, n_bus)
    net.Qd = net.Qd * rng.uniform(0.5, 1.5, n_bus)
    loaded = np.where(net.buses[:, PD] != 0)[0]
    unloaded = np.where((net.buses[:, PD] == 0) & (net.buses[:, QD] == 0))[0]
    net.buses[loaded[0], [PD, QD]] = 0.0
    if len(unloaded) > 0:
        net.buses[unloaded[0], PD] = 12.5
        net.buses[unloaded[0], QD] = 3.1

    # OPF setpoints
    net.gens[:, PG] = rng.uniform(0, 50, net.gens.shape[0])
    net.gens[:, VG] = rng.uniform(0.98, 1.05, net.gens.shape[0])
    net.buses[:, VM] = rng.uniform(0.98, 1.05, n_bus)
    net.buses[:, VA] = rng.uniform(-10, 10, n_bus)

    # topology: drop a branch and a non-slack gen (handles PV->PQ demotion)
    net.deactivate_branches(np.array([net.idx_branches_in_service[3]]))
    droppable = [
        i for i in net.idx_gens_in_service if net.gens[i, 0] != net.ref_bus_idx
    ]
    net.deactivate_gens(np.array([droppable[0]]))

    # admittance + cost perturbations
    scale = rng.uniform(0.8, 1.2, net.branches.shape[0])
    net.branches[:, BR_R] = net.branches[:, BR_R] * scale
    net.branches[:, BR_X] = net.branches[:, BR_X] * scale
    net.branches[:, BR_B] = net.branches[:, BR_B] * scale
    net.gencosts[:, COST] = net.gencosts[:, COST] * rng.uniform(
        0.5,
        2.0,
        net.gencosts.shape[0],
    )
    return net


def parse_reference(net, jl):
    """Fresh parse_file of the perturbed net — the ground truth."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as f:
        path = f.name
    try:
        net.to_mpc(path)
        jl.seval(f'_ref_data = PowerModels.parse_file(raw"{path}")')
        return jl._ref_data
    finally:
        os.unlink(path)


COMPONENT_FIELDS = {
    "bus": ["bus_type", "vm", "va", "base_kv", "vmin", "vmax", "index"],
    "gen": [
        "pg",
        "qg",
        "vg",
        "gen_status",
        "gen_bus",
        "pmax",
        "pmin",
        "qmax",
        "qmin",
        "cost",
        "ncost",
    ],
    "branch": [
        "br_r",
        "br_x",
        "b_fr",
        "b_to",
        "br_status",
        "f_bus",
        "t_bus",
        "tap",
        "shift",
        "rate_a",
        "angmin",
        "angmax",
    ],
}


def test_state_matches_fresh_parse(net, jl):
    perturbed = perturb_richly(copy.deepcopy(net))

    data = _julia_pm_data(perturbed, jl)
    ref = parse_reference(perturbed, jl)

    for comp, fields in COMPONENT_FIELDS.items():
        assert set(data[comp].keys()) == set(ref[comp].keys()), comp
        for key in ref[comp]:
            for field in fields:
                got = np.asarray(data[comp][key][field], dtype=float)
                want = np.asarray(ref[comp][key][field], dtype=float)
                assert np.allclose(got, want, rtol=1e-12, atol=1e-12), (
                    f"{comp}[{key}].{field}: got {got}, want {want}"
                )

    # loads: compare per-bus aggregates; the in-memory path materializes a
    # zero-pd load for every bus, the reference only for loaded buses
    def loads_by_bus(d):
        out = {}
        for load in d["load"].values():
            bus = int(load["load_bus"])
            pd, qd = float(load["pd"]), float(load["qd"])
            acc = out.setdefault(bus, [0.0, 0.0])
            acc[0] += pd
            acc[1] += qd
        return out

    got_loads = loads_by_bus(data)
    want_loads = loads_by_bus(ref)
    for bus, (pd, qd) in got_loads.items():
        want_pd, want_qd = want_loads.get(bus, (0.0, 0.0))
        assert np.isclose(pd, want_pd, rtol=1e-12, atol=1e-15), f"pd bus {bus}"
        assert np.isclose(qd, want_qd, rtol=1e-12, atol=1e-15), f"qd bus {bus}"


def test_solver_results_match_file_path(net, jl):
    """Same solves via the data path and via a direct file solve must agree."""
    perturbed = perturb_richly(copy.deepcopy(net), seed=1)

    res_opf = run_opf(perturbed, jl)
    assert str(res_opf["termination_status"]) == "LOCALLY_SOLVED"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as f:
        path = f.name
    try:
        perturbed.to_mpc(path)
        ref_opf = jl.run_opf(path)
    finally:
        os.unlink(path)

    assert np.isclose(
        res_opf["objective"],
        ref_opf["objective"],
        rtol=1e-8,
    )
    for k in res_opf["solution"]["bus"]:
        assert np.isclose(
            res_opf["solution"]["bus"][k]["vm"],
            ref_opf["solution"]["bus"][k]["vm"],
            rtol=1e-8,
            atol=1e-10,
        )


def test_base_reinit_on_network_switch(jl):
    """Feeding a different base network must re-initialize the Julia base."""
    net_a = load_net_from_file(
        os.path.join(GRID_DIR, "pglib_opf_case24_ieee_rts.m"),
    )
    net_b = load_net_from_file(
        os.path.join(GRID_DIR, "pglib_opf_case14_ieee.m"),
    )

    res_a = run_opf(net_a, jl)
    res_b = run_opf(net_b, jl)
    assert len(res_a["solution"]["bus"]) == 24
    assert len(res_b["solution"]["bus"]) == 14
    # and back
    res_a2 = run_opf(net_a, jl)
    assert len(res_a2["solution"]["bus"]) == 24
