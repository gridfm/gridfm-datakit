"""Checks of the changes that make the lightsim2grid solver faster (no Julia needed)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matpowercaseframes import CaseFrames
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from gridfm_datakit import lightsim2grid as l2g
from gridfm_datakit.network import Network, makeYbus
from gridfm_datakit.process.process_network import (
    _solution_arrays,
    pf_post_processing,
)
from gridfm_datakit.utils.idx_brch import BR_STATUS, F_BUS, T_BUS
from gridfm_datakit.utils.idx_bus import BUS_TYPE, PV

pytestmark = pytest.mark.skipif(
    not l2g.is_lightsim2grid_available(),
    reason="lightsim2grid is not installed. Install with: pip install gridfm-datakit[lightsim2grid]",
)

_GRID = Path(__file__).parents[1] / "powsybl" / "grids" / "ieee14.m"


def _load_net() -> Network:
    # same as load_net_from_file, without the PowerModels correction (which needs Julia)
    frames = CaseFrames(str(_GRID))
    mpc = {
        key: getattr(frames, key).values
        if isinstance(getattr(frames, key), pd.DataFrame)
        else getattr(frames, key)
        for key in frames._attributes
    }
    return Network(mpc)


def _net_with_generator_off() -> Network:
    """A copy of the network with the only generator of a PV bus out of service."""
    net = _load_net()
    for gen in net.idx_gens_in_service:
        perturbed = net.copy_for_perturbation()
        perturbed.deactivate_gens([gen])
        if np.any(perturbed.buses[:, BUS_TYPE] != net.buses[:, BUS_TYPE]):
            return perturbed
    raise AssertionError("no generator whose outage changes a bus type")


def test_generator_outage_does_not_rebuild_the_lsgrid():
    net = _load_net()
    perturbed = _net_with_generator_off()
    assert (net.buses[:, BUS_TYPE] == PV).any()
    assert np.any(perturbed.buses[:, BUS_TYPE] != net.buses[:, BUS_TYPE])

    converted = l2g.update_lightsim2grid(net)
    # a PV bus turned PQ and back is not a structural change: updated in place
    assert l2g.update_lightsim2grid(perturbed, converted) is converted
    assert l2g.update_lightsim2grid(net, converted) is converted

    # and the in-place grid gives what a freshly built one gives
    updated = l2g.update_lightsim2grid(perturbed, converted)
    fresh = l2g.to_lightsim2grid(perturbed)
    for dc in (False, True):
        a = l2g.run_ls_pf(updated.ls_net, perturbed, updated.mapping_l2g, dc=dc)
        b = l2g.run_ls_pf(fresh.ls_net, perturbed, fresh.mapping_l2g, dc=dc)
        for x, y in zip(_solution_arrays(a, perturbed), _solution_arrays(b, perturbed)):
            np.testing.assert_allclose(x, y, rtol=0, atol=1e-9)


def test_klu_solvers_are_selected_when_available():
    converted = l2g.to_lightsim2grid(_load_net())
    available = {a.name for a in converted.ls_net.available_default_algorithms()}
    if "NR_KLU" not in available:
        pytest.skip("this lightsim2grid build has no KLU")
    assert converted.ls_net.get_algo_type().name == "NR_KLU"
    assert converted.ls_net.get_dc_algo_type().name == "DC_KLU"


def test_the_chosen_algorithms_are_logged_once(caplog, monkeypatch):
    from gridfm_datakit.lightsim2grid import convert

    monkeypatch.setattr(convert, "_logged_algorithms", set())
    with caplog.at_level("INFO", logger=convert.logger.name):
        converted = l2g.to_lightsim2grid(_load_net())
        l2g.to_lightsim2grid(_load_net())  # same choice: not logged again
    messages = [r.getMessage() for r in caplog.records if r.name == convert.logger.name]
    assert len(messages) == 1
    assert converted.ls_net.get_algo_type().name in messages[0]
    assert converted.ls_net.get_dc_algo_type().name in messages[0]


def test_array_results_of_the_wrong_shape_are_rejected():
    net = _net_with_generator_off()
    converted = l2g.to_lightsim2grid(net)
    result = l2g.run_ls_pf(
        converted.ls_net,
        net,
        converted.mapping_l2g,
        as_arrays=True,
    )
    branch_flows, gen_pq, bus_vmva = result["solution"]["arrays"]
    for wrong in (
        (branch_flows[:-1], gen_pq, bus_vmva),
        (branch_flows, gen_pq[:-1], bus_vmva),
        (branch_flows, gen_pq, bus_vmva[:-1]),
    ):
        result["solution"]["arrays"] = wrong
        with pytest.raises(AssertionError, match="shape"):
            _solution_arrays(result, net)


@pytest.mark.parametrize("dc", [False, True])
def test_array_results_equal_the_dict_results(dc):
    net = _net_with_generator_off()
    converted = l2g.to_lightsim2grid(net)
    as_dict = l2g.run_ls_pf(converted.ls_net, net, converted.mapping_l2g, dc=dc)
    as_arrays = l2g.run_ls_pf(
        converted.ls_net,
        net,
        converted.mapping_l2g,
        dc=dc,
        as_arrays=True,
    )
    assert "arrays" in as_arrays["solution"]
    for x, y in zip(
        _solution_arrays(as_dict, net),
        _solution_arrays(as_arrays, net),
    ):
        np.testing.assert_allclose(x, y, rtol=0, atol=1e-12)
    assert ("Ybus" in as_arrays["solution"]) == (not dc)  # only valid after an AC solve


def test_ybus_of_lightsim2grid_matches_makeybus():
    net = _net_with_generator_off()
    converted = l2g.to_lightsim2grid(net)
    result = l2g.run_ls_pf(
        converted.ls_net,
        net,
        converted.mapping_l2g,
        as_arrays=True,
    )
    ybus = result["solution"]["Ybus"]
    expected = makeYbus(net.baseMVA, net.buses, net.branches)[0].tocsr()
    expected.sort_indices()
    assert np.array_equal(ybus.indptr, expected.indptr)
    assert np.array_equal(ybus.indices, expected.indices)
    np.testing.assert_allclose(ybus.data, expected.data, rtol=1e-12, atol=1e-12)


def test_post_processing_with_the_lightsim2grid_ybus_matches_makeybus():
    net = _net_with_generator_off()
    converted = l2g.to_lightsim2grid(net)
    kwargs = {"as_arrays": True}
    res = l2g.run_ls_pf(converted.ls_net, net, converted.mapping_l2g, **kwargs)
    res_dc = l2g.run_ls_pf(
        converted.ls_net,
        net,
        converted.mapping_l2g,
        dc=True,
        **kwargs,
    )
    with_ls_ybus = pf_post_processing(0, net, res, res_dc, True)
    res["solution"].pop("Ybus")  # falls back to makeYbus
    with_makeybus = pf_post_processing(0, net, res, res_dc, True)
    for name in ("bus", "gen", "branch", "Y_bus", "runtime"):
        np.testing.assert_allclose(
            with_ls_ybus[name],
            with_makeybus[name],
            rtol=0,
            atol=1e-9,
            equal_nan=True,
            err_msg=name,
        )


def test_connectivity_check_matches_the_reference_implementation():
    net = _load_net()

    def reference(n: Network) -> bool:
        in_service = n.branches[:, BR_STATUS] == 1
        f = n.branches[in_service, F_BUS].astype(np.int64)
        t = n.branches[in_service, T_BUS].astype(np.int64)
        size = n.buses.shape[0]
        adjacency = csr_matrix(
            (np.ones(f.size, dtype=np.uint8), (f, t)),
            shape=(size, size),
        )
        return connected_components(adjacency, directed=False, return_labels=False) == 1

    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        p = net.copy_for_perturbation()
        out = rng.choice(p.branches.shape[0], rng.integers(0, 5), replace=False)
        p.branches[out, BR_STATUS] = 0
        expected = reference(p)
        seen.add(expected)
        assert p.check_single_connected_component() == expected
    assert seen == {True, False}, "the test should cover connected and islanded grids"
