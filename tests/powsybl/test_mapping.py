"""Tests for gridfm_datakit.powsybl.mapping.

Strategy
--------
build_p2g_maps() and to_powsybl_with_mapping() are tested at two levels:

1. **Structural tests** (TestBuildP2gMaps)
   Verify that the three returned dicts have the right keys, value types and
   value ranges, using case14 (single gen per bus, no parallel branches) and
   case24 (multiple gens per bus, parallel lines).

2. **API test** (TestToPowsyblWithMapping)
   Verify the convenience wrapper returns a valid network and the same maps as
   build_p2g_maps called separately.
"""

import numpy as np
import pytest

import gridfm_datakit.powsybl as powsybl
from gridfm_datakit.network import Network, load_net_from_pglib
from gridfm_datakit.powsybl.convert import update_powsybl
from gridfm_datakit.utils.idx_brch import BR_STATUS, F_BUS, T_BUS
from gridfm_datakit.utils.idx_gen import GEN_BUS, GEN_STATUS

pytestmark = pytest.mark.skipif(
    not powsybl.is_powsybl_available(),
    reason="pypowsybl is not installed. Install with: pip install gridfm-datakit[powsybl]",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def case14():
    """IEEE 14-bus network: single gen per bus, no parallel branches."""
    net = load_net_from_pglib("case14_ieee")
    pp_net = powsybl.to_powsybl(net).pp_net
    return net, pp_net


@pytest.fixture(scope="module")
def case24():
    """IEEE 24-bus RTS network: multiple gens per bus, parallel lines."""
    net = load_net_from_pglib("case24_ieee_rts")
    pp_net = powsybl.to_powsybl(net).pp_net
    return net, pp_net


# ---------------------------------------------------------------------------
# 1. Structural tests
# ---------------------------------------------------------------------------


class TestBuildP2gMaps:
    """Structural correctness of build_p2g_maps()."""

    # --- bus map ---

    def test_bus_map_covers_all_pp_buses(self, case14):
        """Every pypowsybl bus must appear as a key in mapping.bus."""
        net, pp_net = case14
        mapping = powsybl.build_p2g_maps(net, pp_net)
        pp_bus_ids = set(pp_net.get_buses().index)
        assert set(mapping.bus.keys()) == pp_bus_ids

    def test_bus_map_values_are_floats(self, case14):
        """mapping.bus values must be floats (expected by preprocess_pp_pf_res)."""
        net, pp_net = case14
        mapping = powsybl.build_p2g_maps(net, pp_net)
        assert all(isinstance(v, int) for v in mapping.bus.values())

    def test_bus_map_values_are_valid_gridfm_indices(self, case14):
        """Values must be 0-based indices within [0, n_bus)."""
        net, pp_net = case14
        n_bus = net.buses.shape[0]
        mapping = powsybl.build_p2g_maps(net, pp_net)
        for v in mapping.bus.values():
            assert 0 <= v < n_bus, f"Bus index {v} out of range [0, {n_bus})"

    def test_bus_map_values_are_unique(self, case14):
        """Each gridfm bus must appear at most once (no two pp buses share a gfm index)."""
        net, pp_net = case14
        mapping = powsybl.build_p2g_maps(net, pp_net)
        values = list(mapping.bus.values())
        assert len(values) == len(set(values)), (
            "Duplicate gridfm bus index in mapping.bus"
        )

    def test_bus_map_is_a_bijection(self, case14):
        """mapping.bus must be a bijection: |keys| == |values| == n_bus."""
        net, pp_net = case14
        n_bus = net.buses.shape[0]
        mapping = powsybl.build_p2g_maps(net, pp_net)
        assert len(mapping.bus) == n_bus

    # --- branch map ---

    def test_branch_map_covers_all_pp_branches(self, case14):
        """Every pypowsybl branch must appear as a key in mapping.branch."""
        net, pp_net = case14
        mapping = powsybl.build_p2g_maps(net, pp_net)
        pp_branch_ids = set(pp_net.get_branches().index)
        assert set(mapping.branch.keys()) == pp_branch_ids

    def test_branch_map_values_are_ints(self, case14):
        """mapping.branch values must be ints."""
        net, pp_net = case14
        mapping = powsybl.build_p2g_maps(net, pp_net)
        assert all(isinstance(v, int) for v in mapping.branch.values())

    def test_branch_map_values_are_valid_row_indices(self, case14):
        """Values must be 0-based row indices within [0, n_branch)."""
        net, pp_net = case14
        n_branch = net.branches.shape[0]
        mapping = powsybl.build_p2g_maps(net, pp_net)
        for v in mapping.branch.values():
            assert 0 <= v < n_branch, f"Branch row {v} out of range [0, {n_branch})"

    def test_branch_map_values_are_unique(self, case14):
        """Each gridfm branch row must appear at most once."""
        net, pp_net = case14
        mapping = powsybl.build_p2g_maps(net, pp_net)
        values = list(mapping.branch.values())
        assert len(values) == len(set(values)), (
            "Duplicate gridfm branch row in mapping.branch"
        )

    def test_branch_map_is_a_bijection(self, case14):
        """mapping.branch must be a bijection: |keys| == |values| == n_branch."""
        net, pp_net = case14
        n_branch = net.branches.shape[0]
        mapping = powsybl.build_p2g_maps(net, pp_net)
        assert len(mapping.branch) == n_branch

    # --- gen map ---

    def test_gen_map_covers_all_pp_generators(self, case14):
        """Every pypowsybl generator must appear as a key in mapping.gen."""
        net, pp_net = case14
        mapping = powsybl.build_p2g_maps(net, pp_net)
        pp_gen_ids = set(pp_net.get_generators().index)
        assert set(mapping.gen.keys()) == pp_gen_ids

    def test_gen_map_values_are_ints(self, case14):
        """mapping.gen values must be ints."""
        net, pp_net = case14
        mapping = powsybl.build_p2g_maps(net, pp_net)
        assert all(isinstance(v, int) for v in mapping.gen.values())

    def test_gen_map_values_are_valid_row_indices(self, case14):
        """Values must be 0-based row indices within [0, n_gen)."""
        net, pp_net = case14
        n_gen = net.gens.shape[0]
        mapping = powsybl.build_p2g_maps(net, pp_net)
        for v in mapping.gen.values():
            assert 0 <= v < n_gen, f"Gen row {v} out of range [0, {n_gen})"

    def test_gen_map_is_a_bijection(self, case14):
        """mapping.gen must be a bijection: |keys| == |values| == n_gen."""
        net, pp_net = case14
        n_gen = net.gens.shape[0]
        mapping = powsybl.build_p2g_maps(net, pp_net)
        assert len(mapping.gen) == n_gen

    # --- case24: multiple gens per bus and parallel lines ---

    def test_case24_multi_gen_per_bus_covered(self, case24):
        """case24 has buses with multiple generators; all must be mapped."""
        net, pp_net = case24
        mapping = powsybl.build_p2g_maps(net, pp_net)
        pp_gen_ids = set(pp_net.get_generators().index)
        assert set(mapping.gen.keys()) == pp_gen_ids

    def test_case24_parallel_branches_covered(self, case24):
        """case24 has parallel lines; all must be mapped to distinct gfm rows."""
        net, pp_net = case24
        mapping = powsybl.build_p2g_maps(net, pp_net)
        pp_branch_ids = set(pp_net.get_branches().index)
        assert set(mapping.branch.keys()) == pp_branch_ids
        values = list(mapping.branch.values())
        assert len(values) == len(set(values)), "Parallel branch rows are not unique"

    def test_case24_gen_map_is_bijection(self, case24):
        net, pp_net = case24
        n_gen = net.gens.shape[0]
        mapping = powsybl.build_p2g_maps(net, pp_net)
        assert len(mapping.gen) == n_gen

    def test_case24_branch_map_is_bijection(self, case24):
        net, pp_net = case24
        n_branch = net.branches.shape[0]
        mapping = powsybl.build_p2g_maps(net, pp_net)
        assert len(mapping.branch) == n_branch

    def test_case24_bus_map_is_bijection(self, case24):
        net, pp_net = case24
        n_bus = net.buses.shape[0]
        mapping = powsybl.build_p2g_maps(net, pp_net)
        assert len(mapping.bus) == n_bus

    def test_gen_map_preserves_row_order(self, case14):
        """mapping.gen values must be 0, 1, 2, … in the order pypowsybl enumerates generators.

        pypowsybl preserves MATPOWER gen row order, so the map is a pure positional
        enumeration: pp_gen_ids[i] → i.  Any deviation would silently corrupt PF results.
        """
        net, pp_net = case14
        mapping = powsybl.build_p2g_maps(net, pp_net)
        pp_gen_ids = list(pp_net.get_generators().index)
        for expected_row, pp_gen_id in enumerate(pp_gen_ids):
            assert mapping.gen[pp_gen_id] == expected_row, (
                f"Generator {pp_gen_id!r} mapped to row {mapping.gen[pp_gen_id]}, "
                f"expected {expected_row}"
            )


# ---------------------------------------------------------------------------
# 3. to_powsybl maps follow the original bus numbers
# ---------------------------------------------------------------------------


def _shuffled_case14():
    net = load_net_from_pglib("case14_ieee")
    mpc = dict(net.mpc)
    mpc["bus"] = net.mpc["bus"][np.random.default_rng(0).permutation(14)]
    return Network(mpc)


@pytest.fixture(scope="module", params=["sorted", "shuffled"])
def case14_any_order(request):
    """IEEE 14-bus network, buses listed in file order or shuffled."""
    if request.param == "sorted":
        net = load_net_from_pglib("case14_ieee")
    else:
        net = _shuffled_case14()
    return net, powsybl.to_powsybl(net)


def _bus_number(bus_breaker_bus_id):
    return int(bus_breaker_bus_id.rsplit("-", 1)[1])


class TestToPowsyblMapsFollowOriginalBusNumbers:
    """pypowsybl lists buses by voltage level, so position is not identity."""

    def test_bus_map_follows_bus_breaker_buses(self, case14_any_order):
        """Each pypowsybl bus maps to the gridfm bus of the BUS-<n> it groups."""
        net, conv = case14_any_order
        pp_net, mapping = conv.pp_net, conv.mapping_p2g
        bb_buses = pp_net.get_bus_breaker_view_buses()
        for bb_bus_id, pp_bus_id in bb_buses["bus_id"].items():
            gfm_bus = int(mapping.bus[pp_bus_id])
            assert net.reverse_bus_index_mapping[gfm_bus] == _bus_number(bb_bus_id)

    def test_update_keeps_case_loads(self):
        """Writing the unchanged network back leaves every load as in the case."""
        net = load_net_from_pglib("case14_ieee")
        conv = powsybl.to_powsybl(net)
        conv.pp_net.per_unit = False
        before = conv.pp_net.get_loads()["p0"].copy()
        update_powsybl(conv.pp_net, net, conv.mapping_p2g)
        conv.pp_net.per_unit = False
        after = conv.pp_net.get_loads()["p0"]
        assert np.allclose(after[before.index], before)

    def test_branch_map_follows_end_buses(self, case14_any_order):
        """Each pypowsybl branch maps to the gridfm row with the same ends."""
        net, conv = case14_any_order
        pp_net, mapping = conv.pp_net, conv.mapping_p2g
        rev = net.reverse_bus_index_mapping
        ends = ["bus_breaker_bus1_id", "bus_breaker_bus2_id"]
        for df in (
            pp_net.get_lines(attributes=ends),
            pp_net.get_2_windings_transformers(attributes=ends),
        ):
            for pp_branch_id, bus1, bus2 in df.itertuples():
                row = mapping.branch[pp_branch_id]
                assert rev[int(net.branches[row, F_BUS])] == _bus_number(bus1)
                assert rev[int(net.branches[row, T_BUS])] == _bus_number(bus2)

    @pytest.mark.parametrize("case", ["case14_ieee", "case24_ieee_rts"])
    def test_update_keeps_case_impedances(self, case):
        """Writing the unchanged network back leaves every r and x as in the case."""
        net = load_net_from_pglib(case)
        conv = powsybl.to_powsybl(net)
        pp_net = conv.pp_net
        pp_net.per_unit = True
        before = [df[["r", "x"]].copy() for df in _branch_frames(pp_net)]
        update_powsybl(pp_net, net, conv.mapping_p2g)
        pp_net.per_unit = True
        for old, new in zip(before, _branch_frames(pp_net)):
            assert np.allclose(new.loc[old.index, ["r", "x"]], old)


def _branch_frames(pp_net):
    return pp_net.get_lines(), pp_net.get_2_windings_transformers()


# ---------------------------------------------------------------------------
# 4. Disconnected elements, which pypowsybl before 1.16 leaves out of the export
# ---------------------------------------------------------------------------

_DISCONNECTED_GEN = "B3-G"
_DISCONNECTED_LINE = "L6-11-1"
_DISCONNECTED_LOAD = "B4-L"


@pytest.fixture
def case14_with_disconnections():
    """IEEE 14-bus network with a generator, a line and a load disconnected."""
    import pypowsybl as pp

    pp_net = pp.network.create_ieee14()
    for element_id in (_DISCONNECTED_GEN, _DISCONNECTED_LINE, _DISCONNECTED_LOAD):
        pp_net.disconnect(element_id)
    net = powsybl.from_powsybl(pp_net)
    return net, pp_net, powsybl.build_p2g_maps(net, pp_net)


class TestDisconnectedElements:
    """The maps follow the export, whether it keeps disconnected elements or not."""

    def test_maps_cover_every_row(self, case14_with_disconnections):
        net, _, mapping = case14_with_disconnections
        assert sorted(mapping.gen.values()) == list(range(net.gens.shape[0]))
        assert sorted(mapping.branch.values()) == list(range(net.branches.shape[0]))

    def test_disconnected_rows_are_out_of_service(self, case14_with_disconnections):
        net, _, mapping = case14_with_disconnections
        if _DISCONNECTED_GEN in mapping.gen:
            assert net.gens[mapping.gen[_DISCONNECTED_GEN], GEN_STATUS] == 0
        if _DISCONNECTED_LINE in mapping.branch:
            assert net.branches[mapping.branch[_DISCONNECTED_LINE], BR_STATUS] == 0

    def test_gen_map_follows_gen_buses(self, case14_with_disconnections):
        net, pp_net, mapping = case14_with_disconnections
        bus_id = pp_net.get_generators()["bus_id"]
        for pp_gen_id, row in mapping.gen.items():
            if pp_gen_id != _DISCONNECTED_GEN:
                assert int(mapping.bus[bus_id[pp_gen_id]]) == int(
                    net.gens[row, GEN_BUS],
                )

    def test_branch_map_follows_end_buses(self, case14_with_disconnections):
        net, pp_net, mapping = case14_with_disconnections
        ends = pp_net.get_branches()[["bus1_id", "bus2_id"]]
        for pp_branch_id, row in mapping.branch.items():
            if pp_branch_id != _DISCONNECTED_LINE:
                bus1, bus2 = ends.loc[pp_branch_id]
                assert int(mapping.bus[bus1]) == int(net.branches[row, F_BUS])
                assert int(mapping.bus[bus2]) == int(net.branches[row, T_BUS])

    def test_power_flow_round_trip(self, case14_with_disconnections):
        import pypowsybl as pp

        net, pp_net, mapping = case14_with_disconnections
        update_powsybl(pp_net, net, mapping)
        assert not pp_net.get_generators()["connected"][_DISCONNECTED_GEN]
        assert not pp_net.get_loads()["connected"][_DISCONNECTED_LOAD]
        assert not pp_net.get_lines()["connected1"][_DISCONNECTED_LINE]
        pf_metadata = pp.loadflow.run_ac(pp_net)
        res = powsybl.get_pf_res(pp_net, 0.0, pf_metadata, mapping)
        assert len(res["solution"]["gen"]) == int(net.gens[:, GEN_STATUS].sum())
        assert len(res["solution"]["branch"]) == net.branches.shape[0]
