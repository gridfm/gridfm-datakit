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

import pytest

from gridfm_datakit.powsybl.api import is_powsybl_available
from gridfm_datakit.network import load_net_from_pglib
from gridfm_datakit.powsybl.convert import to_powsybl
from gridfm_datakit.powsybl.mapping import build_p2g_maps, to_powsybl_with_mapping

pytestmark = pytest.mark.skipif(
    not is_powsybl_available(),
    reason="pypowsybl is not installed. Install with: pip install gridfm-datakit[powsybl]",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def case14():
    """IEEE 14-bus network: single gen per bus, no parallel branches."""
    net = load_net_from_pglib("case14_ieee")
    pp_net = to_powsybl(net)
    return net, pp_net


@pytest.fixture(scope="module")
def case24():
    """IEEE 24-bus RTS network: multiple gens per bus, parallel lines."""
    net = load_net_from_pglib("case24_ieee_rts")
    pp_net = to_powsybl(net)
    return net, pp_net


# ---------------------------------------------------------------------------
# 1. Structural tests
# ---------------------------------------------------------------------------

class TestBuildP2gMaps:
    """Structural correctness of build_p2g_maps()."""

    # --- bus map ---

    def test_bus_map_covers_all_pp_buses(self, case14):
        """Every pypowsybl bus must appear as a key in map_bus_p2g."""
        net, pp_net = case14
        map_bus_p2g, _, _ = build_p2g_maps(net, pp_net)
        pp_bus_ids = set(pp_net.get_buses().index)
        assert set(map_bus_p2g.keys()) == pp_bus_ids

    def test_bus_map_values_are_floats(self, case14):
        """map_bus_p2g values must be floats (expected by preprocess_pp_pf_res)."""
        net, pp_net = case14
        map_bus_p2g, _, _ = build_p2g_maps(net, pp_net)
        assert all(isinstance(v, float) for v in map_bus_p2g.values())

    def test_bus_map_values_are_valid_gridfm_indices(self, case14):
        """Values must be 0-based indices within [0, n_bus)."""
        net, pp_net = case14
        n_bus = net.buses.shape[0]
        map_bus_p2g, _, _ = build_p2g_maps(net, pp_net)
        for v in map_bus_p2g.values():
            assert 0.0 <= v < n_bus, f"Bus index {v} out of range [0, {n_bus})"

    def test_bus_map_values_are_unique(self, case14):
        """Each gridfm bus must appear at most once (no two pp buses share a gfm index)."""
        net, pp_net = case14
        map_bus_p2g, _, _ = build_p2g_maps(net, pp_net)
        values = list(map_bus_p2g.values())
        assert len(values) == len(set(values)), "Duplicate gridfm bus index in map_bus_p2g"

    def test_bus_map_is_a_bijection(self, case14):
        """map_bus_p2g must be a bijection: |keys| == |values| == n_bus."""
        net, pp_net = case14
        n_bus = net.buses.shape[0]
        map_bus_p2g, _, _ = build_p2g_maps(net, pp_net)
        assert len(map_bus_p2g) == n_bus

    # --- branch map ---

    def test_branch_map_covers_all_pp_branches(self, case14):
        """Every pypowsybl branch must appear as a key in map_branch_p2g."""
        net, pp_net = case14
        _, map_branch_p2g, _ = build_p2g_maps(net, pp_net)
        pp_branch_ids = set(pp_net.get_branches().index)
        assert set(map_branch_p2g.keys()) == pp_branch_ids

    def test_branch_map_values_are_ints(self, case14):
        """map_branch_p2g values must be ints."""
        net, pp_net = case14
        _, map_branch_p2g, _ = build_p2g_maps(net, pp_net)
        assert all(isinstance(v, int) for v in map_branch_p2g.values())

    def test_branch_map_values_are_valid_row_indices(self, case14):
        """Values must be 0-based row indices within [0, n_branch)."""
        net, pp_net = case14
        n_branch = net.branches.shape[0]
        _, map_branch_p2g, _ = build_p2g_maps(net, pp_net)
        for v in map_branch_p2g.values():
            assert 0 <= v < n_branch, f"Branch row {v} out of range [0, {n_branch})"

    def test_branch_map_values_are_unique(self, case14):
        """Each gridfm branch row must appear at most once."""
        net, pp_net = case14
        _, map_branch_p2g, _ = build_p2g_maps(net, pp_net)
        values = list(map_branch_p2g.values())
        assert len(values) == len(set(values)), "Duplicate gridfm branch row in map_branch_p2g"

    def test_branch_map_is_a_bijection(self, case14):
        """map_branch_p2g must be a bijection: |keys| == |values| == n_branch."""
        net, pp_net = case14
        n_branch = net.branches.shape[0]
        _, map_branch_p2g, _ = build_p2g_maps(net, pp_net)
        assert len(map_branch_p2g) == n_branch

    # --- gen map ---

    def test_gen_map_covers_all_pp_generators(self, case14):
        """Every pypowsybl generator must appear as a key in map_gen_p2g."""
        net, pp_net = case14
        _, _, map_gen_p2g = build_p2g_maps(net, pp_net)
        pp_gen_ids = set(pp_net.get_generators().index)
        assert set(map_gen_p2g.keys()) == pp_gen_ids

    def test_gen_map_values_are_ints(self, case14):
        """map_gen_p2g values must be ints."""
        net, pp_net = case14
        _, _, map_gen_p2g = build_p2g_maps(net, pp_net)
        assert all(isinstance(v, int) for v in map_gen_p2g.values())

    def test_gen_map_values_are_valid_row_indices(self, case14):
        """Values must be 0-based row indices within [0, n_gen)."""
        net, pp_net = case14
        n_gen = net.gens.shape[0]
        _, _, map_gen_p2g = build_p2g_maps(net, pp_net)
        for v in map_gen_p2g.values():
            assert 0 <= v < n_gen, f"Gen row {v} out of range [0, {n_gen})"

    def test_gen_map_is_a_bijection(self, case14):
        """map_gen_p2g must be a bijection: |keys| == |values| == n_gen."""
        net, pp_net = case14
        n_gen = net.gens.shape[0]
        _, _, map_gen_p2g = build_p2g_maps(net, pp_net)
        assert len(map_gen_p2g) == n_gen

    # --- case24: multiple gens per bus and parallel lines ---

    def test_case24_multi_gen_per_bus_covered(self, case24):
        """case24 has buses with multiple generators; all must be mapped."""
        net, pp_net = case24
        _, _, map_gen_p2g = build_p2g_maps(net, pp_net)
        pp_gen_ids = set(pp_net.get_generators().index)
        assert set(map_gen_p2g.keys()) == pp_gen_ids

    def test_case24_parallel_branches_covered(self, case24):
        """case24 has parallel lines; all must be mapped to distinct gfm rows."""
        net, pp_net = case24
        _, map_branch_p2g, _ = build_p2g_maps(net, pp_net)
        pp_branch_ids = set(pp_net.get_branches().index)
        assert set(map_branch_p2g.keys()) == pp_branch_ids
        values = list(map_branch_p2g.values())
        assert len(values) == len(set(values)), "Parallel branch rows are not unique"

    def test_case24_gen_map_is_bijection(self, case24):
        net, pp_net = case24
        n_gen = net.gens.shape[0]
        _, _, map_gen_p2g = build_p2g_maps(net, pp_net)
        assert len(map_gen_p2g) == n_gen

    def test_case24_branch_map_is_bijection(self, case24):
        net, pp_net = case24
        n_branch = net.branches.shape[0]
        _, map_branch_p2g, _ = build_p2g_maps(net, pp_net)
        assert len(map_branch_p2g) == n_branch

    def test_case24_bus_map_is_bijection(self, case24):
        net, pp_net = case24
        n_bus = net.buses.shape[0]
        map_bus_p2g, _, _ = build_p2g_maps(net, pp_net)
        assert len(map_bus_p2g) == n_bus


# ---------------------------------------------------------------------------
# 2. to_powsybl_with_mapping API tests
# ---------------------------------------------------------------------------

class TestToPowsyblWithMapping:
    """Tests for the convenience wrapper to_powsybl_with_mapping()."""

    def test_returns_four_tuple(self, case14):
        net, _ = case14
        result = to_powsybl_with_mapping(net)
        assert len(result) == 4

    def test_first_element_is_pypowsybl_network(self, case14):
        """The first return value must be a pypowsybl Network object."""
        import pypowsybl as pp
        net, _ = case14
        pp_net_out, _, _, _ = to_powsybl_with_mapping(net)
        assert isinstance(pp_net_out, pp.network.Network)

    def test_maps_match_build_p2g_maps(self, case14):
        """to_powsybl_with_mapping maps must equal build_p2g_maps called separately."""
        net, _ = case14
        pp_net_out, bus1, branch1, gen1 = to_powsybl_with_mapping(net)
        bus2, branch2, gen2 = build_p2g_maps(net, pp_net_out)
        assert bus1 == bus2
        assert branch1 == branch2
        assert gen1 == gen2

    def test_custom_network_id(self, case14):
        """network_id parameter is forwarded to pypowsybl without error."""
        net, _ = case14
        pp_net_out, _, _, _ = to_powsybl_with_mapping(net, network_id="test_net")
        assert pp_net_out is not None
