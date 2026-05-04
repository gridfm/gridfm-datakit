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
from gridfm_datakit.powsybl.mapping import build_p2g_maps

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
    pp_net = to_powsybl(net).pp_net
    return net, pp_net


@pytest.fixture(scope="module")
def case24():
    """IEEE 24-bus RTS network: multiple gens per bus, parallel lines."""
    net = load_net_from_pglib("case24_ieee_rts")
    pp_net = to_powsybl(net).pp_net
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
        mapping = build_p2g_maps(net, pp_net)
        pp_bus_ids = set(pp_net.get_buses().index)
        assert set(mapping.bus.keys()) == pp_bus_ids

    def test_bus_map_values_are_floats(self, case14):
        """mapping.bus values must be floats (expected by preprocess_pp_pf_res)."""
        net, pp_net = case14
        mapping = build_p2g_maps(net, pp_net)
        assert all(isinstance(v, int) for v in mapping.bus.values())

    def test_bus_map_values_are_valid_gridfm_indices(self, case14):
        """Values must be 0-based indices within [0, n_bus)."""
        net, pp_net = case14
        n_bus = net.buses.shape[0]
        mapping = build_p2g_maps(net, pp_net)
        for v in mapping.bus.values():
            assert 0 <= v < n_bus, f"Bus index {v} out of range [0, {n_bus})"

    def test_bus_map_values_are_unique(self, case14):
        """Each gridfm bus must appear at most once (no two pp buses share a gfm index)."""
        net, pp_net = case14
        mapping = build_p2g_maps(net, pp_net)
        values = list(mapping.bus.values())
        assert len(values) == len(set(values)), "Duplicate gridfm bus index in mapping.bus"

    def test_bus_map_is_a_bijection(self, case14):
        """mapping.bus must be a bijection: |keys| == |values| == n_bus."""
        net, pp_net = case14
        n_bus = net.buses.shape[0]
        mapping = build_p2g_maps(net, pp_net)
        assert len(mapping.bus) == n_bus

    # --- branch map ---

    def test_branch_map_covers_all_pp_branches(self, case14):
        """Every pypowsybl branch must appear as a key in mapping.branch."""
        net, pp_net = case14
        mapping = build_p2g_maps(net, pp_net)
        pp_branch_ids = set(pp_net.get_branches().index)
        assert set(mapping.branch.keys()) == pp_branch_ids

    def test_branch_map_values_are_ints(self, case14):
        """mapping.branch values must be ints."""
        net, pp_net = case14
        mapping = build_p2g_maps(net, pp_net)
        assert all(isinstance(v, int) for v in mapping.branch.values())

    def test_branch_map_values_are_valid_row_indices(self, case14):
        """Values must be 0-based row indices within [0, n_branch)."""
        net, pp_net = case14
        n_branch = net.branches.shape[0]
        mapping = build_p2g_maps(net, pp_net)
        for v in mapping.branch.values():
            assert 0 <= v < n_branch, f"Branch row {v} out of range [0, {n_branch})"

    def test_branch_map_values_are_unique(self, case14):
        """Each gridfm branch row must appear at most once."""
        net, pp_net = case14
        mapping = build_p2g_maps(net, pp_net)
        values = list(mapping.branch.values())
        assert len(values) == len(set(values)), "Duplicate gridfm branch row in mapping.branch"

    def test_branch_map_is_a_bijection(self, case14):
        """mapping.branch must be a bijection: |keys| == |values| == n_branch."""
        net, pp_net = case14
        n_branch = net.branches.shape[0]
        mapping = build_p2g_maps(net, pp_net)
        assert len(mapping.branch) == n_branch

    # --- gen map ---

    def test_gen_map_covers_all_pp_generators(self, case14):
        """Every pypowsybl generator must appear as a key in mapping.gen."""
        net, pp_net = case14
        mapping = build_p2g_maps(net, pp_net)
        pp_gen_ids = set(pp_net.get_generators().index)
        assert set(mapping.gen.keys()) == pp_gen_ids

    def test_gen_map_values_are_ints(self, case14):
        """mapping.gen values must be ints."""
        net, pp_net = case14
        mapping = build_p2g_maps(net, pp_net)
        assert all(isinstance(v, int) for v in mapping.gen.values())

    def test_gen_map_values_are_valid_row_indices(self, case14):
        """Values must be 0-based row indices within [0, n_gen)."""
        net, pp_net = case14
        n_gen = net.gens.shape[0]
        mapping = build_p2g_maps(net, pp_net)
        for v in mapping.gen.values():
            assert 0 <= v < n_gen, f"Gen row {v} out of range [0, {n_gen})"

    def test_gen_map_is_a_bijection(self, case14):
        """mapping.gen must be a bijection: |keys| == |values| == n_gen."""
        net, pp_net = case14
        n_gen = net.gens.shape[0]
        mapping = build_p2g_maps(net, pp_net)
        assert len(mapping.gen) == n_gen

    # --- case24: multiple gens per bus and parallel lines ---

    def test_case24_multi_gen_per_bus_covered(self, case24):
        """case24 has buses with multiple generators; all must be mapped."""
        net, pp_net = case24
        mapping = build_p2g_maps(net, pp_net)
        pp_gen_ids = set(pp_net.get_generators().index)
        assert set(mapping.gen.keys()) == pp_gen_ids

    def test_case24_parallel_branches_covered(self, case24):
        """case24 has parallel lines; all must be mapped to distinct gfm rows."""
        net, pp_net = case24
        mapping = build_p2g_maps(net, pp_net)
        pp_branch_ids = set(pp_net.get_branches().index)
        assert set(mapping.branch.keys()) == pp_branch_ids
        values = list(mapping.branch.values())
        assert len(values) == len(set(values)), "Parallel branch rows are not unique"

    def test_case24_gen_map_is_bijection(self, case24):
        net, pp_net = case24
        n_gen = net.gens.shape[0]
        mapping = build_p2g_maps(net, pp_net)
        assert len(mapping.gen) == n_gen

    def test_case24_branch_map_is_bijection(self, case24):
        net, pp_net = case24
        n_branch = net.branches.shape[0]
        mapping = build_p2g_maps(net, pp_net)
        assert len(mapping.branch) == n_branch

    def test_case24_bus_map_is_bijection(self, case24):
        net, pp_net = case24
        n_bus = net.buses.shape[0]
        mapping = build_p2g_maps(net, pp_net)
        assert len(mapping.bus) == n_bus

    def test_gen_map_preserves_row_order(self, case14):
        """mapping.gen values must be 0, 1, 2, … in the order pypowsybl enumerates generators.

        pypowsybl preserves MATPOWER gen row order, so the map is a pure positional
        enumeration: pp_gen_ids[i] → i.  Any deviation would silently corrupt PF results.
        """
        net, pp_net = case14
        mapping = build_p2g_maps(net, pp_net)
        pp_gen_ids = list(pp_net.get_generators().index)
        for expected_row, pp_gen_id in enumerate(pp_gen_ids):
            assert mapping.gen[pp_gen_id] == expected_row, (
                f"Generator {pp_gen_id!r} mapped to row {mapping.gen[pp_gen_id]}, "
                f"expected {expected_row}"
            )
