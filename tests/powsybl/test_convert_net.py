"""
Test cases for gridfm_datakit.powsybl.convert_net function.

Tests conversion from gridfm_datakit Network to LoadedNetwork with pypowsybl representation.
Note: Element count conservation tests are in test_convert.py::TestToPowsyblElementConservation
"""

import pytest
import numpy as np

from gridfm_datakit.powsybl.api import is_powsybl_available
from gridfm_datakit.network import load_net_from_pglib

pytestmark = pytest.mark.skipif(
    is_powsybl_available() is False,
    reason="pypowsybl is not installed. Install with: pip install gridfm-datakit[powsybl]",
)


@pytest.fixture
def gridfm_case14():
    """Load gridfm_datakit case14_ieee network."""
    return load_net_from_pglib("case14_ieee")


@pytest.fixture
def gridfm_case24():
    """Load gridfm_datakit case24_ieee_rts network."""
    return load_net_from_pglib("case24_ieee_rts")


@pytest.fixture
def gridfm_case57():
    """Load gridfm_datakit case57_ieee network."""
    return load_net_from_pglib("case57_ieee")


class TestConvertNet:
    """Test cases for the convert_net function."""

    def test_returns_loaded_network(self, gridfm_case14):
        """Test that convert_net returns a LoadedNetwork object."""
        from gridfm_datakit.powsybl import convert_net, LoadedNetwork

        loaded = convert_net(gridfm_case14)
        assert isinstance(loaded, LoadedNetwork)

    def test_has_pb_net(self, gridfm_case14):
        """Test that the result has a pypowsybl network."""
        from gridfm_datakit.powsybl import convert_net

        loaded = convert_net(gridfm_case14)
        assert loaded.pp_net is not None
        assert hasattr(loaded.pp_net, "get_buses")

    def test_has_gfm(self, gridfm_case14):
        """Test that the result has the original gfm network."""
        from gridfm_datakit.powsybl import convert_net

        loaded = convert_net(gridfm_case14)
        assert loaded.gfm_net is gridfm_case14

    def test_has_metadata(self, gridfm_case14):
        """Test that the result has metadata with gen_costs."""
        from gridfm_datakit.powsybl import convert_net, NetworkMetadata

        loaded = convert_net(gridfm_case14)
        assert isinstance(loaded.metadata, NetworkMetadata)
        assert hasattr(loaded.metadata, "gen_costs")

    def test_custom_network_id(self, gridfm_case14):
        """Test that convert_net accepts a custom network ID."""
        from gridfm_datakit.powsybl import convert_net

        loaded = convert_net(gridfm_case14, network_id="custom_id")
        assert loaded.pp_net.id == "custom_id"

    def test_preserves_original_network(self, gridfm_case14):
        """Test that convert_net preserves the original network unchanged."""
        from gridfm_datakit.powsybl import convert_net

        original_buses = gridfm_case14.buses.copy()
        original_gens = gridfm_case14.gens.copy()
        original_branches = gridfm_case14.branches.copy()

        loaded = convert_net(gridfm_case14)

        assert np.array_equal(loaded.gfm_net.buses, original_buses)
        assert np.array_equal(loaded.gfm_net.gens, original_gens)
        assert np.array_equal(loaded.gfm_net.branches, original_branches)

    def test_case24(self, gridfm_case24):
        """Test convert_net with case24 network."""
        from gridfm_datakit.powsybl import convert_net, LoadedNetwork

        loaded = convert_net(gridfm_case24)
        assert isinstance(loaded, LoadedNetwork)
        assert loaded.gfm_net.buses.shape[0] == 24

    def test_case57(self, gridfm_case57):
        """Test convert_net with case57 network."""
        from gridfm_datakit.powsybl import convert_net, LoadedNetwork

        loaded = convert_net(gridfm_case57)
        assert isinstance(loaded, LoadedNetwork)
        assert loaded.gfm_net.buses.shape[0] == 57


class TestConvertNetGenCosts:
    """Test that generator costs are correctly extracted."""

    def test_gen_costs_extracted(self, gridfm_case14):
        """Test that gen_costs are extracted from the gfm network."""
        from gridfm_datakit.powsybl import convert_net

        loaded = convert_net(gridfm_case14)
        assert len(loaded.metadata.gen_costs) > 0

    def test_gen_costs_count_matches_generators(self, gridfm_case14):
        """Test that number of gen_costs matches number of generators."""
        from gridfm_datakit.powsybl import convert_net

        loaded = convert_net(gridfm_case14)

        n_generators = loaded.gfm_net.gens.shape[0]
        n_gen_costs = len(loaded.metadata.gen_costs)

        assert n_gen_costs == n_generators

    def test_gen_costs_format(self, gridfm_case14):
        """Test that gen_costs have the correct format."""
        from gridfm_datakit.powsybl import convert_net

        loaded = convert_net(gridfm_case14)

        for gen_idx, costs in loaded.metadata.gen_costs.items():
            assert isinstance(gen_idx, str)
            assert isinstance(costs, tuple)
            assert len(costs) > 0

    def test_gen_costs_indices_valid(self, gridfm_case14):
        """Test that gen_costs indices are valid generator indices."""
        from gridfm_datakit.powsybl import convert_net

        loaded = convert_net(gridfm_case14)
        n_generators = loaded.gfm_net.gens.shape[0]

        for gen_idx in loaded.metadata.gen_costs.keys():
            idx = int(gen_idx)
            assert 0 <= idx < n_generators

    def test_gen_costs_coefficients_preserved(self, gridfm_case14):
        """Test that gen_cost coefficients match the gencost matrix."""
        from gridfm_datakit.powsybl import convert_net

        loaded = convert_net(gridfm_case14)
        gencost_matrix = loaded.gfm_net.gencosts

        NCOST_IDX = 3
        COST_IDX = 4

        for gen_idx, costs in loaded.metadata.gen_costs.items():
            i = int(gen_idx)
            ncost = int(gencost_matrix[i, NCOST_IDX])
            expected_coeffs = tuple(gencost_matrix[i, COST_IDX : COST_IDX + ncost])

            for j, (actual, expected) in enumerate(zip(costs, expected_coeffs)):
                assert np.isclose(actual, expected)
