"""
Test cases for gridfm_datakit.powsybl.convert module.

Tests bidirectional conversion between pypowsybl Network and gridfm_datakit Network.
"""

import pytest
import numpy as np


from gridfm_datakit.powsybl.api import is_powsybl_available
from gridfm_datakit.network import load_net_from_pglib, Network
from gridfm_datakit.utils.idx_bus import (
    BUS_I,
    BUS_TYPE,
    PD,
    QD,
    GS,
    BS,
    VMAX,
    VMIN,
    REF,
)
from gridfm_datakit.utils.idx_gen import (
    PMAX,
    PMIN,
)

pytestmark = pytest.mark.skipif(
    not is_powsybl_available(),
    reason="pypowsybl is not installed. Install with: pip install gridfm-datakit[powsybl]",
)


@pytest.fixture(scope="module")
def pp_ieee14():
    """Create a pypowsybl IEEE 14 bus network."""
    import pypowsybl as pp
    return pp.network.create_ieee14()

@pytest.fixture(scope="module")
def load_net_ieee14():
    """Load ieee14 into LoadNet class."""
    from pathlib import Path
    from gridfm_datakit.powsybl import load_net
    from gridfm_datakit.network import load_net_from_pglib
    import gridfm_datakit.grids as _grids_pkg

    network_path = Path(_grids_pkg.__file__).parent / "pglib_opf_case14_ieee.m"
    if not network_path.exists():
        load_net_from_pglib("case14_ieee")  # downloads the file as a side-effect
    return load_net(str(network_path))

@pytest.fixture(scope="module")
def pp_ieee9():
    """Create a pypowsybl IEEE 9 bus network."""
    import pypowsybl as pp
    return pp.network.create_ieee9()


@pytest.fixture(scope="module")
def gridfm_case14():
    """Load gridfm_datakit case14_ieee network."""
    return load_net_from_pglib("case14_ieee")


@pytest.fixture(scope="module")
def gridfm_case24():
    """Load gridfm_datakit case24_ieee_rts network."""
    return load_net_from_pglib("case24_ieee_rts")


class TestFromPowsybl:
    """Test conversion from pypowsybl Network to gridfm_datakit Network."""

    def test_from_powsybl_returns_network(self, pp_ieee14):
        """Test that from_powsybl returns a Network object."""
        from gridfm_datakit.powsybl.convert import from_powsybl

        net = from_powsybl(pp_ieee14)
        assert isinstance(net, Network), "Should return a Network object"

    def test_from_powsybl_base_mva(self, pp_ieee14):
        """Test that base_mva is correctly read from pypowsybl network."""
        from gridfm_datakit.powsybl.convert import from_powsybl

        net = from_powsybl(pp_ieee14)
        expected_base_mva = pp_ieee14.nominal_apparent_power
        assert net.baseMVA == expected_base_mva, (
            f"Base MVA should be {expected_base_mva}"
        )

    def test_from_powsybl_bus_matrix_shape(self, pp_ieee14):
        """Test that bus matrix has correct shape."""
        from gridfm_datakit.powsybl.convert import from_powsybl

        net = from_powsybl(pp_ieee14)
        buses_df = pp_ieee14.get_buses()

        assert net.buses.shape[0] == len(buses_df), (
            f"Should have {len(buses_df)} buses"
        )
        assert net.buses.shape[1] >= 13, "Bus matrix should have at least 13 columns"

    def test_from_powsybl_gen_matrix_shape(self, pp_ieee14):
        """Test that generator matrix has correct shape."""
        from gridfm_datakit.powsybl.convert import from_powsybl

        net = from_powsybl(pp_ieee14)
        gens_df = pp_ieee14.get_generators()

        assert net.gens.shape[0] == len(gens_df), (
            f"Should have {len(gens_df)} generators"
        )
        assert net.gens.shape[1] >= 10, (
            "Generator matrix should have at least 10 columns"
        )

    def test_from_powsybl_branch_matrix_shape(self, pp_ieee14):
        """Test that branch matrix has correct shape."""
        from gridfm_datakit.powsybl.convert import from_powsybl

        net = from_powsybl(pp_ieee14)
        lines_df = pp_ieee14.get_lines()
        trafos_df = pp_ieee14.get_2_windings_transformers()
        expected_branches = len(lines_df) + len(trafos_df)

        assert net.branches.shape[0] == expected_branches, (
            f"Should have {expected_branches} branches"
        )
        assert net.branches.shape[1] >= 13, (
            "Branch matrix should have at least 13 columns"
        )

    def test_from_powsybl_has_ref_bus(self, pp_ieee14):
        """Test that converted network has exactly one reference bus."""
        from gridfm_datakit.powsybl.convert import from_powsybl

        net = from_powsybl(pp_ieee14)
        ref_count = np.sum(net.buses[:, BUS_TYPE] == REF)

        assert ref_count == 1, f"Should have exactly 1 reference bus, got {ref_count}"

    def test_from_powsybl_gencost_matrix(self, pp_ieee14):
        """Test that gencost matrix is created with correct shape."""
        from gridfm_datakit.powsybl.convert import from_powsybl

        net = from_powsybl(pp_ieee14)

        assert net.gencosts.shape[0] == net.gens.shape[0], (
            "Gencost should have same number of rows as generators"
        )

    def test_from_powsybl_custom_cost_coeffs(self, pp_ieee14):
        """Test that custom cost coefficients are applied."""
        from gridfm_datakit.powsybl.convert import from_powsybl, ConversionOptions
        from gridfm_datakit.utils.idx_cost import COST, NCOST

        custom_coeffs = (0.01, 20.0, 100.0)
        # gen_costs is a dict mapping generator index (str) to cost coefficients
        gens_df = pp_ieee14.get_generators()
        gen_costs = {str(i): custom_coeffs for i in range(len(gens_df))}
        options = ConversionOptions(gen_costs=gen_costs)
        net = from_powsybl(pp_ieee14, options=options)

        for i in range(net.gencosts.shape[0]):
            assert net.gencosts[i, NCOST] == len(custom_coeffs)
            for j, coeff in enumerate(custom_coeffs):
                assert net.gencosts[i, COST + j] == coeff

    def test_from_powsybl_bus_indices_continuous(self, pp_ieee14):
        """Test that internal bus indices are continuous."""
        from gridfm_datakit.powsybl.convert import from_powsybl

        net = from_powsybl(pp_ieee14)

        internal_indices = net.buses[:, BUS_I].astype(int)
        expected_indices = np.arange(len(internal_indices))

        np.testing.assert_array_equal(
            np.sort(internal_indices),
            expected_indices,
            "Bus indices should be continuous",
        )

    def test_from_powsybl_voltage_limits(self, pp_ieee14):
        """Test that voltage limits are set."""
        from gridfm_datakit.powsybl.convert import from_powsybl

        net = from_powsybl(pp_ieee14)

        assert np.all(net.buses[:, VMAX] > 0), "VMAX should be positive"
        assert np.all(net.buses[:, VMIN] > 0), "VMIN should be positive"
        assert np.all(net.buses[:, VMAX] >= net.buses[:, VMIN]), (
            "VMAX should be >= VMIN"
        )

    def test_from_powsybl_single_connected_component(self, pp_ieee14):
        """Test that converted network passes connectivity check."""
        from gridfm_datakit.powsybl.convert import from_powsybl

        net = from_powsybl(pp_ieee14)

        assert net.check_single_connected_component(), (
            "Network should be a single connected component"
        )

    def test_from_powsybl_ieee9(self, pp_ieee9):
        """Test conversion with IEEE 9 bus network."""
        from gridfm_datakit.powsybl.convert import from_powsybl

        net = from_powsybl(pp_ieee9)

        assert isinstance(net, Network)
        assert net.buses.shape[0] > 0
        assert net.gens.shape[0] > 0
        assert net.branches.shape[0] > 0

    def test_from_powsybl_empty_network_raises(self):
        """Test that converting an empty network raises ValueError."""
        from gridfm_datakit.powsybl.convert import from_powsybl
        import pypowsybl as pp

        empty_net = pp.network.create_empty("empty")

        with pytest.raises(ValueError, match="no buses"):
            from_powsybl(empty_net)


class TestToPowsybl:
    """Test conversion from gridfm_datakit Network to pypowsybl Network."""

    def test_to_powsybl_returns_network(self, gridfm_case14):
        """Test that to_powsybl returns a ConvertedNetwork object."""
        from gridfm_datakit.powsybl.convert import to_powsybl, ConvertedNetwork
        import pypowsybl as pp

        result = to_powsybl(gridfm_case14)
        assert isinstance(result, ConvertedNetwork), (
            "Should return a ConvertedNetwork object"
        )
        assert isinstance(result.pp_net, pp.network.Network), (
            "ConvertedNetwork.pp_net should be a pypowsybl Network"
        )

    def test_to_powsybl_network_id(self, gridfm_case14):
        """Test that network ID is correctly set."""
        from gridfm_datakit.powsybl.convert import to_powsybl

        pp_net = to_powsybl(gridfm_case14, network_id="test_network").pp_net
        assert pp_net.id == "test_network", "Network ID should be 'test_network'"

    def test_to_powsybl_bus_count(self, gridfm_case14):
        """Test that correct number of buses are created."""
        from gridfm_datakit.powsybl.convert import to_powsybl

        pp_net = to_powsybl(gridfm_case14).pp_net
        buses_df = pp_net.get_buses()

        assert len(buses_df) == gridfm_case14.buses.shape[0], (
            f"Should have {gridfm_case14.buses.shape[0]} buses"
        )

    def test_to_powsybl_generator_count(self, gridfm_case14):
        """Test that correct number of generators are created."""
        from gridfm_datakit.powsybl.convert import to_powsybl

        pp_net = to_powsybl(gridfm_case14).pp_net
        gens_df = pp_net.get_generators()

        assert len(gens_df) == gridfm_case14.gens.shape[0], (
            f"Should have {gridfm_case14.gens.shape[0]} generators"
        )

    def test_to_powsybl_branch_count(self, gridfm_case14):
        """Test that correct number of branches are created."""
        from gridfm_datakit.powsybl.convert import to_powsybl

        pp_net = to_powsybl(gridfm_case14).pp_net
        lines_df = pp_net.get_lines()
        trafos_df = pp_net.get_2_windings_transformers()

        total_branches = len(lines_df) + len(trafos_df)
        assert total_branches == gridfm_case14.branches.shape[0], (
            f"Should have {gridfm_case14.branches.shape[0]} branches"
        )

    def test_to_powsybl_substations_created(self, gridfm_case14):
        """Test that substations are created."""
        from gridfm_datakit.powsybl.convert import to_powsybl

        pp_net = to_powsybl(gridfm_case14).pp_net
        substations_df = pp_net.get_substations()

        assert len(substations_df) > 0, "Should have at least one substation"

    def test_to_powsybl_voltage_levels_created(self, gridfm_case14):
        """Test that voltage levels are created."""
        from gridfm_datakit.powsybl.convert import to_powsybl

        pp_net = to_powsybl(gridfm_case14).pp_net
        vl_df = pp_net.get_voltage_levels()

        # pypowsybl's native MATPOWER loader may merge voltage levels
        assert len(vl_df) > 0, "Should have at least one voltage level"

    def test_to_powsybl_loads_created(self, gridfm_case14):
        """Test that loads are created for buses with non-zero demand."""
        from gridfm_datakit.powsybl.convert import to_powsybl

        pp_net = to_powsybl(gridfm_case14).pp_net
        loads_df = pp_net.get_loads()

        # Count buses with non-zero load
        buses_with_load = np.sum(
            (gridfm_case14.buses[:, PD] != 0) | (gridfm_case14.buses[:, QD] != 0)
        )

        assert len(loads_df) == buses_with_load, (
            f"Should have {buses_with_load} loads"
        )

    def test_to_powsybl_case24(self, gridfm_case24):
        """Test conversion with case24_ieee_rts network."""
        from gridfm_datakit.powsybl.convert import to_powsybl
        import pypowsybl as pp

        pp_net = to_powsybl(gridfm_case24).pp_net

        assert isinstance(pp_net, pp.network.Network)
        assert len(pp_net.get_buses()) == gridfm_case24.buses.shape[0]
        assert len(pp_net.get_generators()) == gridfm_case24.gens.shape[0]


class TestRoundTrip:
    """Test round-trip conversion: gridfm -> pypowsybl -> gridfm."""

    def test_roundtrip_bus_count(self, gridfm_case14):
        """Test that bus count is preserved in round-trip conversion."""
        from gridfm_datakit.powsybl.convert import from_powsybl, to_powsybl

        # gridfm -> pypowsybl -> gridfm
        pp_net = to_powsybl(gridfm_case14).pp_net
        net_back = from_powsybl(pp_net)

        assert net_back.buses.shape[0] == gridfm_case14.buses.shape[0], (
            "Bus count should be preserved in round-trip"
        )

    def test_roundtrip_gen_count(self, gridfm_case14):
        """Test that generator count is preserved in round-trip conversion."""
        from gridfm_datakit.powsybl.convert import from_powsybl, to_powsybl

        pp_net = to_powsybl(gridfm_case14).pp_net
        net_back = from_powsybl(pp_net)

        assert net_back.gens.shape[0] == gridfm_case14.gens.shape[0], (
            "Generator count should be preserved in round-trip"
        )

    def test_roundtrip_branch_count(self, gridfm_case14):
        """Test that branch count is preserved in round-trip conversion."""
        from gridfm_datakit.powsybl.convert import from_powsybl, to_powsybl

        pp_net = to_powsybl(gridfm_case14).pp_net
        net_back = from_powsybl(pp_net)

        assert net_back.branches.shape[0] == gridfm_case14.branches.shape[0], (
            "Branch count should be preserved in round-trip"
        )

    def test_roundtrip_connectivity(self, gridfm_case14):
        """Test that network connectivity is preserved in round-trip."""
        from gridfm_datakit.powsybl.convert import from_powsybl, to_powsybl

        pp_net = to_powsybl(gridfm_case14).pp_net
        net_back = from_powsybl(pp_net)

        assert net_back.check_single_connected_component(), (
            "Network should remain connected after round-trip"
        )

    def test_roundtrip_has_ref_bus(self, gridfm_case14):
        """Test that reference bus is preserved in round-trip."""
        from gridfm_datakit.powsybl.convert import from_powsybl, to_powsybl

        pp_net = to_powsybl(gridfm_case14).pp_net
        net_back = from_powsybl(pp_net)

        ref_count = np.sum(net_back.buses[:, BUS_TYPE] == REF)
        assert ref_count == 1, "Should have exactly one reference bus after round-trip"

    def test_roundtrip_powsybl_ieee14(self, pp_ieee14):
        """Test round-trip: pypowsybl -> gridfm -> pypowsybl."""
        from gridfm_datakit.powsybl.convert import from_powsybl, to_powsybl

        # pypowsybl -> gridfm -> pypowsybl
        gfm_net = from_powsybl(pp_ieee14)
        pp_back = to_powsybl(gfm_net).pp_net

        # Compare counts
        original_buses = len(pp_ieee14.get_buses())
        converted_buses = len(pp_back.get_buses())

        assert converted_buses == original_buses, (
            f"Bus count should be preserved: {original_buses} -> {converted_buses}"
        )
        

class TestRoundTripWithLoadNet:
    """Test the conservation of the to_powsybl and from_powsybl starting with load_net."""

    def _numeric(self, df):
        """Extract numeric columns as a numpy array, preserving row order."""
        return df.select_dtypes(include='number').fillna(0.0).to_numpy()

    def _assert_same_elements_same_values(self, original, roundtrip, label):
        """Assert same element IDs are present and values match per element ID.

        Aligns the round-tripped dataframe to the original's index order so the
        row-by-row numeric comparison is always in a consistent (original) order,
        regardless of the internal ordering pypowsybl uses after a round-trip.
        """
        assert set(original.index) == set(roundtrip.index), (
            f"{label}: same IDs must be present after round-trip"
        )
        aligned = roundtrip.reindex(original.index)
        np.testing.assert_allclose(self._numeric(original), self._numeric(aligned), rtol=1e-5,
                                   err_msg=f"{label}: values must match per element ID")

    def test_roundtrip_with_load_net_ieee14_gens(self, load_net_ieee14):
        """Test the roundtrip conservation of generators on ieee14, starting from load_net"""
        from gridfm_datakit.powsybl.convert import from_powsybl, to_powsybl

        pp_net = load_net_ieee14.pp_net
        pp_net_roundtrip = to_powsybl(from_powsybl(pp_net)).pp_net

        self._assert_same_elements_same_values(
            pp_net.get_generators(), pp_net_roundtrip.get_generators(), "generators"
        )

    def test_roundtrip_with_load_net_ieee14_loads(self, load_net_ieee14):
        """Test the roundtrip conservation of loads on ieee14, starting from load_net"""
        from gridfm_datakit.powsybl.convert import from_powsybl, to_powsybl

        pp_net = load_net_ieee14.pp_net
        pp_net_roundtrip = to_powsybl(from_powsybl(pp_net)).pp_net

        self._assert_same_elements_same_values(
            pp_net.get_loads(), pp_net_roundtrip.get_loads(), "loads"
        )

    def test_roundtrip_with_load_net_ieee14_buses(self, load_net_ieee14):
        """Test the roundtrip conservation of buses on ieee14, starting from load_net"""
        from gridfm_datakit.powsybl.convert import from_powsybl, to_powsybl

        pp_net = load_net_ieee14.pp_net
        pp_net_roundtrip = to_powsybl(from_powsybl(pp_net)).pp_net

        self._assert_same_elements_same_values(
            pp_net.get_buses(), pp_net_roundtrip.get_buses(), "buses"
        )

    def test_roundtrip_with_load_net_ieee14_branches(self, load_net_ieee14):
        """Test the roundtrip conservation of branches on ieee14, starting from load_net"""
        from gridfm_datakit.powsybl.convert import from_powsybl, to_powsybl

        pp_net = load_net_ieee14.pp_net
        pp_net_roundtrip = to_powsybl(from_powsybl(pp_net)).pp_net

        self._assert_same_elements_same_values(
            pp_net.get_lines(), pp_net_roundtrip.get_lines(), "lines"
        )

        xfmrs = pp_net.get_2_windings_transformers()
        xfmrs_rt = pp_net_roundtrip.get_2_windings_transformers()
        assert set(xfmrs.index) == set(xfmrs_rt.index), "Transformer IDs must match"
        if len(xfmrs) > 0:
            self._assert_same_elements_same_values(xfmrs, xfmrs_rt, "transformers")


class TestImportCheck:
    """Test the import availability check."""

    def test_check_pypowsybl_available_no_error(self):
        """Test that check_powsybl_available doesn't raise when pypowsybl is installed."""
        from gridfm_datakit.powsybl.api import check_powsybl_available

        # Should not raise since we're in a test that requires pypowsybl
        check_powsybl_available()


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_network_with_shunts(self, gridfm_case14):
        """Test conversion of network with shunt elements."""
        from gridfm_datakit.powsybl.convert import to_powsybl

        # Add some shunt admittance to a bus
        gridfm_case14.buses[0, GS] = 10.0  # MW
        gridfm_case14.buses[0, BS] = 20.0  # MVAr

        pp_net = to_powsybl(gridfm_case14).pp_net
        shunts_df = pp_net.get_shunt_compensators()

        assert len(shunts_df) >= 1, "Should have at least one shunt compensator"

    def test_all_branches_as_lines(self, gridfm_case24):
        """Test that branches are modeled as lines or transformers.

        Note: pypowsybl's native MATPOWER loader may interpret some branches
        as transformers based on voltage ratios. The total number of lines
        and transformers should cover all branches.
        """
        from gridfm_datakit.powsybl.convert import to_powsybl

        pp_net = to_powsybl(gridfm_case24).pp_net
        lines_df = pp_net.get_lines()
        trafos_df = pp_net.get_2_windings_transformers()

        # pypowsybl's loader creates lines and possibly transformers
        total_branches = len(lines_df) + len(trafos_df)
        assert total_branches > 0, "Should have at least one branch element"

    def test_generator_power_limits(self, pp_ieee14):
        """Test that generator power limits are preserved."""
        from gridfm_datakit.powsybl.convert import from_powsybl

        net = from_powsybl(pp_ieee14)
        gens_df = pp_ieee14.get_generators()

        for i, (gen_id, gen) in enumerate(gens_df.iterrows()):
            max_p = gen.get("max_p", 9999.0)
            min_p = gen.get("min_p", 0.0)

            if not np.isnan(max_p):
                assert net.gens[i, PMAX] == max_p, (
                    f"PMAX mismatch for generator {i}"
                )
            if not np.isnan(min_p):
                assert net.gens[i, PMIN] == min_p, (
                    f"PMIN mismatch for generator {i}"
                )

    def test_to_mpc_after_from_powsybl(self, pp_ieee14, tmp_path):
        """Test that a converted network can be saved to MATPOWER format."""
        from gridfm_datakit.powsybl.convert import from_powsybl

        net = from_powsybl(pp_ieee14)

        # Save to MATPOWER file
        mpc_file = tmp_path / "test_case.m"
        net.to_mpc(str(mpc_file))

        assert mpc_file.exists(), "MATPOWER file should be created"
        assert mpc_file.stat().st_size > 0, "MATPOWER file should not be empty"


class TestToPowsyblElementConservation:
    """Test that element counts are conserved during to_powsybl conversion."""

    def test_bus_count_conserved(self, gridfm_case14):
        """Test that the number of buses is conserved."""
        from gridfm_datakit.powsybl.convert import to_powsybl

        pp_net = to_powsybl(gridfm_case14).pp_net
        pp_buses = len(pp_net.get_buses())

        assert pp_buses == gridfm_case14.buses.shape[0], (
            f"Bus count should be conserved: {pp_buses} != {gridfm_case14.buses.shape[0]}"
        )

    def test_generator_count_conserved(self, gridfm_case14):
        """Test that the number of generators is conserved."""
        from gridfm_datakit.powsybl.convert import to_powsybl

        pp_net = to_powsybl(gridfm_case14).pp_net
        pp_gens = len(pp_net.get_generators())

        assert pp_gens == gridfm_case14.gens.shape[0], (
            f"Generator count should be conserved: {pp_gens} != {gridfm_case14.gens.shape[0]}"
        )

    def test_load_count_conserved(self, gridfm_case14):
        """Test that loads are created for buses with non-zero demand."""
        from gridfm_datakit.powsybl.convert import to_powsybl

        pp_net = to_powsybl(gridfm_case14).pp_net
        pp_loads = len(pp_net.get_loads())

        # Count buses with non-zero load
        buses_with_load = np.sum(
            (gridfm_case14.buses[:, PD] != 0) | (gridfm_case14.buses[:, QD] != 0)
        )

        assert pp_loads == buses_with_load, (
            f"Load count should match buses with load: {pp_loads} != {buses_with_load}"
        )

    def test_branch_count_conserved(self, gridfm_case14):
        """Test that total branches (lines + transformers) are conserved."""
        from gridfm_datakit.powsybl.convert import to_powsybl

        pp_net = to_powsybl(gridfm_case14).pp_net
        pp_lines = len(pp_net.get_lines())
        pp_trafos = len(pp_net.get_2_windings_transformers())
        total_branches = pp_lines + pp_trafos

        assert total_branches == gridfm_case14.branches.shape[0], (
            f"Branch count should be conserved: {total_branches} != {gridfm_case14.branches.shape[0]}"
        )

    def test_shunt_count_conserved(self, gridfm_case14):
        """Test that shunt compensators are created for buses with non-zero shunt."""
        from gridfm_datakit.powsybl.convert import to_powsybl

        pp_net = to_powsybl(gridfm_case14).pp_net
        pp_shunts = len(pp_net.get_shunt_compensators())

        # Count buses with non-zero shunt admittance
        buses_with_shunt = np.sum(
            (gridfm_case14.buses[:, GS] != 0) | (gridfm_case14.buses[:, BS] != 0)
        )

        assert pp_shunts == buses_with_shunt, (
            f"Shunt count should match buses with shunt: {pp_shunts} != {buses_with_shunt}"
        )

    def test_conservation_case24(self, gridfm_case24):
        """Test element conservation for case24 network."""
        from gridfm_datakit.powsybl.convert import to_powsybl

        pp_net = to_powsybl(gridfm_case24).pp_net

        # Buses
        assert len(pp_net.get_buses()) == gridfm_case24.buses.shape[0]

        # Generators
        assert len(pp_net.get_generators()) == gridfm_case24.gens.shape[0]

        # Branches (lines + transformers)
        total_branches = len(pp_net.get_lines()) + len(pp_net.get_2_windings_transformers())
        assert total_branches == gridfm_case24.branches.shape[0]

        # Loads
        buses_with_load = np.sum(
            (gridfm_case24.buses[:, PD] != 0) | (gridfm_case24.buses[:, QD] != 0)
        )
        assert len(pp_net.get_loads()) == buses_with_load

        # Shunts
        buses_with_shunt = np.sum(
            (gridfm_case24.buses[:, GS] != 0) | (gridfm_case24.buses[:, BS] != 0)
        )
        assert len(pp_net.get_shunt_compensators()) == buses_with_shunt
