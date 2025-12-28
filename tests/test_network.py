"""
Test cases for gridfm_datakit.network module
Tests the Network class and load functions with matpowercaseframes integration
"""

import pytest
import numpy as np
from gridfm_datakit.network import load_net_from_pglib, Network


class TestNetwork:
    """Test class for Network class and load functions"""

    def test_load_net_from_pglib_basic(self):
        """Test basic functionality of load_net_from_pglib with case24_ieee_rts"""
        # Load the network
        network = load_net_from_pglib("case24_ieee_rts")

        # Verify return type
        assert isinstance(network, Network), "Should return a Network object"

        # Verify basic network properties
        assert network.baseMVA == 100, "Base MVA should be 100"
        assert network.version() == "2" or network.version() == 2, (
            "Version should be '2' or 2"
        )

        # Verify matrix shapes (note: indices are 0-based in Network class)
        assert network.buses.shape == (24, 13), (
            "Bus matrix should have 24 buses and 13 columns"
        )
        assert network.gens.shape[0] == 33, "Should have 33 generators"
        assert network.branches.shape == (38, 13), (
            "Branch matrix should have 38 branches and 13 columns"
        )
        assert network.gencosts.shape[0] == 33, "Should have 33 generator cost entries"

    def test_network_matrix_structure(self):
        """Test that Network matrices contain all required columns according to MATPOWER specification"""
        # Load the network
        network = load_net_from_pglib("case24_ieee_rts")

        # Test bus matrix (should have 13 columns)
        bus_matrix = network.buses
        assert len(bus_matrix.shape) == 2, "Bus matrix should be 2D"
        assert bus_matrix.shape[1] >= 13, (
            f"Bus matrix should have at least 13 columns, got {bus_matrix.shape[1]}"
        )

        # Test gen matrix (should have 10 columns in standard MATPOWER format)
        gen_matrix = network.gens
        assert len(gen_matrix.shape) == 2, "Gen matrix should be 2D"
        assert gen_matrix.shape[1] >= 10, (
            f"Gen matrix should have at least 10 columns, got {gen_matrix.shape[1]}"
        )

        # Test branch matrix (should have 13 columns)
        branch_matrix = network.branches
        assert len(branch_matrix.shape) == 2, "Branch matrix should be 2D"
        assert branch_matrix.shape[1] >= 13, (
            f"Branch matrix should have at least 13 columns, got {branch_matrix.shape[1]}"
        )

        # Test gencost matrix (should exist)
        gencost_matrix = network.gencosts
        assert len(gencost_matrix.shape) == 2, "Gencost matrix should be 2D"
        assert gencost_matrix.shape[0] == gen_matrix.shape[0], (
            "Gencost should have same number of rows as gen matrix"
        )

    def test_network_data_types(self):
        """Test that Network data contains appropriate data types"""
        # Load the network
        network = load_net_from_pglib("case24_ieee_rts")

        # Test that matrices are numpy arrays
        assert isinstance(network.buses, np.ndarray), "Bus matrix should be numpy array"
        assert isinstance(network.gens, np.ndarray), "Gen matrix should be numpy array"
        assert isinstance(network.branches, np.ndarray), (
            "Branch matrix should be numpy array"
        )
        assert isinstance(network.gencosts, np.ndarray), (
            "Gencost matrix should be numpy array"
        )

        # Test that scalar values are appropriate types
        assert isinstance(network.baseMVA, (int, float, np.number)), (
            "Base MVA should be numeric"
        )
        assert isinstance(network.version(), (int, float, np.number, str)), (
            "Version should be numeric or string"
        )

    def test_network_bus_data(self):
        """Test specific bus matrix data structure"""
        # Load the network
        network = load_net_from_pglib("case24_ieee_rts")

        bus_matrix = network.buses

        # Test bus numbering (Network class converts to continuous 0-based indexing) using mapping
        bus_numbers = bus_matrix[:, 0].astype(
            int,
        )  # First column is bus index (continuous)
        n_buses = bus_matrix.shape[0]
        assert set(bus_numbers.tolist()) == set(range(n_buses)), (
            "Bus indices should be continuous 0..n-1"
        )
        # Validate reverse mapping recovers original MATPOWER IDs
        mapped_back = np.array(
            [network.reverse_bus_index_mapping[int(i)] for i in bus_numbers],
        )
        assert set(mapped_back.tolist()) == set(
            network.original_bus_indices.astype(int).tolist(),
        ), "Reverse mapping should match original IDs"

        # Test bus types (should be 1, 2, or 3)
        bus_types = bus_matrix[:, 1]  # Second column is bus type
        assert np.all(np.isin(bus_types, [1, 2, 3])), (
            "Bus types should be 1 (PQ), 2 (PV), or 3 (ref)"
        )

        # Test that there's exactly one reference bus (type 3)
        ref_buses = np.sum(bus_types == 3)
        assert ref_buses == 1, f"Should have exactly 1 reference bus, found {ref_buses}"

    def test_network_gen_data(self):
        """Test specific generator matrix data structure"""
        # Load the network
        network = load_net_from_pglib("case24_ieee_rts")

        gen_matrix = network.gens

        # Test generator bus connections (should be valid bus numbers, 0-based)
        gen_buses = gen_matrix[:, 0]  # First column is generator bus
        assert np.all(gen_buses >= 0), "Generator buses should be >= 0 (0-based)"
        assert np.all(gen_buses < network.buses.shape[0]), (
            "Generator buses should be < number of buses (0-based)"
        )

        # Test generator status (should be 0 or 1)
        gen_status = gen_matrix[:, 7]  # 8th column is generator status
        assert np.all(np.isin(gen_status, [0, 1])), "Generator status should be 0 or 1"

    def test_network_branch_data(self):
        """Test specific branch matrix data structure"""
        # Load the network
        network = load_net_from_pglib("case24_ieee_rts")

        branch_matrix = network.branches

        # Test branch connections (should be valid bus numbers, 0-based)
        from_buses = branch_matrix[:, 0]  # First column is from bus
        to_buses = branch_matrix[:, 1]  # Second column is to bus

        assert np.all(from_buses >= 0), "From buses should be >= 0 (0-based)"
        assert np.all(from_buses < network.buses.shape[0]), (
            "From buses should be < number of buses (0-based)"
        )
        assert np.all(to_buses >= 0), "To buses should be >= 0 (0-based)"
        assert np.all(to_buses < network.buses.shape[0]), (
            "To buses should be < number of buses (0-based)"
        )

        # Test branch status (should be 0 or 1)
        branch_status = branch_matrix[:, 10]  # 11th column is branch status
        assert np.all(np.isin(branch_status, [0, 1])), "Branch status should be 0 or 1"

    def test_load_net_from_pglib_invalid_case(self):
        """Test that function raises appropriate error for invalid case name"""
        with pytest.raises(
            Exception,
        ):  # Could be requests.exceptions.RequestException or other
            load_net_from_pglib("non_existent_case")

    def test_network_matpowercaseframes_integration(self):
        """Test that the Network class properly integrates with matpowercaseframes"""
        # Load the network
        network = load_net_from_pglib("case24_ieee_rts")

        # Verify that the Network class preserves matpowercaseframes data
        # This tests the specific conversion logic we implemented
        assert network.version() is not None, (
            "Version should be preserved from matpowercaseframes"
        )
        assert network.baseMVA is not None, (
            "Base MVA should be preserved from matpowercaseframes"
        )

        # Verify that DataFrames were converted to numpy arrays
        for attr_name in ["buses", "gens", "branches", "gencosts"]:
            attr = getattr(network, attr_name)
            assert isinstance(attr, np.ndarray), (
                f"{attr_name} should be converted to numpy array"
            )
            assert len(attr.shape) == 2, f"{attr_name} should be 2D array"

    def test_network_properties(self):
        """Test Network class properties"""
        # Load the network
        network = load_net_from_pglib("case24_ieee_rts")

        # Test idx_gens_in_service property
        idx_gens_in_service = network.idx_gens_in_service
        assert isinstance(idx_gens_in_service, np.ndarray), (
            "idx_gens_in_service should be numpy array"
        )
        assert len(idx_gens_in_service) > 0, (
            "Should have at least one generator in service"
        )
        assert np.all(idx_gens_in_service >= 0), "Generator indices should be >= 0"
        assert np.all(idx_gens_in_service < network.gens.shape[0]), (
            "Generator indices should be < number of generators"
        )

        # Test idx_branches_in_service property
        idx_branches_in_service = network.idx_branches_in_service
        assert isinstance(idx_branches_in_service, np.ndarray), (
            "idx_branches_in_service should be numpy array"
        )
        assert len(idx_branches_in_service) > 0, (
            "Should have at least one branch in service"
        )
        assert np.all(idx_branches_in_service >= 0), "Branch indices should be >= 0"
        assert np.all(idx_branches_in_service < network.branches.shape[0]), (
            "Branch indices should be < number of branches"
        )

        # Test Pd property (active power demand)
        pd = network.Pd
        assert isinstance(pd, np.ndarray), "Pd should be numpy array"
        assert len(pd) == network.buses.shape[0], (
            "Pd should have same length as number of buses"
        )

        # Test Qd property (reactive power demand)
        qd = network.Qd
        assert isinstance(qd, np.ndarray), "Qd should be numpy array"
        assert len(qd) == network.buses.shape[0], (
            "Qd should have same length as number of buses"
        )

        # Test Pg property (active power generation)
        pg = network.Pg_gen
        assert isinstance(pg, np.ndarray), "Pg should be numpy array"
        assert len(pg) == network.gens.shape[0], (
            "Pg should have same length as number of generators"
        )

        # Test Qg property (reactive power generation)
        qg = network.Qg_gen
        assert isinstance(qg, np.ndarray), "Qg should be numpy array"
        assert len(qg) == network.gens.shape[0], (
            "Qg should have same length as number of generators"
        )

    def test_network_connected_component_check(self):
        """Test the check_single_connected_component method"""
        # Load the network
        network = load_net_from_pglib("case24_ieee_rts")

        # Test that the network forms a single connected component
        is_connected = network.check_single_connected_component()
        assert isinstance(is_connected, bool), (
            "check_single_connected_component should return boolean"
        )
        assert is_connected, "case24_ieee_rts should form a single connected component"

    def test_network_deactivate_methods(self):
        """Test the deactivate_branches and deactivate_gens methods"""
        # Load the network
        network = load_net_from_pglib("case24_ieee_rts")

        # Test deactivating a generator
        network.deactivate_gens(np.array([0]))
        assert network.gens[0, 7] == 0, "Generator should be deactivated"

        # Test deactivating a branch
        network.deactivate_branches(np.array([0]))
        assert network.branches[0, 10] == 0, "Branch should be deactivated"

        # Test that idx_gens_in_service and idx_branches_in_service reflect the changes
        assert 0 not in network.idx_gens_in_service, (
            "Deactivated generator should not be in service"
        )
        assert 0 not in network.idx_branches_in_service, (
            "Deactivated branch should not be in service"
        )

    def test_network_to_mpc_method(self):
        """Test the to_mpc method for saving to file"""
        import tempfile
        import os

        # Load the network
        network = load_net_from_pglib("case24_ieee_rts")

        # Test saving to a temporary file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".m",
            delete=False,
        ) as tmp_file:
            tmp_filename = tmp_file.name

        try:
            network.to_mpc(tmp_filename)

            # Verify file was created and has content
            assert os.path.exists(tmp_filename), "MPC file should be created"
            assert os.path.getsize(tmp_filename) > 0, "MPC file should not be empty"

            # Read and verify basic content
            with open(tmp_filename, "r") as f:
                content = f.read()
                assert "function mpc = case_from_dict" in content, (
                    "Should contain function definition"
                )
                assert "mpc.version" in content, "Should contain version"
                assert "mpc.baseMVA" in content, "Should contain baseMVA"
                assert "mpc.bus" in content, "Should contain bus matrix"
                assert "mpc.gen" in content, "Should contain gen matrix"
                assert "mpc.branch" in content, "Should contain branch matrix"
                assert "mpc.gencost" in content, "Should contain gencost matrix"

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)
