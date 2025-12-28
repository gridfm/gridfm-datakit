"""
Test cases for topology perturbation functionality
Tests that perturbations don't modify the original network and work correctly
"""

import pytest
import numpy as np
from gridfm_datakit.network import load_net_from_pglib
from gridfm_datakit.perturbations.topology_perturbation import (
    NoPerturbationGenerator,
    NMinusKGenerator,
    RandomComponentDropGenerator,
)
from gridfm_datakit.utils.idx_brch import F_BUS, T_BUS


class TestTopologyPerturbation:
    """Test class for topology perturbation functionality"""

    def test_no_perturbation_generator(self):
        """Test that NoPerturbationGenerator returns the original network"""
        # Load the original network
        original_network = load_net_from_pglib("case24_ieee_rts")

        # Store original state
        original_branch_status = original_network.branches[
            :,
            10,
        ].copy()  # BR_STATUS column
        original_gen_status = original_network.gens[:, 7].copy()  # GEN_STATUS column

        # Create no perturbation generator
        no_perturbation_generator = NoPerturbationGenerator()

        # Generate perturbed topologies (should just return the original)
        perturbed_networks = list(no_perturbation_generator.generate(original_network))

        # Verify original network is unchanged
        np.testing.assert_array_equal(
            original_network.branches[:, 10],
            original_branch_status,
            "Original network branch status should be unchanged",
        )
        np.testing.assert_array_equal(
            original_network.gens[:, 7],
            original_gen_status,
            "Original network generator status should be unchanged",
        )

        # Verify we got exactly one network (the original)
        assert len(perturbed_networks) == 1, (
            f"Expected 1 network, got {len(perturbed_networks)}"
        )
        assert perturbed_networks[0] == original_network, (
            "Should return the original network object"
        )

    def test_random_component_drop_generator(self):
        """Test that RandomComponentDropGenerator creates different network states"""
        # Load the original network
        original_network = load_net_from_pglib("case24_ieee_rts")

        # Create random component drop generator
        random_generator = RandomComponentDropGenerator(
            n_topology_variants=5,
            k=2,
            base_net=original_network,
            elements=["branch", "gen"],
        )

        # Generate perturbed topologies
        perturbed_networks = list(random_generator.generate(original_network))

        # Store the states of all networks
        network_states = []
        for network in perturbed_networks:
            state = {
                "branch_status": network.branches[:, 10].copy(),  # BR_STATUS
                "gen_status": network.gens[:, 7].copy(),  # GEN_STATUS
            }
            network_states.append(state)

        # Verify that at least some networks are different
        # (It's possible but unlikely that all random perturbations are identical)
        different_networks = 0
        for i in range(len(network_states)):
            for j in range(i + 1, len(network_states)):
                if not np.array_equal(
                    network_states[i]["branch_status"],
                    network_states[j]["branch_status"],
                ) or not np.array_equal(
                    network_states[i]["gen_status"],
                    network_states[j]["gen_status"],
                ):
                    different_networks += 1

        # We expect at least some networks to be different
        assert different_networks > 0, (
            "All perturbed networks are identical, which is unlikely"
        )

    def test_n_minus_k_generator_connectivity(self):
        """Test that NMinusKGenerator respects connectivity constraints"""
        # Load the original network
        original_network = load_net_from_pglib("case24_ieee_rts")

        # Verify original network is connected
        assert original_network.check_single_connected_component(), (
            "Original network should be connected"
        )

        # Create N-k generator
        n_minus_k_generator = NMinusKGenerator(
            k=1,  # Only drop 1 component at a time
            base_net=original_network,
        )

        # Generate perturbed topologies
        perturbed_networks = list(n_minus_k_generator.generate(original_network))

        # Verify all perturbed networks are still connected
        for i, network in enumerate(perturbed_networks):
            assert network.check_single_connected_component(), (
                f"Perturbed network {i} should be connected"
            )

    def test_topology_perturbation_branch_deactivation(self):
        """Test that branch deactivation works correctly"""
        test_network = load_net_from_pglib("case24_ieee_rts")

        # Get original branch status
        original_branch_status = test_network.branches[:, 10].copy()

        # Deactivate some branches
        branches_to_deactivate = np.array([0, 1, 2])
        test_network.deactivate_branches(branches_to_deactivate)

        # Verify branches are deactivated
        assert np.all(test_network.branches[branches_to_deactivate, 10] == 0), (
            "Branches should be deactivated"
        )

        # Verify other branches are unchanged
        other_branches = np.setdiff1d(
            np.arange(len(test_network.branches)),
            branches_to_deactivate,
        )
        np.testing.assert_array_equal(
            test_network.branches[other_branches, 10],
            original_branch_status[other_branches],
            "Other branches should be unchanged",
        )

    def test_topology_perturbation_generator_deactivation(self):
        """Test that generator deactivation works correctly"""
        test_network = load_net_from_pglib("case24_ieee_rts")

        # Get original generator status
        original_gen_status = test_network.gens[:, 7].copy()

        # Deactivate some generators
        gens_to_deactivate = np.array([0, 1])
        test_network.deactivate_gens(gens_to_deactivate)

        # Verify generators are deactivated
        assert np.all(test_network.gens[gens_to_deactivate, 7] == 0), (
            "Generators should be deactivated"
        )

        # Verify other generators are unchanged
        other_gens = np.setdiff1d(np.arange(len(test_network.gens)), gens_to_deactivate)
        np.testing.assert_array_equal(
            test_network.gens[other_gens, 7],
            original_gen_status[other_gens],
            "Other generators should be unchanged",
        )

    def test_n_minus_k_generator_invalid_parameters(self):
        """Test that NMinusKGenerator handles invalid parameters gracefully"""
        # Load the original network
        original_network = load_net_from_pglib("case24_ieee_rts")

        # Test with k=0 (should raise ValueError)
        with pytest.raises(ValueError, match="k must be greater than 0"):
            NMinusKGenerator(k=0, base_net=original_network)

        # Test with k=1 (should work)
        n_minus_k_generator = NMinusKGenerator(k=1, base_net=original_network)
        perturbed_networks = list(n_minus_k_generator.generate(original_network))
        assert len(perturbed_networks) > 0, (
            "Should generate at least one network with k=1"
        )

    def test_random_component_drop_different_elements(self):
        """Test RandomComponentDropGenerator with different element types"""
        # Load the original network
        original_network = load_net_from_pglib("case24_ieee_rts")

        # Test with different element combinations
        element_combinations = [["branch"], ["gen"], ["branch", "gen"]]

        for elements in element_combinations:
            random_generator = RandomComponentDropGenerator(
                n_topology_variants=2,
                k=1,
                base_net=original_network,
                elements=elements,
            )

            # This should not raise an exception
            perturbed_networks = list(random_generator.generate(original_network))
            assert len(perturbed_networks) <= 2, (
                f"Should generate at most 2 variants for elements {elements}"
            )

    def test_network_copy_independence(self):
        """Test that Network objects are independent after copying"""
        # Load the original network
        original_network = load_net_from_pglib("case24_ieee_rts")

        # Create a copy (this should be done in the topology generator)
        import copy

        copied_network = copy.deepcopy(original_network)

        # Modify the copied network
        copied_network.deactivate_branches([0, 1])
        copied_network.deactivate_gens([0])

        # Verify original is unchanged
        assert np.all(original_network.branches[:, 10] == 1), (
            "Original network branches should be in service"
        )
        assert np.all(original_network.gens[:, 7] == 1), (
            "Original network generators should be in service"
        )

        # Verify copy is modified
        assert copied_network.branches[0, 10] == 0, (
            "Copied network branch 0 should be out of service"
        )
        assert copied_network.branches[1, 10] == 0, (
            "Copied network branch 1 should be out of service"
        )
        assert copied_network.gens[0, 7] == 0, (
            "Copied network generator 0 should be out of service"
        )

    def test_n_minus_k_generator_comprehensive(self):
        """Test NMinusKGenerator comprehensively"""
        # Load the original network
        original_network = load_net_from_pglib("case24_ieee_rts")

        # Test with k=1
        n_minus_k_generator = NMinusKGenerator(k=1, base_net=original_network)

        # Check that it has the right number of combinations
        expected_combinations = (
            len(original_network.idx_branches_in_service) + 1
        )  # +1 for dropping 0 components
        assert len(n_minus_k_generator.component_combinations) == expected_combinations

        # Generate all topologies
        perturbed_networks = list(n_minus_k_generator.generate(original_network))

        # Should have at least one topology (the original with no components dropped)
        assert len(perturbed_networks) >= 1, "Should generate at least one topology"

        # All generated topologies should be connected
        for network in perturbed_networks:
            assert network.check_single_connected_component(), (
                "All generated topologies should be connected"
            )

    def test_random_component_drop_generator_comprehensive(self):
        """Test RandomComponentDropGenerator comprehensively"""
        # Load the original network
        original_network = load_net_from_pglib("case24_ieee_rts")

        # Test with branches only
        random_generator = RandomComponentDropGenerator(
            n_topology_variants=3,
            k=1,
            base_net=original_network,
            elements=["branch"],
        )

        # Check that components_to_drop contains only branches
        for idx, element_type in random_generator.components_to_drop:
            assert element_type == "branch", "Should only contain branches"
            assert idx in original_network.idx_branches_in_service, (
                "Should only contain in-service branches"
            )

        # Generate topologies
        perturbed_networks = list(random_generator.generate(original_network))

        # Should generate exactly the requested number of variants
        assert len(perturbed_networks) == 3, "Should generate exactly 3 variants"

        # All generated topologies should be connected
        for network in perturbed_networks:
            assert network.check_single_connected_component(), (
                "All generated topologies should be connected"
            )

    def test_topology_generator_preserves_original(self):
        """Test that topology generators preserve the original network"""
        # Load the original network
        original_network = load_net_from_pglib("case24_ieee_rts")

        # Store original state
        original_branch_status = original_network.branches[:, 10].copy()
        original_gen_status = original_network.gens[:, 7].copy()

        # Test with NMinusKGenerator
        n_minus_k_generator = NMinusKGenerator(k=1, base_net=original_network)
        list(n_minus_k_generator.generate(original_network))

        # Verify original is unchanged
        np.testing.assert_array_equal(
            original_network.branches[:, 10],
            original_branch_status,
            "Original network should be unchanged after NMinusKGenerator",
        )
        np.testing.assert_array_equal(
            original_network.gens[:, 7],
            original_gen_status,
            "Original network should be unchanged after NMinusKGenerator",
        )

        # Test with RandomComponentDropGenerator
        random_generator = RandomComponentDropGenerator(
            n_topology_variants=2,
            k=1,
            base_net=original_network,
            elements=["branch"],
        )
        list(random_generator.generate(original_network))

        # Verify original is unchanged
        np.testing.assert_array_equal(
            original_network.branches[:, 10],
            original_branch_status,
            "Original network should be unchanged after RandomComponentDropGenerator",
        )
        np.testing.assert_array_equal(
            original_network.gens[:, 7],
            original_gen_status,
            "Original network should be unchanged after RandomComponentDropGenerator",
        )

    def test_disconnect_bus6_and_check_connectivity(self):
        """Test that disconnecting all branches from/to bus 6 creates disconnected components"""
        # Load the original network
        original_network = load_net_from_pglib("case24_ieee_rts")

        # Verify original network is connected
        assert original_network.check_single_connected_component(), (
            "Original network should be connected"
        )

        # Create a copy for testing
        test_network = load_net_from_pglib("case24_ieee_rts")

        # Find all branches connected to bus 6 (0-indexed, so bus 6 is index 5)
        target_bus = 5  # Bus 6 in 0-indexed system

        # Find branches where either from_bus or to_bus is the target bus
        branches_to_bus6 = []
        for branch_idx in test_network.idx_branches_in_service:
            from_bus = int(test_network.branches[branch_idx, F_BUS])  # F_BUS
            to_bus = int(test_network.branches[branch_idx, T_BUS])  # T_BUS

            if from_bus == target_bus or to_bus == target_bus:
                branches_to_bus6.append(branch_idx)

        # Verify we found some branches connected to bus 6
        assert len(branches_to_bus6) > 0, (
            f"Should find branches connected to bus 6, found {len(branches_to_bus6)}"
        )

        # Deactivate all branches connected to bus 6
        test_network.deactivate_branches(np.array(branches_to_bus6))

        # Verify branches are deactivated
        assert np.all(test_network.branches[branches_to_bus6, 10] == 0), (
            "All branches to bus 6 should be deactivated"
        )

        # Check connectivity - should now be disconnected
        is_connected = test_network.check_single_connected_component()
        assert not is_connected, (
            "Network should be disconnected after removing all branches to bus 6"
        )

        # Verify the original network is still connected (test didn't modify it)
        assert original_network.check_single_connected_component(), (
            "Original network should still be connected"
        )
