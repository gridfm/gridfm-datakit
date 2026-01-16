"""
Minimal tests for generator cost perturbation functionality
Tests that no perturbation preserves costs and perturbation changes them
"""

import numpy as np
import copy
from gridfm_datakit.network import load_net_from_pglib
from gridfm_datakit.perturbations.generator_perturbation import (
    NoGenPerturbationGenerator,
    PerturbGenCostGenerator,
    PermuteGenCostGenerator,
)
from gridfm_datakit.utils.idx_cost import STARTUP, SHUTDOWN, NCOST, COST


class TestGeneratorPerturbation:
    """Test class for generator cost perturbation functionality"""

    def test_no_generator_perturbation_preserves_values(self):
        """Test that NoGenPerturbationGenerator preserves cost values"""
        # Load the original network
        original_network = load_net_from_pglib("case24_ieee_rts")

        # Store original cost values
        original_gencosts = original_network.gencosts.copy()

        # Create no perturbation generator
        no_perturbation_generator = NoGenPerturbationGenerator()

        # Create a simple generator that yields the original network
        def example_generator():
            yield original_network

        # Generate perturbed networks (should just return the original)
        perturbed_networks = list(
            no_perturbation_generator.generate(example_generator()),
        )

        # Verify we got exactly one network (the original)
        assert len(perturbed_networks) == 1, (
            f"Expected 1 network, got {len(perturbed_networks)}"
        )

        # Verify cost values are unchanged
        np.testing.assert_array_equal(
            perturbed_networks[0].gencosts,
            original_gencosts,
            "Generator cost values should be unchanged with no perturbation",
        )

    def test_generator_cost_perturbation_changes_values(self):
        """Test that PerturbGenCostGenerator changes ALL cost coefficients for non-constant generators"""
        # Load the original network
        original_network = load_net_from_pglib("case24_ieee_rts")

        # Store original cost values
        original_gencosts = original_network.gencosts.copy()

        # Identify generators with non-constant costs (c1 or c2 != 0)
        # MATPOWER cost order: [c2, c1, c0] for NCOST=3
        costs = original_gencosts[:, COST:]
        non_constant_mask = np.any(costs[:, :-1] != 0, axis=1)

        # Ensure there are actually non-constant generators in the test
        assert np.sum(non_constant_mask) > 0, (
            "Test requires generators with non-constant costs"
        )

        # Create perturbation generator with significant sigma
        perturb_generator = PerturbGenCostGenerator(
            base_net=original_network,
            sigma=0.1,
        )

        # Generate one perturbation
        test_network = copy.deepcopy(original_network)

        def example_gen():
            yield test_network

        perturbed_networks = list(perturb_generator.generate(example_gen()))
        perturbed_gencosts = perturbed_networks[0].gencosts

        # Check that ALL cost coefficients have changed for each non-constant generator
        for gen_idx in np.where(non_constant_mask)[0]:
            original_costs = original_gencosts[gen_idx, COST:]
            perturbed_costs = perturbed_gencosts[gen_idx, COST:]

            # Check each coefficient individually (c2, c1, c0)
            for coeff_idx in range(len(original_costs)):
                if original_costs[coeff_idx] != 0:
                    assert original_costs[coeff_idx] != perturbed_costs[coeff_idx], (
                        f"Generator {gen_idx}, coefficient {coeff_idx}: "
                        f"Expected value to change from {original_costs[coeff_idx]}, "
                        f"but it remained the same. All non-constant generators should have "
                        f"all their cost coefficients changed by perturbation."
                    )

    def test_generator_cost_perturbation_preserves_structure(self):
        """Test that PerturbGenCostGenerator preserves the structure of cost coefficients"""
        # Load the original network
        original_network = load_net_from_pglib("case24_ieee_rts")

        # Store original structure
        original_ncost = original_network.gencosts[:, NCOST].copy()
        original_startup = original_network.gencosts[:, STARTUP].copy()
        original_shutdown = original_network.gencosts[:, SHUTDOWN].copy()

        # Create perturbation generator
        perturb_generator = PerturbGenCostGenerator(
            base_net=original_network,
            sigma=0.1,
        )

        # Create a simple generator that yields a copy of the original network
        def example_generator():
            yield copy.deepcopy(original_network)

        # Generate perturbed networks
        perturbed_networks = list(perturb_generator.generate(example_generator()))

        # Verify we got exactly one network
        assert len(perturbed_networks) == 1, (
            f"Expected 1 network, got {len(perturbed_networks)}"
        )

        perturbed_network = perturbed_networks[0]

        # Verify structural elements are preserved
        np.testing.assert_array_equal(
            perturbed_network.gencosts[:, NCOST],
            original_ncost,
            "NCOST values should be unchanged",
        )
        np.testing.assert_array_equal(
            perturbed_network.gencosts[:, STARTUP],
            original_startup,
            "STARTUP values should be unchanged",
        )
        np.testing.assert_array_equal(
            perturbed_network.gencosts[:, SHUTDOWN],
            original_shutdown,
            "SHUTDOWN values should be unchanged",
        )

    def test_generator_cost_permutation_changes_order(self):
        """Test that PermuteGenCostGenerator changes the order of cost coefficients"""
        # Load the original network
        original_network = load_net_from_pglib("case24_ieee_rts")

        # Store original cost values
        original_gencosts = original_network.gencosts.copy()

        # Create permutation generator
        permute_generator = PermuteGenCostGenerator(base_net=original_network)

        # Create a simple generator that yields a copy of the original network
        def example_generator():
            yield copy.deepcopy(original_network)

        # Generate permuted networks
        permuted_networks = list(permute_generator.generate(example_generator()))

        # Verify we got exactly one network
        assert len(permuted_networks) == 1, (
            f"Expected 1 network, got {len(permuted_networks)}"
        )

        permuted_network = permuted_networks[0]
        permuted_gencosts = permuted_network.gencosts

        # Check that cost values are actually different
        # This might not always be true due to randomness, so we'll check multiple times
        different_permutations = 0
        for _ in range(10):  # Try multiple permutations
            test_network = copy.deepcopy(original_network)

            def example_gen():
                yield test_network

            permuted_networks = list(permute_generator.generate(example_gen()))
            permuted_gencosts = permuted_networks[0].gencosts

            if not np.array_equal(original_gencosts, permuted_gencosts):
                different_permutations += 1

        # We expect at least some permutations to be different
        assert different_permutations > 0, (
            "Permutation should change the order of generator cost values"
        )

    def test_generator_cost_permutation_preserves_structure(self):
        """Test that PermuteGenCostGenerator preserves the structure of cost coefficients"""
        # Load the original network
        original_network = load_net_from_pglib("case24_ieee_rts")

        # Store original structure
        original_ncost = original_network.gencosts[:, NCOST].copy()
        original_startup = original_network.gencosts[:, STARTUP].copy()
        original_shutdown = original_network.gencosts[:, SHUTDOWN].copy()

        # Create permutation generator
        permute_generator = PermuteGenCostGenerator(base_net=original_network)

        # Create a simple generator that yields a copy of the original network
        def example_generator():
            yield copy.deepcopy(original_network)

        # Generate permuted networks
        permuted_networks = list(permute_generator.generate(example_generator()))

        # Verify we got exactly one network
        assert len(permuted_networks) == 1, (
            f"Expected 1 network, got {len(permuted_networks)}"
        )

        permuted_network = permuted_networks[0]

        # Verify structural elements are preserved
        np.testing.assert_array_equal(
            permuted_network.gencosts[:, NCOST],
            original_ncost,
            "NCOST values should be unchanged",
        )
        np.testing.assert_array_equal(
            permuted_network.gencosts[:, STARTUP],
            original_startup,
            "STARTUP values should be unchanged",
        )
        np.testing.assert_array_equal(
            permuted_network.gencosts[:, SHUTDOWN],
            original_shutdown,
            "SHUTDOWN values should be unchanged",
        )

    def test_generator_cost_perturbation_scaling_range(self):
        """Test that PerturbGenCostGenerator respects the scaling range"""
        # Load the original network
        original_network = load_net_from_pglib("case24_ieee_rts")

        # Create perturbation generator with sigma=0.2
        sigma = 0.2
        perturb_generator = PerturbGenCostGenerator(
            base_net=original_network,
            sigma=sigma,
        )

        # Expected scaling range
        expected_lower = max(0.0, 1.0 - sigma)  # 0.8
        expected_upper = 1.0 + sigma  # 1.2

        # Create a simple generator that yields a copy of the original network
        def example_generator():
            yield copy.deepcopy(original_network)

        # Generate multiple perturbed networks to test scaling range
        for _ in range(20):  # Test multiple perturbations
            test_network = copy.deepcopy(original_network)
            original_costs = test_network.gencosts[:, COST:].copy()

            def example_gen():
                yield test_network

            perturbed_networks = list(perturb_generator.generate(example_gen()))
            perturbed_costs = perturbed_networks[0].gencosts[:, COST:]

            # Calculate scaling factors
            # check everything is not zero
            assert np.any(original_costs != 0), "Original costs should not be zero"
            assert np.any(perturbed_costs != 0), "Perturbed costs should not be zero"
            scaling_factors = perturbed_costs / original_costs

            # Filter out NaN values (where original costs are zero)
            valid_scaling_factors = scaling_factors[~np.isnan(scaling_factors)]

            # Only check if there are valid scaling factors
            if len(valid_scaling_factors) > 0:
                # Check that scaling factors are within expected range
                assert np.all(valid_scaling_factors >= expected_lower), (
                    f"Scaling factors should be >= {expected_lower}"
                )
                assert np.all(valid_scaling_factors <= expected_upper), (
                    f"Scaling factors should be <= {expected_upper}"
                )


def test_generator_cost_sigma_zero_no_change():
    """PerturbGenCostGenerator with sigma=0 should not change gencosts."""
    net = load_net_from_pglib("case24_ieee_rts")
    costs0 = net.gencosts.copy()
    gen = PerturbGenCostGenerator(base_net=net, sigma=0.0)

    def gen_net():
        import copy

        yield copy.deepcopy(net)

    [net_out] = list(gen.generate(gen_net()))
    assert np.allclose(net_out.gencosts, costs0, rtol=0.0, atol=0.0), (
        "gencosts should be unchanged when sigma=0"
    )


def test_permute_skips_zero_and_constant_cost_generators():
    """PermuteGenCostGenerator should skip generators with zero or constant-only costs."""
    net = load_net_from_pglib("case24_ieee_rts")

    # Modify some generators to have zero or constant-only costs
    # MATPOWER cost order: [c2, c1, c0] for NCOST=3

    # Generator 0: zero cost (all coefficients = 0)
    net.gencosts[0, COST:] = 0

    # Generator 1: constant only (c2=0, c1=0, c0=100)
    net.gencosts[1, COST : COST + 2] = 0  # Set c2 and c1 to 0
    net.gencosts[1, COST + 2] = 100  # Set c0 to 100

    # Save original costs for these generators
    original_zero_cost = net.gencosts[0, COST:].copy()
    original_constant_cost = net.gencosts[1, COST:].copy()

    # Create permutation generator
    permute_gen = PermuteGenCostGenerator(base_net=net)

    # Test multiple permutations
    for _ in range(10):
        test_net = copy.deepcopy(net)

        def gen_net():
            yield test_net

        [perturbed_net] = list(permute_gen.generate(gen_net()))

        # Verify zero-cost generator is unchanged
        np.testing.assert_array_equal(
            perturbed_net.gencosts[0, COST:],
            original_zero_cost,
            "Zero-cost generator should not be permuted",
        )

        # Verify constant-only generator is unchanged
        np.testing.assert_array_equal(
            perturbed_net.gencosts[1, COST:],
            original_constant_cost,
            "Constant-only generator should not be permuted",
        )


def test_perturb_skips_zero_and_constant_cost_generators():
    """PerturbGenCostGenerator should skip generators with zero or constant-only costs."""
    net = load_net_from_pglib("case24_ieee_rts")

    # Modify some generators to have zero or constant-only costs
    # MATPOWER cost order: [c2, c1, c0] for NCOST=3

    # Generator 0: zero cost (all coefficients = 0)
    net.gencosts[0, COST:] = 0

    # Generator 1: constant only (c2=0, c1=0, c0=100)
    net.gencosts[1, COST : COST + 2] = 0  # Set c2 and c1 to 0
    net.gencosts[1, COST + 2] = 100  # Set c0 to 100

    # Save original costs for these generators
    original_zero_cost = net.gencosts[0, COST:].copy()
    original_constant_cost = net.gencosts[1, COST:].copy()

    # Create perturbation generator with significant sigma
    perturb_gen = PerturbGenCostGenerator(base_net=net, sigma=0.5)

    # Test multiple perturbations
    for _ in range(10):
        test_net = copy.deepcopy(net)

        def gen_net():
            yield test_net

        [perturbed_net] = list(perturb_gen.generate(gen_net()))

        # Verify zero-cost generator is unchanged
        np.testing.assert_array_equal(
            perturbed_net.gencosts[0, COST:],
            original_zero_cost,
            "Zero-cost generator should not be perturbed",
        )

        # Verify constant-only generator is unchanged
        np.testing.assert_array_equal(
            perturbed_net.gencosts[1, COST:],
            original_constant_cost,
            "Constant-only generator should not be perturbed",
        )
