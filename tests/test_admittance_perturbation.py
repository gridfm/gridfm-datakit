"""
Minimal tests for admittance perturbation functionality
Tests that no perturbation preserves admittance and perturbation changes it
"""

import numpy as np
import copy
from gridfm_datakit.network import load_net_from_pglib
from gridfm_datakit.perturbations.admittance_perturbation import (
    NoAdmittancePerturbationGenerator,
    PerturbAdmittanceGenerator,
)
from gridfm_datakit.utils.idx_brch import BR_R, BR_X


class TestAdmittancePerturbation:
    """Test class for admittance perturbation functionality"""

    def test_no_admittance_perturbation_preserves_values(self):
        """Test that NoAdmittancePerturbationGenerator preserves admittance values"""
        # Load the original network
        original_network = load_net_from_pglib("case24_ieee_rts")

        # Store original admittance values
        original_r = original_network.branches[:, BR_R].copy()
        original_x = original_network.branches[:, BR_X].copy()

        # Create no perturbation generator
        no_perturbation_generator = NoAdmittancePerturbationGenerator()

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

        # Verify admittance values are unchanged
        np.testing.assert_array_equal(
            perturbed_networks[0].branches[:, BR_R],
            original_r,
            "R values should be unchanged with no perturbation",
        )
        np.testing.assert_array_equal(
            perturbed_networks[0].branches[:, BR_X],
            original_x,
            "X values should be unchanged with no perturbation",
        )

    def test_admittance_perturbation_changes_values(self):
        """Test that PerturbAdmittanceGenerator changes admittance values"""
        # Load the original network
        original_network = load_net_from_pglib("case24_ieee_rts")

        # Store original admittance values
        original_r = original_network.branches[:, BR_R].copy()
        original_x = original_network.branches[:, BR_X].copy()

        # Create perturbation generator with sigma=0.1
        perturb_generator = PerturbAdmittanceGenerator(
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
        perturbed_r = perturbed_network.branches[:, BR_R]
        perturbed_x = perturbed_network.branches[:, BR_X]

        # Check that admittance values are actually different
        # This might not always be true due to randomness, so we'll check multiple times
        different_perturbations = 0
        for _ in range(10):  # Try multiple perturbations
            test_network = copy.deepcopy(original_network)

            def example_gen():
                yield test_network

            perturbed_networks = list(perturb_generator.generate(example_gen()))
            perturbed_r = perturbed_networks[0].branches[:, BR_R]
            perturbed_x = perturbed_networks[0].branches[:, BR_X]

            if not np.array_equal(original_r, perturbed_r) or not np.array_equal(
                original_x,
                perturbed_x,
            ):
                different_perturbations += 1

        # We expect at least some perturbations to be different
        assert different_perturbations > 0, (
            "Perturbation should change the admittance values"
        )

    def test_admittance_sigma_zero_no_change(self):
        """PerturbAdmittanceGenerator with sigma=0 should not change R/X."""
        net = load_net_from_pglib("case24_ieee_rts")
        r0 = net.branches[:, BR_R].copy()
        x0 = net.branches[:, BR_X].copy()
        gen = PerturbAdmittanceGenerator(base_net=net, sigma=0.0)

        def gen_net():
            yield net

        [net_out] = list(gen.generate(gen_net()))
        np.testing.assert_array_equal(net_out.branches[:, BR_R], r0)
        np.testing.assert_array_equal(net_out.branches[:, BR_X], x0)


if __name__ == "__main__":
    test = TestAdmittancePerturbation()
    test.test_no_admittance_perturbation_preserves_values()
    test.test_admittance_perturbation_changes_values()
    test.test_admittance_sigma_zero_no_change()
