import numpy as np
from abc import ABC, abstractmethod
from typing import Generator, List, Union
from gridfm_datakit.network import Network
from gridfm_datakit.utils.idx_cost import NCOST, COST


class GenerationGenerator(ABC):
    """Abstract base class for applying perturbations to generator elements
    in a network."""

    def __init__(self) -> None:
        """Initialize the generation generator."""
        pass

    @abstractmethod
    def generate(
        self,
        example_generator: Generator[Network, None, None],
    ) -> Union[Generator[Network, None, None], List[Network]]:
        """Generate generation perturbations.

        Args:
            example_generator: A generator producing example (load/topology/generation)
                scenarios to which generator cost perturbations are added.

        Yields:
            A generation-perturbed scenario.
        """
        pass


class NoGenPerturbationGenerator(GenerationGenerator):
    """Generator that yields the original network generator without any perturbations."""

    def __init__(self):
        """Initialize the no-perturbation generator"""
        pass

    def generate(
        self,
        example_generator: Generator[Network, None, None],
    ) -> Generator[Network, None, None]:
        """Yield the original examples without any perturbations.

        Args:
            example_generator: A generator producing example (load/topology/generation)
                scenarios to which generator cost perturbations should be applied.

        Yields:
            The original example produced by the example_generator.
        """
        for example in example_generator:
            yield example


class PermuteGenCostGenerator(GenerationGenerator):
    """Class for permuting generator costs.

    This class is for generating different generation scenarios
    by permuting cost coefficients between and among generators
    of power grid networks. Only generators with non-zero linear (c1)
    or quadratic (c2) cost coefficients are permuted, preserving
    zero-cost and constant-only generators.
    """

    def __init__(self, base_net: Network) -> None:
        """
        Initialize the gen-cost permuation generator.

        Args:
            base_net: The base power network.
        """
        self.base_net = base_net
        self.num_gens = base_net.gens.shape[0]  # acount for deactivated generators
        assert np.all(base_net.gencosts[:, NCOST] == base_net.gencosts[:, NCOST][0]), (
            "All generators must have the same number of cost coefficients"
        )

        # Identify generators to permute (skip zero-cost and constant-only cost)
        costs = base_net.gencosts[
            :,
            COST:,
        ]  # all cost coefficients [c2, c1, c0] for NCOST=3
        # Keep only generators with non-constant terms (c1, c2, ...) non-zero
        # costs[:, :-1] excludes the last coefficient (c0, the constant term)
        self.permutable_mask = np.any(costs[:, :-1] != 0, axis=1)
        self.permutable_indices = np.where(self.permutable_mask)[0]

    def generate(
        self,
        example_generator: Generator[Network, None, None],
    ) -> Generator[Network, None, None]:
        """Generate a network with permuted generator cost coefficients.

        Args:
            example_generator: A generator producing example
                (load/topology/generation) scenarios to which generator
                cost coefficient permutations should be applied.

        Yields:
            An example scenario with cost coeffiecients in the
            poly_cost table permuted (only for dispatchable generators)
        """
        for scenario in example_generator:
            # Only permute the selected generators
            new_idx = np.random.permutation(self.permutable_indices)
            # Permute the rows (generators) of the cost coefficients (and NCOST, although we assume it is the same for all generators)
            scenario.gencosts[self.permutable_indices, NCOST:] = scenario.gencosts[
                new_idx,
                NCOST:,
            ]
            yield scenario


class PerturbGenCostGenerator(GenerationGenerator):
    """Class for perturbing generator cost.

    This class is for generating different generation scenarios
    by randomly perturbing cost coefficients of generators in a
    power network by multiplying with a scaling factor sampled
    from a uniform distribution. Only generators with non-zero
    linear (c1) or quadratic (c2) cost coefficients are perturbed,
    preserving zero-cost and constant-only generators.
    """

    def __init__(self, base_net: Network, sigma: float) -> None:
        """
        Initialize the gen-cost perturbation generator.

        Args:
            base_net: The base power network.
            sigma: A constant that specifies the range from which to draw
                samples from a uniform distribution to be used as a scaling
                factor for cost coefficient perturbations. The range is
                set as [max([0, 1-sigma]), 1+sigma).
        """
        self.base_net = base_net
        self.num_gens = base_net.gens.shape[0]  # acount for deactivated generators
        # assert all generators have the same number of cost coefficients
        assert np.all(base_net.gencosts[:, NCOST] == base_net.gencosts[:, NCOST][0]), (
            "All generators must have the same number of cost coefficients"
        )
        n_costs = int(base_net.gencosts[:, NCOST][0])
        self.lower = np.max([0.0, 1.0 - sigma])
        self.upper = 1.0 + sigma

        # Mask generators to perturb (skip zero-cost or constant-only)
        costs = base_net.gencosts[
            :,
            COST:,
        ]  # all cost coefficients [c2, c1, c0] for NCOST=3
        # Keep only generators with non-constant terms (c1, c2, ...) non-zero
        # costs[:, :-1] excludes the last coefficient (c0, the constant term)
        self.perturb_mask = np.any(costs[:, :-1] != 0, axis=1)
        self.n_perturbable = np.sum(self.perturb_mask)

        # Sample size for only the perturbable generators
        self.sample_size = [self.n_perturbable, n_costs]

    def generate(
        self,
        example_generator: Generator[Network, None, None],
    ) -> Generator[Network, None, None]:
        """Generate a network with perturbed generator cost coefficients.

        Args:
            example_generator: A generator producing example (load/topology) scenarios
                to which generator cost coefficient perturbations should be added.

        Yields:
            An example scenario with cost coeffiecients in the poly_cost
            table perturbed by multiplying with a scaling factor (only for dispatchable generators).
        """
        for example in example_generator:
            # Generate scaling factors only for generators that can be perturbed
            scale_fact = np.random.uniform(
                low=self.lower,
                high=self.upper,
                size=self.sample_size,
            )
            # Apply scaling only to selected generators
            example.gencosts[self.perturb_mask, COST:] = (
                example.gencosts[self.perturb_mask, COST:] * scale_fact
            )
            yield example
