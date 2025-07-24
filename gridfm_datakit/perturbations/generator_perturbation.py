import numpy as np
import pandapower as pp
from abc import ABC, abstractmethod
from typing import Generator, List, Union


class GenerationGenerator(ABC):
    """Abstract base class for applying perturbations to generator elements
    in a network."""

    def __init__(self) -> None:
        """Initialize the generation generator."""
        pass

    @abstractmethod
    def generate(
        self,
        net_topologies: Generator[pp.pandapowerNet, None, None],
    ) -> Union[Generator[pp.pandapowerNet, None, None], List[pp.pandapowerNet]]:
        """Generate generation perturbations.

        Args:
            net: The power network to perturb.

        Yields:
            A perturbed generation scenario.
        """
        pass


class NoGenPerturbationGenerator(GenerationGenerator):
    """Generator that yields the original network without any perturbations."""

    def __init__(self):
        pass

    def generate(
        self,
        net_topologies: Generator[pp.pandapowerNet, None, None],
    ) -> Generator[pp.pandapowerNet, None, None]:
        """Yield the original network without any perturbations.

        Args:
            net: The power network.

        Yields:
            The original power network.
        """
        for net_top in net_topologies:
            yield net_top
        # return net_topologies


class PermuteGenCostGenerator(GenerationGenerator):
    """Class for permuting generator costs.

    This class is for generating different scenarios by permuting
    all the coeffiecient costs between and among generators of
    power grid networks.
    """

    def __init__(self, base_net):
        self.base_net = base_net
        self.num_gens = len(base_net.poly_cost)
        self.permute_cols = self.base_net.poly_cost.columns[2:]
        # net.poly_cost[column_names]

    def generate(
        self,
        net_topologies: Generator[pp.pandapowerNet, None, None],
    ) -> Generator[pp.pandapowerNet, None, None]:
        """Generate network with permuted generator cost coefficients.

        Args:
            net: The power network.

        Yields:
            The same power network, but with cost coeffiecients in the
            poly_cost table permuted
        """
        for net_top in net_topologies:
            new_idx = np.random.permutation(self.num_gens)
            net_top.poly_cost[self.permute_cols] = (
                net_top.poly_cost[self.permute_cols]
                .iloc[new_idx]
                .reset_index(drop=True)
            )
            yield net_top


class PerturbGenCostGenerator(GenerationGenerator):
    """Class for perturbing generator cost.

    This class is for generating different generation scenarios
    by randomly perturbing all the cost coeffiecient of generators in
    a power network by multiplying with a scaling factor, sampled from
    a uniform distribution.
    """

    def __init__(self, base_net, sigma):
        self.base_net = base_net
        self.num_gens = len(base_net.poly_cost)
        self.perturb_cols = self.base_net.poly_cost.columns[2:]
        self.lower = 1 - sigma
        self.upper = 1 + sigma
        self.sample_size = [self.num_gens, len(self.perturb_cols)]

    def generate(
        self,
        net_topologies: Generator[pp.pandapowerNet, None, None],
    ) -> Generator[pp.pandapowerNet, None, None]:
        """Generate network with perturbed generator cost coefficients.

        Args:
            net: The power network.
            sigma: A constant that specifies the range from which to draw
                samples from a uniform distribution to be used as a scaling
                factor.  The range is given as [1-sigma, 1+sigma]

        Yields:
            The same power network, but with cost coeffiecients in the
            poly_cost table perturbed by multiplying with a scaling factor
        """
        for net_top in net_topologies:
            scl_fct = np.random.uniform(
                low=self.lower,
                high=self.upper,
                size=self.sample_size,
            )
            net_top.poly_cost[self.perturb_cols] = (
                net_top.poly_cost[self.perturb_cols] * scl_fct
            )
            yield net_top
