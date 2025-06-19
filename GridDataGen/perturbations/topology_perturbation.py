import numpy as np
import pandapower as pp
import copy
from itertools import combinations
from abc import ABC, abstractmethod
import pandapower.topology as top
import warnings
from typing import Generator, List, Union


# Abstract base class for topology generation
class TopologyGenerator(ABC):
    """Abstract base class for generating perturbed network topologies."""

    def __init__(self) -> None:
        """Initialize the topology generator."""
        pass

    @abstractmethod
    def generate(
        self,
        net: pp.pandapowerNet,
    ) -> Union[Generator[pp.pandapowerNet, None, None], List[pp.pandapowerNet]]:
        """Generate perturbed topologies.

        Args:
            net: The power network to perturb.

        Yields:
            A perturbed network topology.
        """
        pass


class NoPerturbationGenerator(TopologyGenerator):
    """Generator that yields the original network without any perturbations."""

    def generate(
        self,
        net: pp.pandapowerNet,
    ) -> Generator[pp.pandapowerNet, None, None]:
        """Yield the original network without any perturbations.

        Args:
            net: The power network.

        Yields:
            The original power network.
        """
        yield net


class NMinusKGenerator(TopologyGenerator):
    """Generate perturbed topologies for N-k contingency analysis.

    Only considers lines and transformers. Generates ALL possible topologies with at most k
    components set out of service (lines and transformers).

    Only topologies that are feasible (= no unsupplied buses) are yielded.

    Attributes:
        k: Maximum number of components to drop.
        components_to_drop: List of tuples containing component indices and types.
        component_combinations: List of all possible combinations of components to drop.
    """

    def __init__(self, k: int, base_net: pp.pandapowerNet) -> None:
        """Initialize the N-k generator.

        Args:
            k: Maximum number of components to drop.
            base_net: The base power network.

        Raises:
            ValueError: If k is 0.
            Warning: If k > 1, as this may result in slow data generation.
        """
        super().__init__()
        if k > 1:
            warnings.warn("k>1. This may result in slow data generation process.")
        if k == 0:
            raise ValueError(
                'k must be greater than 0. Use "none" as argument for the generator_type if you don\'t want to generate any perturbation',
            )
        self.k = k

        # Prepare the list of components to drop
        self.components_to_drop = [(index, "line") for index in base_net.line.index] + [
            (index, "trafo") for index in base_net.trafo.index
        ]

        # Generate all combinations of at most k components
        self.component_combinations = []
        for r in range(self.k + 1):
            self.component_combinations.extend(combinations(self.components_to_drop, r))

        print(
            f"Number of possible topologies with at most {self.k} dropped components: {len(self.component_combinations)}",
        )

    def generate(
        self,
        net: pp.pandapowerNet,
    ) -> Generator[pp.pandapowerNet, None, None]:
        """Generate perturbed topologies by dropping components.

        Args:
            net: The power network.

        Yields:
            A perturbed network topology with at most k components removed.
        """
        for selected_components in self.component_combinations:
            perturbed_topology = copy.deepcopy(net)

            # Separate lines and transformers
            lines_to_drop = [e[0] for e in selected_components if e[1] == "line"]
            trafos_to_drop = [e[0] for e in selected_components if e[1] == "trafo"]

            # Drop selected lines and transformers
            if lines_to_drop:
                perturbed_topology.line.loc[lines_to_drop, "in_service"] = False
            if trafos_to_drop:
                perturbed_topology.trafo.loc[trafos_to_drop, "in_service"] = False

            # Check network feasibility and yield the topology
            if not len(top.unsupplied_buses(perturbed_topology)):
                yield perturbed_topology


class RandomComponentDropGenerator(TopologyGenerator):
    """Generate perturbed topologies by randomly setting components out of service.

    Generates perturbed topologies by randomly setting out of service at most k components among the selected element types.
    Only topologies that are feasible (= no unsupplied buses) are yielded.

    Attributes:
        n_topology_variants: Number of topology variants to generate.
        k: Maximum number of components to drop.
        components_to_drop: List of tuples containing component indices and types.
    """

    def __init__(
        self,
        n_topology_variants: int,
        k: int,
        base_net: pp.pandapowerNet,
        elements: List[str] = ["line", "trafo", "gen", "sgen"],
    ) -> None:
        """Initialize the random component drop generator.

        Args:
            n_topology_variants: Number of topology variants to generate.
            k: Maximum number of components to drop.
            base_net: The base power network.
            elements: List of element types to consider for dropping.
        """
        super().__init__()
        self.n_topology_variants = n_topology_variants
        self.k = k

        # Create a list of all components that can be dropped
        self.components_to_drop = []
        for element in elements:
            self.components_to_drop.extend(
                [(index, element) for index in base_net[element].index],
            )

    def generate(
        self,
        net: pp.pandapowerNet,
    ) -> Generator[pp.pandapowerNet, None, None]:
        """Generate perturbed topologies by randomly setting components out of service.

        Args:
            net: The power network.

        Yields:
            A perturbed network topology.
        """
        n_generated_topologies = 0

        # Stop after we generated n_topology_variants
        while n_generated_topologies < self.n_topology_variants:
            perturbed_topology = copy.deepcopy(net)

            # draw the number of components to drop from a uniform distribution
            r = np.random.randint(
                1,
                self.k + 1,
            )  # TODO: decide if we want to be able to set 0 components out of service

            # Randomly select r<=k components to drop
            components = tuple(
                np.random.choice(range(len(self.components_to_drop)), r, replace=False),
            )

            # Convert indices back to actual components
            selected_components = tuple(
                self.components_to_drop[idx] for idx in components
            )

            # Separate lines, transformers, generators, and static generators
            lines_to_drop = [e[0] for e in selected_components if e[1] == "line"]
            trafos_to_drop = [e[0] for e in selected_components if e[1] == "trafo"]
            gens_to_turn_off = [e[0] for e in selected_components if e[1] == "gen"]
            sgens_to_turn_off = [e[0] for e in selected_components if e[1] == "sgen"]

            # Drop selected lines and transformers, turn off generators and static generators
            if lines_to_drop:
                perturbed_topology.line.loc[lines_to_drop, "in_service"] = False
            if trafos_to_drop:
                perturbed_topology.trafo.loc[trafos_to_drop, "in_service"] = False
            if gens_to_turn_off:
                perturbed_topology.gen.loc[gens_to_turn_off, "in_service"] = False
            if sgens_to_turn_off:
                perturbed_topology.sgen.loc[sgens_to_turn_off, "in_service"] = False

            # Check network feasibility and yield the topology
            if not len(top.unsupplied_buses(perturbed_topology)):
                yield perturbed_topology
                n_generated_topologies += 1
