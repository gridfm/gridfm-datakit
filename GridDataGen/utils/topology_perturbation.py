import numpy as np
import pandapower as pp
from GridDataGen.utils.io import *
from GridDataGen.utils.process_network import *
from GridDataGen.utils.config import *
from GridDataGen.utils.solvers import *
import copy
from itertools import combinations
from abc import ABC, abstractmethod
import pandapower.topology as top
import math
import warnings
import time


# Abstract base class for topology generation
class TopologyGenerator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate(self, net):
        """
        Abstract method for generating perturbed topologies.

        Args:
            net (pandapowerNet): The power network.

        Yields:
            pandapowerNet: A perturbed network topology.
        """
        pass


class NoPerturbationGenerator(TopologyGenerator):
    def generate(self, net):
        """
        Yields the original network without any perturbations.

        Args:
            net (pandapowerNet): The power network.

        Yields:
            pandapowerNet: The original power network.
        """
        yield net


class NMinusKGenerator(TopologyGenerator):
    """
    Generate perturbed topologies for N-k contingency analysis. Only considers lines and transformers
    Generates ALL possible topologies with at most k dropped components (lines and transformers)
    """

    def __init__(self, k, base_net):
        super().__init__()
        # warn when k>1 as the number of combinations is exponential in k, which may result in slow data gen process
        if k > 1:
            warnings.warn("k>1. This may result in slow data generation process.")
        if k == 0:
            raise ValueError(
                'k must be greater than 0. Use "none" as argument for the generator_type if you don\'t want to generate any perturbation'
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
            f"Number of possible topologies with at most {self.k} dropped components: {len(self.component_combinations)}"
        )

    def generate(self, net):
        """
        Args:
            net (pandapowerNet): The power network.

        Yields:
            pandapowerNet: A perturbed network topology with at most k components removed.
        """

        for selected_components in self.component_combinations:
            perturbed_topology = copy.deepcopy(net)

            # Separate lines and transformers
            lines_to_drop = [e[0] for e in selected_components if e[1] == "line"]
            trafos_to_drop = [e[0] for e in selected_components if e[1] == "trafo"]

            # Drop selected lines and transformers
            if lines_to_drop:
                pp.drop_lines(perturbed_topology, lines_to_drop)
            if trafos_to_drop:
                pp.drop_trafos(perturbed_topology, trafos_to_drop)

            # Check network feasibility and yield the topology
            if not len(top.unsupplied_buses(perturbed_topology)):
                yield perturbed_topology


class RandomComponentDropGenerator(TopologyGenerator):
    """
    Generates perturbed topologies by randomly dropping at most k components among lines, transformers, generators, and static generators.
    Note that the generators and static generators are not dropped, but in_service is set to False!
    """

    def __init__(self, n_topology_variants, k, base_net):
        super().__init__()
        self.n_topology_variants = n_topology_variants
        self.k = k

        # Create a list of all components that can be dropped
        self.components_to_drop = (
            [(index, "line") for index in base_net.line.index]
            + [(index, "trafo") for index in base_net.trafo.index]
            + [(index, "gen") for index in base_net.gen.index]
            + [(index, "sgen") for index in base_net.sgen.index]
        )

    def generate(self, net):
        """
        Args:
            net (pandapowerNet): The power network.

        Yields:
            pandapowerNet: A perturbed network topology.
        """
        n_generated_topologies = 0

        # Stop after we generated n_topology_variants
        while n_generated_topologies < self.n_topology_variants:
            perturbed_topology = copy.deepcopy(net)

            # draw the number of components to drop from a uniform distribution
            r = np.random.randint(0, self.k + 1)

            # Randomly select r<=k components to drop
            components = tuple(
                np.random.choice(range(len(self.components_to_drop)), r, replace=False)
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
                pp.drop_lines(perturbed_topology, lines_to_drop)
            if trafos_to_drop:
                pp.drop_trafos(perturbed_topology, trafos_to_drop)
            if gens_to_turn_off:
                perturbed_topology.gen.loc[gens_to_turn_off, "in_service"] = False
            if sgens_to_turn_off:
                perturbed_topology.sgen.loc[sgens_to_turn_off, "in_service"] = False

            # Check network feasibility and yield the topology
            if not len(top.unsupplied_buses(perturbed_topology)):
                yield perturbed_topology
                n_generated_topologies += 1


class MostOverloadedLineDropGenerator(TopologyGenerator):
    """
    Generates the n_topology_variants perturbed topologies obtained by dropping the most overloaded line or transformer.
    Each topology has only one dropped component
    """

    def __init__(self, n_topology_variants):
        super().__init__()
        self.n_topology_variants = n_topology_variants

    def generate(self, net):
        """
        Args:
            net (pandapowerNet): The power network.

        Yields:
            pandapowerNet: A perturbed network topology.
        """
        success = run_opf(net)
        if not success:
            print(
                "OPF did not converge on base topology. Cannot prioritize line overloads. Skipping the entire load scenario..."
            )
            return

        line_loading = net.res_line.loading_percent / 100
        trafo_loading = net.res_trafo.loading_percent / 100

        # Combine overloaded components (lines and transformers) into a single list with their loading percentage
        overloaded_components = [
            (index, "line", line_loading.loc[index]) for index in line_loading.index
        ] + [
            (index, "trafo", trafo_loading.loc[index]) for index in trafo_loading.index
        ]

        # Sort the combined list by the loading percentage in descending order
        overloaded_components.sort(key=lambda x: x[2], reverse=True)

        n_generated_topologies = 0

        for component_to_drop in overloaded_components:

            if n_generated_topologies >= self.n_topology_variants:
                break

            perturbed_topology = copy.deepcopy(net)

            # Drop selected line or transformer
            component_type = component_to_drop[1]
            components_to_drop = [component_to_drop[0]]

            if component_type == "line":
                pp.drop_lines(perturbed_topology, components_to_drop)
            else:
                pp.drop_trafos(perturbed_topology, components_to_drop)

            # Check network feasibility and yield the topology
            if not len(top.unsupplied_buses(perturbed_topology)):
                yield perturbed_topology
                n_generated_topologies += 1


def initialize_generator(generator_type, n_topology_variants, k, base_net):
    """
    Initialize the appropriate topology generator based on the given generator type.

    Args:
        generator_type (str): Type of topology generator (e.g., "n_minus_k", "random", "overloaded").
        n_topology_variants (int): Number of unique perturbed topologies to generate.
        k (int): Max nb of components to drop in each perturbation.
        base_net: Base topology: Make sure you call initialize_generator for every different base topology!!!

    Returns:
        TopologyGenerator: The initialized topology generator.
    """
    if generator_type == "n_minus_k":
        return NMinusKGenerator(k, base_net)
    elif generator_type == "random":
        return RandomComponentDropGenerator(n_topology_variants, k, base_net)
    elif generator_type == "overloaded":
        return MostOverloadedLineDropGenerator(n_topology_variants)
    elif generator_type == "none":
        return NoPerturbationGenerator()
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")
