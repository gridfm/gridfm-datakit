import numpy as np
import copy
from itertools import combinations
from abc import ABC, abstractmethod
import warnings
from typing import Generator, List, Mapping, Sequence, Union
from gridfm_datakit.network import Network
from gridfm_datakit.utils.idx_gen import GEN_BUS


# Abstract base class for topology generation
class TopologyGenerator(ABC):
    """Abstract base class for generating perturbed network topologies."""

    def __init__(self) -> None:
        """Initialize the topology generator."""
        pass

    @abstractmethod
    def generate(
        self,
        net: Network,
    ) -> Union[Generator[Network, None, None], List[Network]]:
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
        net: Network,
    ) -> Generator[Network, None, None]:
        """Yield the original network without any perturbations.

        Args:
            net: The power network.

        Yields:
            The original power network.
        """
        yield copy.deepcopy(net)


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

    def __init__(self, k: int, base_net: dict) -> None:
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
        self.components_to_drop = base_net.idx_branches_in_service

        # Generate all combinations of at most k components
        self.component_combinations = []
        for r in range(self.k + 1):
            self.component_combinations.extend(combinations(self.components_to_drop, r))

        print(
            f"Number of possible topologies with at most {self.k} dropped components: {len(self.component_combinations)}",
        )

    def generate(
        self,
        net: Network,
    ) -> Generator[Network, None, None]:
        """Generate perturbed topologies by dropping components.
        Does not change the original network.

        Args:
            net: The power network.

        Yields:
            A perturbed network topology with at most k components removed.
        """
        for selected_components in self.component_combinations:
            perturbed_topology = copy.deepcopy(net)

            perturbed_topology.deactivate_branches(selected_components)

            # Check network feasibility and yield the topology
            if perturbed_topology.check_single_connected_component():
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
        base_net: Network,
        elements: List[str] = ["branch", "gen"],
        outage_count_probabilities: Sequence[float] | Mapping[int, float] | None = None,
        max_generation_attempts: int | None = None,
    ) -> None:
        """Initialize the random component drop generator.

        Args:
            n_topology_variants: Number of topology variants to generate.
            k: Maximum number of components to drop.
            base_net: The base power network.
            elements: List of element types to consider for dropping.
            outage_count_probabilities: Optional probabilities over outage counts.
                Index or key `i` means probability of sampling `i` outages.
            max_generation_attempts: Optional cap on sampled topology attempts
                before failing if too many sampled topologies are infeasible.
                If not provided, a default limit of
                ``max(500, 50 * n_topology_variants)`` is used.
        """
        super().__init__()
        self.n_topology_variants = n_topology_variants
        self.k = k

        # Create a list of all components that can be dropped
        self.components_to_drop = []
        if "branch" in elements:
            self.components_to_drop.extend(
                (idx, "branch") for idx in base_net.idx_branches_in_service
            )
        if "gen" in elements:
            self.components_to_drop.extend(
                (idx, "gen")
                for idx in base_net.idx_gens_in_service
                if base_net.gens[idx, GEN_BUS] != base_net.ref_bus_idx
            )

        # Preserve the current 1..k uniform sampling when no explicit count
        # probabilities are provided, but allow configurable sampling over 0..k.
        self.outage_count_values, self.outage_count_probabilities = (
            self._normalize_outage_count_probabilities(k, outage_count_probabilities)
        )
        self.max_generation_attempts = self._resolve_max_generation_attempts(
            n_topology_variants,
            max_generation_attempts,
        )

    def generate(
        self,
        net: Network,
    ) -> Generator[Network, None, None]:
        """Generate perturbed topologies by randomly setting components out of service.

        Args:
            net: The power network.

        Yields:
            A perturbed network topology.
        """
        n_generated_topologies = 0
        n_attempts = 0

        # Stop after we generated n_topology_variants
        while n_generated_topologies < self.n_topology_variants:
            if n_attempts >= self.max_generation_attempts:
                raise RuntimeError(
                    "Unable to generate "
                    f"{self.n_topology_variants} feasible topologies within "
                    f"{self.max_generation_attempts} attempts. "
                    "Consider reducing k or relaxing outage_count_probabilities.",
                )
            n_attempts += 1
            perturbed_topology = copy.deepcopy(net)

            r = self._sample_outage_count()

            # Randomly select r<=k components to drop
            components = tuple(
                np.random.choice(range(len(self.components_to_drop)), r, replace=False),
            )

            # Convert indices back to actual components
            selected_components = tuple(
                self.components_to_drop[idx] for idx in components
            )

            # Separate lines, transformers, generators, and static generators
            branches_to_drop = [
                idx for idx, element in selected_components if element == "branch"
            ]
            gens_to_drop = [
                idx for idx, element in selected_components if element == "gen"
            ]

            # Drop selected lines and transformers, turn off generators and static generators
            perturbed_topology.deactivate_branches(branches_to_drop)
            perturbed_topology.deactivate_gens(gens_to_drop)

            # Check network feasibility and yield the topology
            if perturbed_topology.check_single_connected_component():
                yield perturbed_topology
                n_generated_topologies += 1

    def _sample_outage_count(self) -> int:
        if self.outage_count_probabilities is None:
            return int(np.random.randint(1, self.k + 1))
        return int(
            np.random.choice(
                self.outage_count_values,
                p=self.outage_count_probabilities,
            ),
        )

    @staticmethod
    def _resolve_max_generation_attempts(
        n_topology_variants: int,
        max_generation_attempts: int | None,
    ) -> int:
        if max_generation_attempts is None:
            return max(500, 50 * max(1, int(n_topology_variants)))
        if int(max_generation_attempts) <= 0:
            raise ValueError("max_generation_attempts must be greater than 0.")
        return int(max_generation_attempts)

    @staticmethod
    def _normalize_outage_count_probabilities(
        k: int,
        probabilities: Sequence[float] | Mapping[int, float] | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Validate optional outage-count probabilities for random topology sampling."""

        if probabilities is None:
            return None, None

        if isinstance(probabilities, Mapping):
            allowed_counts = np.arange(k + 1, dtype=int)
            probability_values = np.zeros(k + 1, dtype=float)
            for raw_count, raw_probability in probabilities.items():
                count = int(raw_count)
                if count < 0 or count > k:
                    raise ValueError(
                        f"Outage count {count} is outside the supported range 0..{k}.",
                    )
                probability_values[count] = float(raw_probability)
            return RandomComponentDropGenerator._validate_outage_count_probabilities(
                allowed_counts,
                probability_values,
            )

        probability_values = np.asarray(probabilities, dtype=float)
        if probability_values.ndim != 1:
            raise ValueError(
                "outage_count_probabilities must be a one-dimensional sequence.",
            )
        if len(probability_values) != k + 1:
            raise ValueError(
                "outage_count_probabilities sequence must have length k + 1 so index i maps to i outages.",
            )
        allowed_counts = np.arange(k + 1, dtype=int)
        return RandomComponentDropGenerator._validate_outage_count_probabilities(
            allowed_counts,
            probability_values,
        )

    @staticmethod
    def _validate_outage_count_probabilities(
        counts: np.ndarray,
        probability_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.any(probability_values < 0.0):
            raise ValueError("outage_count_probabilities must be non-negative.")
        total_probability = float(probability_values.sum())
        if not np.isclose(total_probability, 1.0, atol=1e-8):
            raise ValueError(
                "outage_count_probabilities must sum to 1.0 within numerical tolerance.",
            )
        if not np.any(probability_values > 0.0):
            raise ValueError(
                "outage_count_probabilities must contain at least one positive value.",
            )
        return counts.astype(int, copy=True), probability_values.astype(
            float,
            copy=True,
        )
