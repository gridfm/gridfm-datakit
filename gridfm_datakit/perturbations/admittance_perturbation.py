import numpy as np
import pandapower as pp
from abc import ABC, abstractmethod
from typing import Generator, List, Union


class AdmittanceGenerator(ABC):
    """Abstract base class for applying perturbations to line admittances."""

    def __init__(self) -> None:
        """Initialize the admittance generator."""
        pass

    @abstractmethod
    def generate(
        self,
        example_generator: Generator[pp.pandapowerNet, None, None],
    ) -> Union[Generator[pp.pandapowerNet, None, None], List[pp.pandapowerNet]]:
        """Generate admittance perturbations.

        Args:
            example_generator: A generator producing example (load/topology/generation)
            scenarios to which line admittance perturbations are added.

        Yields:
            An admittance-perturbed scenario.
        """
        pass


class NoAdmittancePerturbationGenerator(AdmittanceGenerator):
    """Generator that yields the original example generator without any perturbations."""

    def __init__(self):
        """Initialize the no-perturbation generator"""
        pass

    def generate(
        self,
        example_generator: Generator[pp.pandapowerNet, None, None],
    ) -> Generator[pp.pandapowerNet, None, None]:
        """Yield the original examples without any perturbations.

        Args:
            example_generator: A generator producing example
            (load/topology/generation) scenarios to which line admittance
            perturbations are added.

        Yields:
            The original example produced by the example_generator.
        """
        for example in example_generator:
            yield example


class PerturbAdmittanceGenerator(AdmittanceGenerator):
    """Class for applying perturbations to line admittances.

    This class is for generating different line admittance scenarios
    by applying perturbations to the resistance (R) and reactance (X)
    values of the lines.  Perturbations are applied as a scaling factor
    sampled from a uniform distribution with a given lower and upper
    bound.
    """

    def __init__(self, base_net: pp.pandapowerNet, sigma: float) -> None:
        """
        Initialize the line admittance perturbation generator.

        Args:
            base_net: The base power network.
        """
        self.base_net = base_net
        self.r_original = self.base_net.line["r_ohm_per_km"].values
        self.x_original = self.base_net.line["x_ohm_per_km"].values
        self.lower = np.max([0.0, 1.0 - sigma])
        self.upper = 1.0 + sigma
        self.sample_size = self.r_original.shape[0]

    def generate(
        self,
        example_generator: Generator[pp.pandapowerNet, None, None],
    ) -> Generator[pp.pandapowerNet, None, None]:
        """Generate a network with perturbed line admittance values.

        Args:
            example_generator: A generator producing example
                (load/topology/generation) scenarios to which line admittance
                perturbations are added.

            sigma: A constant that specifies the range from which to draw
                samples from a uniform distribution to be used as a scaling
                factor for resistance and and reactance. The range is
                set as [max([0,1-sigma]), 1+sigma)

        Yields:
            An example scenario with random perturbations applied to line
            admittances.
        """
        for example in example_generator:
            example.line["r_ohm_per_km"] = np.random.uniform(
                self.lower * self.r_original,
                self.upper * self.r_original,
                self.r_original.shape[0],
            )

            example.line["x_ohm_per_km"] = np.random.uniform(
                self.lower * self.x_original,
                self.upper * self.x_original,
                self.x_original.shape[0],
            )

            yield example
