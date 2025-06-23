import argparse
from gridfm_datakit.perturbations.load_perturbation import (
    LoadScenarioGeneratorBase,
    LoadScenariosFromAggProfile,
    Powergraph,
)
from typing import Dict, Any
import warnings
from pandapower import pandapowerNet
from gridfm_datakit.perturbations.topology_perturbation import (
    NMinusKGenerator,
    RandomComponentDropGenerator,
    NoPerturbationGenerator,
    TopologyGenerator,
)


class NestedNamespace(argparse.Namespace):
    """A namespace object that supports nested structures.

    This class extends argparse.Namespace to support hierarchical configurations,
    allowing for easy access and manipulation of nested parameters.

    Attributes:
        __dict__: Dictionary containing the namespace attributes.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initializes a NestedNamespace with the given keyword arguments.

        Args:
            **kwargs: Key-value pairs to initialize the namespace.
        """
        for key, value in kwargs.items():
            if isinstance(value, dict):
                # Recursively convert dictionaries to NestedNamespace
                setattr(self, key, NestedNamespace(**value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the NestedNamespace back to a dictionary.

        Returns:
            Dict containing the namespace attributes, with nested NestedNamespace
            objects converted to dictionaries.
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, NestedNamespace):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def flatten(self, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """Flattens the namespace into a single-level dictionary.

        Args:
            parent_key: Prefix for the keys in the flattened dictionary.
            sep: Separator for nested keys.

        Returns:
            Dict with dot-separated keys representing the nested structure.
        """
        items = []
        for key, value in self.__dict__.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, NestedNamespace):
                items.extend(value.flatten(new_key, sep=sep).items())
            else:
                items.append((new_key, value))
        return dict(items)


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    """Flattens a nested dictionary into a single-level dictionary.

    Args:
        d: The dictionary to flatten.
        parent_key: Prefix for the keys in the flattened dictionary.
        sep: Separator for nested keys. Defaults to '.'.

    Returns:
        A flattened version of the input dictionary with dot-separated keys.
    """
    items = []
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """Reconstructs a nested dictionary from a flattened dictionary.

    Args:
        d: The flattened dictionary to unflatten.
        sep: Separator used in the flattened keys. Defaults to '.'.

    Returns:
        A nested dictionary reconstructed from the flattened input.
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        target = result
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result


def merge_dict(base: Dict[str, Any], updates: Dict[str, Any]) -> None:
    """Recursively merges updates into a base dictionary.

    Only merges keys that exist in the base dictionary. Raises errors for
    invalid updates.

    Args:
        base: The original dictionary to be updated.
        updates: The dictionary containing updates.

    Raises:
        KeyError: If a key in updates does not exist in base.
        TypeError: If a key in base is not a dictionary but updates attempt to
            provide nested values.
    """
    for key, value in updates.items():
        if key not in base:
            raise KeyError(f"Key '{key}' not found in base configuration.")

        if isinstance(value, dict):
            if not isinstance(base[key], dict):
                raise TypeError(
                    f"Default config expects  {type(base[key])}, but got a dict at key '{key}'",
                )
            # Recursively merge dictionaries
            merge_dict(base[key], value)
        else:
            # Update the existing key
            base[key] = value


def get_load_scenario_generator(args: NestedNamespace) -> LoadScenarioGeneratorBase:
    """Creates and returns a load scenario generator based on configuration.

    Args:
        args: Configuration namespace containing load generator parameters.

    Returns:
        An instance of a LoadScenarioGeneratorBase subclass configured according
        to the provided arguments.

    Note:
        Currently supports 'agg_load_profile' and 'powergraph' generator types.
    """
    if args.generator == "agg_load_profile":
        return LoadScenariosFromAggProfile(
            args.agg_profile,
            args.sigma,
            args.change_reactive_power,
            args.global_range,
            args.max_scaling_factor,
            args.step_size,
            args.start_scaling_factor,
        )
    if args.generator == "powergraph":
        unused_args = {
            key: value
            for key, value in args.flatten().items()
            if key not in ["type", "agg_profile"]
        }
        if unused_args:
            warnings.warn(
                f"The following arguments are not used by the powergraph generator: {unused_args}",
                UserWarning,
            )

        return Powergraph(args.agg_profile)


def initialize_generator(
    args: NestedNamespace,
    base_net: pandapowerNet,
) -> TopologyGenerator:
    """Initialize the appropriate topology generator based on the given arguments.

    Args:
        args: Configuration arguments containing generator type and parameters.
        base_net: Base network to analyze.

    Returns:
        TopologyGenerator: The initialized topology generator.

    Raises:
        ValueError: If the generator type is unknown.
    """
    if args.type == "n_minus_k":
        if not hasattr(args, "k"):
            raise ValueError("k parameter is required for n_minus_k generator")
        generator = NMinusKGenerator(args.k, base_net)
        used_args = {"k": args.k, "base_net": base_net}

    elif args.type == "random":
        if not all(hasattr(args, attr) for attr in ["n_topology_variants", "k"]):
            raise ValueError(
                "n_topology_variants and k parameters are required for random generator",
            )
        elements = getattr(args, "elements", ["line", "trafo", "gen", "sgen"])
        generator = RandomComponentDropGenerator(
            args.n_topology_variants,
            args.k,
            base_net,
            elements,
        )
        used_args = {
            "n_topology_variants": args.n_topology_variants,
            "k": args.k,
            "base_net": base_net,
            "elements": elements,
        }

    elif args.type == "none":
        generator = NoPerturbationGenerator()
        used_args = {}

    else:
        raise ValueError(f"Unknown generator type: {args.type}")

    # Check for unused arguments
    unused_args = {
        key: value
        for key, value in args.flatten().items()
        if key not in used_args and key != "type"
    }
    if unused_args:
        warnings.warn(
            f'The following arguments are not used by the topology generator "{args.type}": {unused_args}',
            UserWarning,
        )

    return generator
