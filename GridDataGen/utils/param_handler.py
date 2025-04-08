import yaml
import argparse
import itertools
from GridDataGen.utils.load import *


class NestedNamespace(argparse.Namespace):
    """
    A namespace object that supports nested structures, allowing for
    easy access and manipulation of hierarchical configurations.

    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                # Recursively convert dictionaries to NestedNamespace
                setattr(self, key, NestedNamespace(**value))
            else:
                setattr(self, key, value)

    def to_dict(self):
        # Recursively convert NestedNamespace back to dictionary
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, NestedNamespace):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def flatten(self, parent_key="", sep="."):
        # Flatten the dictionary with dot-separated keys
        items = []
        for key, value in self.__dict__.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, NestedNamespace):
                items.extend(value.flatten(new_key, sep=sep).items())
            else:
                items.append((new_key, value))
        return dict(items)


def flatten_dict(d, parent_key="", sep="."):
    """
    Flatten a nested dictionary into a single-level dictionary with dot-separated keys.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str, optional): Prefix for the keys in the flattened dictionary.
        sep (str, optional): Separator for nested keys. Defaults to '.'.

    Returns:
        dict: A flattened version of the input dictionary.
    """
    items = []
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def unflatten_dict(d, sep="."):
    """
    Reconstruct a nested dictionary from a flattened dictionary with dot-separated keys.

    Args:
        d (dict): The flattened dictionary to unflatten.
        sep (str, optional): Separator used in the flattened keys. Defaults to '.'.

    Returns:
        dict: A nested dictionary reconstructed from the flattened input.
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        target = result
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result


def merge_dict(base, updates):
    """
    Recursively merge updates into a base dictionary, but only if the keys exist in the base.

    Args:
        base (dict): The original dictionary to be updated.
        updates (dict): The dictionary containing updates.

    Raises:
        KeyError: If a key in updates does not exist in base.
        TypeError: If a key in base is not a dictionary but updates attempt to provide nested values.
    """
    for key, value in updates.items():
        if key not in base:
            raise KeyError(f"Key '{key}' not found in base configuration.")

        if isinstance(value, dict):
            if not isinstance(base[key], dict):
                raise TypeError(
                    f"Default config expects  {type(base[key])}, but got a dict at key '{key}'"
                )
            # Recursively merge dictionaries
            merge_dict(base[key], value)
        else:
            # Update the existing key
            base[key] = value


def get_load_scenario_generator(args) -> LoadScenarioGeneratorBase:
    """
    Returns a load scenario generator class.
    """
    if args.load.generator == "agg_load_profile":
        return LoadScenariosFromAggProfile(
            args.load.agg_profile,
            args.load.sigma,
            args.load.change_reactive_power,
            args.load.global_range,
            args.load.max_scaling_factor,
            args.load.step_size,
            args.load.start_scaling_factor,
        )
    if args.load.generator == "powergraph":
        return Powergraph(args.load.agg_profile)
