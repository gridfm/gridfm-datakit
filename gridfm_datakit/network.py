import os
from pandapower.auxiliary import pandapowerNet
import requests
from importlib import resources
import pandapower as pp
import pandapower.networks as pn
import warnings


def load_net_from_pp(grid_name: str) -> pandapowerNet:
    """Loads a network from the pandapower library.

    Args:
        grid_name: Name of the grid case file in pandapower library.

    Returns:
        pandapowerNet: Loaded power network configuration.
    """
    network = getattr(pn, grid_name)()
    return network


def load_net_from_file(network_path: str) -> pandapowerNet:
    """Loads a network from a matpower file.

    Args:
        network_path: Path to the matpower file (without extension).

    Returns:
        pandapowerNet: Loaded power network configuration with reindexed buses.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    network = pp.converter.from_mpc(str(network_path))
    warnings.resetwarnings()

    old_bus_indices = network.bus.index
    new_bus_indices = range(len(network.bus))

    # Create a mapping dictionary
    bus_mapping = dict(zip(old_bus_indices, new_bus_indices))

    # Reindex the buses in the network
    pp.reindex_buses(network, bus_mapping)

    return network


def load_net_from_pglib(grid_name: str) -> pandapowerNet:
    """Loads a power grid network from PGLib.

    Downloads the network file if not locally available and loads it into a pandapower network.
    The buses are reindexed to ensure continuous indices.

    Args:
        grid_name: Name of the grid file without the prefix 'pglib_opf_' (e.g., 'case14_ieee', 'case118_ieee').

    Returns:
        pandapowerNet: Loaded power network configuration with reindexed buses.

    Raises:
        requests.exceptions.RequestException: If download fails.
    """
    # Construct file paths
    file_path = str(
        resources.files("gridfm_datakit.grids").joinpath(f"pglib_opf_{grid_name}.m"),
    )

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Download file if not exists
    if not os.path.exists(file_path):
        url = f"https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/pglib_opf_{grid_name}.m"
        response = requests.get(url)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            f.write(response.content)

    # Load network from file
    warnings.filterwarnings("ignore", category=FutureWarning)
    network = pp.converter.from_mpc(file_path)
    warnings.resetwarnings()

    old_bus_indices = network.bus.index
    new_bus_indices = range(len(network.bus))

    # Create a mapping dictionary
    bus_mapping = dict(zip(old_bus_indices, new_bus_indices))

    # Reindex the buses in the network
    pp.reindex_buses(network, bus_mapping)

    return network
