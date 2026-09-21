"""
lightsim2grid integration module for gridfm_datakit.

lightsim2grid is a C++ power flow engine. This module bridges it with
gridfm_datakit's internal :class:`~gridfm_datakit.network.Network`, following
the layout of :mod:`gridfm_datakit.powsybl`:

* :func:`convert_net` — convert a gridfm_datakit Network to a lightsim2grid
  ``LSGrid`` and the index maps between the two.
* :func:`update_lightsim2grid` — re-synchronise the ``LSGrid`` with a perturbed
  copy of the Network.
* :func:`run_ls_pf` — run an AC or DC power flow and format the result like
  PowerModels' one, for ``pf_post_processing``.

Unlike PowSyBl there is no dedicated reader: the network is always read with
the native reader and lightsim2grid only replaces the power flow solver
(``settings.pf_solver: lightsim2grid``). OPF is still solved by PowerModels.
"""

from dataclasses import dataclass
from typing import Any

from gridfm_datakit.network import Network

from .api import (
    check_lightsim2grid_available,
    is_lightsim2grid_available,
    lightsim2grid_network,
)
from .convert import ConvertedNetwork, to_lightsim2grid, update_lightsim2grid
from .mapping import MappingL2G, build_l2g_maps
from .preprocess import get_pf_res, run_ls_pf


@dataclass
class LoadedNetwork:
    """Bundles the lightsim2grid and gridfm_datakit representations of a network."""

    ls_net: Any  # lightsim2grid.network.LSGrid
    gfm_net: Network
    mapping_l2g: MappingL2G


def convert_net(network: Network) -> LoadedNetwork:
    """Convert a gridfm_datakit Network to lightsim2grid.

    Args:
        network: The network to convert.

    Returns:
        The lightsim2grid LSGrid, the network itself and the index maps between the two.
    """
    conv = to_lightsim2grid(network)
    return LoadedNetwork(
        ls_net=conv.ls_net,
        gfm_net=network,
        mapping_l2g=conv.mapping_l2g,
    )


__all__ = [
    "convert_net",
    "to_lightsim2grid",
    "update_lightsim2grid",
    "build_l2g_maps",
    "MappingL2G",
    "ConvertedNetwork",
    "LoadedNetwork",
    "lightsim2grid_network",
    "is_lightsim2grid_available",
    "check_lightsim2grid_available",
    "run_ls_pf",
    "get_pf_res",
]
