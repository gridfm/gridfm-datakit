"""Index maps between a gridfm_datakit Network and a lightsim2grid LSGrid.

``lightsim2grid.network.init_from_matpower`` documents a deterministic layout:

* one lightsim2grid bus per row of ``net.buses`` (same order),
* one lightsim2grid generator per row of ``net.gens`` (same order, out of
  service ones included),
* branches are split in *powerlines* and *transformers*, each list keeping the
  order of the rows of ``net.branches``. A branch is a transformer if
  ``TAP != 0`` or ``SHIFT != 0``.
"""

from dataclasses import dataclass

import numpy as np

from gridfm_datakit.network import Network
from gridfm_datakit.utils.idx_brch import SHIFT, TAP
from gridfm_datakit.utils.idx_bus import BUS_I


@dataclass
class MappingL2G:
    """lightsim2grid-to-gridfm index maps.

    Attributes
    ----------
    line_rows : np.ndarray
        ``line_rows[k]`` is the ``net.branches`` row of the k-th lightsim2grid powerline.
    trafo_rows : np.ndarray
        ``trafo_rows[k]`` is the ``net.branches`` row of the k-th lightsim2grid transformer.
    bus_index : np.ndarray
        ``bus_index[r]`` is the continuous bus index (``BUS_I``) of the r-th
        lightsim2grid bus (= r-th row of ``net.buses``).
    """

    line_rows: np.ndarray
    trafo_rows: np.ndarray
    bus_index: np.ndarray


def build_l2g_maps(net: Network) -> MappingL2G:
    """Build the index maps of the LSGrid that :func:`to_lightsim2grid` creates from ``net``.

    Args:
        net: The network the LSGrid is built from.

    Returns:
        The maps: which ``net.branches`` rows are powerlines and transformers in
        lightsim2grid, and the bus index of every lightsim2grid bus.
    """
    is_trafo = (net.branches[:, TAP] != 0) | (net.branches[:, SHIFT] != 0)
    return MappingL2G(
        line_rows=np.flatnonzero(~is_trafo),
        trafo_rows=np.flatnonzero(is_trafo),
        bus_index=net.buses[:, BUS_I].astype(int),
    )
