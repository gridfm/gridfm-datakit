from dataclasses import dataclass
from typing import Dict

from gridfm_datakit.network import Network

from .api import check_powsybl_available


@dataclass
class MappingP2G:
    """Index maps from pypowsybl element IDs to gridfm row indices.

    Attributes
    ----------
    bus : Dict[str, float]
        ``{pp_bus_id: gfm_bus_index}``
    branch : Dict[str, int]
        ``{pp_branch_id: gfm_branch_row}``
    gen : Dict[str, int]
        ``{pp_gen_id: gfm_gen_row}``
    """

    bus: Dict[str, float]
    branch: Dict[str, int]
    gen: Dict[str, int]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def build_p2g_maps(
    network: Network,
    pp_net,
) -> MappingP2G:
    """Build pypowsybl-to-gridfm ID maps in O(n) by parsing pypowsybl element IDs.

    When a gridfm :class:`~gridfm_datakit.network.Network` is converted to
    pypowsybl via :func:`~gridfm_datakit.powsybl.convert.to_powsybl`, the
    element IDs encode the **original** (pre-normalisation) bus numbers from
    ``network.reverse_bus_index_mapping``.  This function reads those IDs and
    converts back to 0-based gridfm indices via ``network.bus_index_mapping``.

    The maps can be built once on the **base network** and then reused across
    all perturbed scenarios, because perturbations preserve element identity
    and row ordering, and ``to_powsybl`` always uses ``reverse_bus_index_mapping``
    to produce consistent IDs.

    Parameters
    ----------
    network:
        The gridfm_datakit Network passed to ``to_powsybl()`` to produce *pp_net*.
    pp_net:
        The pypowsybl network produced by ``to_powsybl(network)``.

    Returns
    -------
    MappingP2G
        Dataclass bundling the three maps: ``bus``, ``branch``, and ``gen``.

    Raises
    ------
    ValueError
        If any pypowsybl element ID does not match the expected naming pattern,
        or if one or more pypowsybl buses cannot be assigned a gridfm index.

    Examples
    --------
    >>> from gridfm_datakit.network import load_net_from_pglib
    >>> from gridfm_datakit.powsybl.convert import to_powsybl
    >>> from gridfm_datakit.powsybl.mapping import build_p2g_maps
    >>>
    >>> net = load_net_from_pglib("case14_ieee")
    >>> result = to_powsybl(net)
    >>> mapping = build_p2g_maps(net, result.pp_net)
    >>> mapping.bus    # pp_bus_id → gfm index
    >>> mapping.branch # pp_branch_id → gfm row
    >>> mapping.gen    # pp_gen_id → gfm row
    """
    check_powsybl_available()

    # -------------------------------------------------------------------------
    # 0. Bus map - direct enumeration (row order is preserved by pypowsybl)
    # -------------------------------------------------------------------------
    map_bus_p2g: Dict[str, float] = {
        pp_bus_id: gfm_row for gfm_row, pp_bus_id in enumerate(pp_net.get_buses().index)
    }

    # -------------------------------------------------------------------------
    # 1. Gen map — direct enumeration (row order is preserved by pypowsybl)
    # -------------------------------------------------------------------------
    map_gen_p2g: Dict[str, int] = {
        pp_gen_id: gfm_row
        for gfm_row, pp_gen_id in enumerate(pp_net.get_generators().index)
    }

    # -------------------------------------------------------------------------
    # 2. Branch map — direct enumeration (lines, transformers, ...)
    # -------------------------------------------------------------------------
    offset = 0
    map_branch_p2g: Dict[str, int] = {}
    for df in (pp_net.get_lines(), pp_net.get_2_windings_transformers()):
        for row, pp_branch_id in enumerate(df.index):
            map_branch_p2g[pp_branch_id] = offset + row
        offset += len(df)

    return MappingP2G(bus=map_bus_p2g, branch=map_branch_p2g, gen=map_gen_p2g)
