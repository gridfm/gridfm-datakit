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
    """Build pypowsybl-to-gridfm index maps for a network read by ``from_powsybl``.

    :func:`~gridfm_datakit.powsybl.convert.from_powsybl` writes buses,
    generators and branches (lines, then 2-winding transformers) in the order
    pypowsybl lists them, so each map pairs the i-th pypowsybl element with
    gridfm row i.  A network built by
    :func:`~gridfm_datakit.powsybl.convert.to_powsybl` follows the gridfm
    tables instead, so use the maps ``to_powsybl`` returns.

    The maps can be built once on the **base network** and then reused across
    all perturbed scenarios, because perturbations preserve element identity
    and row ordering.

    Parameters
    ----------
    network:
        The gridfm_datakit Network returned by ``from_powsybl(pp_net)``.
    pp_net:
        The pypowsybl network passed to ``from_powsybl``.

    Returns
    -------
    MappingP2G
        Dataclass bundling the three maps: ``bus``, ``branch``, and ``gen``.

    Examples
    --------
    >>> import pypowsybl as pp
    >>> from gridfm_datakit.powsybl.convert import from_powsybl
    >>> from gridfm_datakit.powsybl.mapping import build_p2g_maps
    >>>
    >>> pp_net = pp.network.create_ieee14()
    >>> net = from_powsybl(pp_net)
    >>> mapping = build_p2g_maps(net, pp_net)
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
