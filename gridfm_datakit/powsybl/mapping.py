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


def build_p2g_maps_by_bus_number(
    network: Network,
    pp_net,
) -> MappingP2G:
    """Build pypowsybl-to-gridfm index maps for a network built by ``to_powsybl``.

    ``to_powsybl`` names each bus-breaker bus ``BUS-<n>`` after its original
    bus number.  pypowsybl lists buses by voltage level, and buses joined by a
    transformer share one, so position does not give the gridfm row.

    Args:
        network: The gridfm_datakit Network passed to ``to_powsybl``.
        pp_net: The pypowsybl network ``to_powsybl(network)`` built.

    Returns:
        MappingP2G with the bus map read from the bus-breaker buses.

    Raises:
        ValueError: If a pypowsybl bus groups several gridfm buses or has none.
    """
    maps = build_p2g_maps(network, pp_net)
    maps.bus = {}
    bb_buses = pp_net.get_bus_breaker_view_buses(attributes=["bus_id"])
    for bb_bus_id, pp_bus_id in bb_buses["bus_id"].items():
        if not pp_bus_id:
            continue
        gfm_bus = network.bus_index_mapping[_bus_number(bb_bus_id)]
        if maps.bus.setdefault(pp_bus_id, gfm_bus) != gfm_bus:
            raise ValueError(f"pypowsybl bus {pp_bus_id} groups several gridfm buses")
    unmapped = set(pp_net.get_buses().index) - set(maps.bus)
    if unmapped:
        raise ValueError(f"No gridfm bus for pypowsybl buses {sorted(unmapped)}")
    return maps


def _bus_number(bus_breaker_bus_id: str) -> int:
    """Return the original bus number of a ``BUS-<n>`` bus-breaker bus ID."""
    return int(bus_breaker_bus_id.rsplit("-", 1)[1])
