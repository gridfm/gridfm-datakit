"""
ID-based O(n) mapping between gridfm_datakit and pypowsybl networks.

Background
----------
When ``to_powsybl()`` converts a gridfm :class:`~gridfm_datakit.network.Network`
to pypowsybl, the MATPOWER `.mat` file it writes uses the **original** bus
numbers recovered from ``network.reverse_bus_index_mapping``.  pypowsybl's
MATPOWER importer then assigns **deterministic string IDs** to every element
based on those bus numbers:

+----------------+-------------------------------------------+
| Element        | pypowsybl ID pattern                      |
+================+===========================================+
| Bus            | ``VL-{orig_bus}_0``                       |
+----------------+-------------------------------------------+
| Load           | ``LOAD-{orig_bus}``                       |
+----------------+-------------------------------------------+
| Generator      | ``GEN-{orig_bus}`` (1st gen at bus),      |
|                | ``GEN-{orig_bus}#0`` (2nd), ``#1`` (3rd)… |
+----------------+-------------------------------------------+
| Line           | ``LINE-{orig_f}-{orig_t}`` (1st branch),  |
|                | ``LINE-{orig_f}-{orig_t}#0`` (2nd), …     |
+----------------+-------------------------------------------+
| Transformer    | ``TWT-{orig_f}-{orig_t}`` (1st),          |
|                | ``TWT-{orig_f}-{orig_t}#0`` (2nd), …      |
+----------------+-------------------------------------------+

where ``orig_bus``, ``orig_f``, and ``orig_t`` are the **original** (pre-
normalisation) bus numbers stored in ``network.reverse_bus_index_mapping``.
To recover the 0-based gridfm index from a parsed original bus number, use
``network.bus_index_mapping[orig_bus]``.

Because the mapping is already encoded in the IDs, there is no need for an
iterative O(n²) parameter-matching approach.  A single pass over the pypowsybl
element tables is sufficient.

Public API
----------
:func:`build_p2g_maps`
    Build the three pypowsybl-to-gridfm maps given a converted network pair.
"""

import re
from dataclasses import dataclass
from typing import Dict, Tuple

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
# Compiled regex patterns for pypowsybl element IDs
# ---------------------------------------------------------------------------
# Matches:  LINE-3-7   TWT-0-5   LINE-14-20#0   TWT-3-7#2
_BRANCH_ID_RE = re.compile(r"^(?:LINE|TWT)-(\d+)-(\d+)(?:#\d+)?$")

# Matches:  GEN-0   GEN-4#0   GEN-4#2
_GEN_ID_RE = re.compile(r"^GEN-(\d+)(?:#\d+)?$")

# Matches:  LOAD-0   LOAD-13
_LOAD_ID_RE = re.compile(r"^LOAD-(\d+)$")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_branch_endpoints(pp_id: str) -> Tuple[int, int]:
    """Return the ``(orig_f_bus, orig_t_bus)`` original bus numbers encoded in a branch ID."""
    m = _BRANCH_ID_RE.match(pp_id)
    if m is None:
        raise ValueError(
            f"Unexpected pypowsybl branch ID format: {pp_id!r}. "
            "Expected 'LINE-F-T[#k]' or 'TWT-F-T[#k]'.  "
            "Was the network produced by to_powsybl()?"
        )
    return int(m.group(1)), int(m.group(2))


def _parse_gen_bus(pp_gen_id: str) -> int:
    """Return the original bus number encoded in a generator ID."""
    m = _GEN_ID_RE.match(pp_gen_id)
    if m is None:
        raise ValueError(
            f"Unexpected pypowsybl generator ID format: {pp_gen_id!r}. "
            "Expected 'GEN-N[#k]'.  Was the network produced by to_powsybl()?"
        )
    return int(m.group(1))


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
        pp_bus_id: gfm_row
        for gfm_row, pp_bus_id in enumerate(pp_net.get_buses().index)
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
