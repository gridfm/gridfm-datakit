import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import scipy.io

from gridfm_datakit.network import Network
from gridfm_datakit.utils.idx_brch import F_BUS, T_BUS
from gridfm_datakit.utils.idx_gen import GEN_BUS, MBASE

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
    reverse_bus_index_mapping : Dict[int, int]
        ``{gfm_bus_index: original_bus_number}``. Power-flow bus results are
        keyed by that original number, the same way PowerModels keys them.
    """

    bus: Dict[str, float]
    branch: Dict[str, int]
    gen: Dict[str, int]
    reverse_bus_index_mapping: Dict[int, int]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def build_p2g_maps(
    network: Network,
    pp_net,
) -> MappingP2G:
    """Build pypowsybl-to-gridfm index maps for a network read by ``from_powsybl``.

    :func:`~gridfm_datakit.powsybl.convert.from_powsybl` builds the gridfm
    tables from pypowsybl's MATPOWER export.  The export writes buses and
    branches (lines, then 2-winding transformers) in the order pypowsybl lists
    them, so the bus and branch maps pair the i-th pypowsybl element with
    gridfm row i.  It writes generators grouped by bus number and leaves some
    out, so the generator map follows the rows of the export, see
    :func:`_tagged_export`.  A generator the export leaves out has no
    entry.  A network built by
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
    # 1. Gen map - the rows of the MATPOWER export
    # -------------------------------------------------------------------------
    mpc, gen_ids = _tagged_export(pp_net)
    map_gen_p2g: Dict[str, int] = {
        gen_ids[int(tag) - 1]: gfm_row
        for gfm_row, tag in enumerate(mpc["gen"][:, MBASE])
        if tag >= 1
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

    return MappingP2G(
        bus=map_bus_p2g,
        branch=map_branch_p2g,
        gen=map_gen_p2g,
        reverse_bus_index_mapping={
            int(row): int(orig)
            for row, orig in network.reverse_bus_index_mapping.items()
        },
    )


def build_p2g_maps_by_bus_number(
    network: Network,
    pp_net,
) -> MappingP2G:
    """Build pypowsybl-to-gridfm index maps for a network built by ``to_powsybl``.

    ``to_powsybl`` names each bus-breaker bus ``BUS-<n>`` after its original
    bus number.  pypowsybl lists buses by voltage level, and buses joined by a
    transformer share one, so position does not give the gridfm row.  Branches
    are matched by original from and to buses, parallel branches in table
    order, because pypowsybl lists lines before transformers.  Generators are
    named ``GEN-<n>``, then ``GEN-<n>#0``, ``GEN-<n>#1`` for the next ones on
    bus ``n`` in table order, and pypowsybl lists them by bus.

    Args:
        network: The gridfm_datakit Network passed to ``to_powsybl``.
        pp_net: The pypowsybl network ``to_powsybl(network)`` built.

    Returns:
        MappingP2G with the bus, branch and generator maps read from the
        original bus numbers.

    Raises:
        ValueError: If a pypowsybl bus groups several gridfm buses or has none,
            a pypowsybl branch has no gridfm branch with the same ends, or a
            pypowsybl generator has no gridfm generator on its bus.
    """
    check_powsybl_available()
    maps = MappingP2G(
        bus={},
        branch={},
        gen=_gen_rows_by_id(network, pp_net),
        reverse_bus_index_mapping={
            int(row): int(orig)
            for row, orig in network.reverse_bus_index_mapping.items()
        },
    )
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

    rev = network.reverse_bus_index_mapping
    rows_by_ends: Dict[Tuple[int, int], List[int]] = {}
    for row, (f_bus, t_bus) in enumerate(network.branches[:, [F_BUS, T_BUS]]):
        rows_by_ends.setdefault((rev[int(f_bus)], rev[int(t_bus)]), []).append(row)
    ends = ["bus_breaker_bus1_id", "bus_breaker_bus2_id"]
    for df in (
        pp_net.get_lines(attributes=ends),
        pp_net.get_2_windings_transformers(attributes=ends),
    ):
        for pp_branch_id, bus1, bus2 in df.itertuples():
            key = (_bus_number(bus1), _bus_number(bus2))
            rows = rows_by_ends.get(key)
            if not rows:
                raise ValueError(
                    f"No gridfm branch from bus {key[0]} to bus {key[1]} "
                    f"for pypowsybl branch {pp_branch_id}",
                )
            maps.branch[pp_branch_id] = rows.pop(0)
    return maps


def _bus_number(bus_breaker_bus_id: str) -> int:
    """Return the original bus number of a ``BUS-<n>`` bus-breaker bus ID."""
    return int(bus_breaker_bus_id.rsplit("-", 1)[1])


def _gen_rows_by_id(network: Network, pp_net) -> Dict[str, int]:
    """Pair each ``GEN-<n>[#k]`` generator with its gridfm row on bus ``n``."""
    rev = network.reverse_bus_index_mapping
    rows_by_bus: Dict[int, List[int]] = {}
    for row, gfm_bus in enumerate(network.gens[:, GEN_BUS]):
        rows_by_bus.setdefault(rev[int(gfm_bus)], []).append(row)
    gen_rows: Dict[str, int] = {}
    for pp_gen_id in pp_net.get_generators().index:
        number, _, copy = pp_gen_id.removeprefix("GEN-").partition("#")
        rows = rows_by_bus.get(int(number), [])
        k = int(copy) + 1 if copy else 0
        if k >= len(rows):
            raise ValueError(
                f"No gridfm generator on bus {number} for pypowsybl generator {pp_gen_id}",
            )
        gen_rows[pp_gen_id] = rows[k]
    return gen_rows


def _tagged_export(pp_net) -> Tuple[Any, List[str]]:
    """Export *pp_net* to MATPOWER with each generator's row tag in MBASE.

    The MATPOWER export carries no generator IDs.  It writes generators by bus
    number, connected ones first, drops some, and adds rows for static VAR
    compensators and VSC stations.  Each generator's ``rated_s``, which the
    export writes as MBASE, is set to its position plus one for the export
    and restored afterwards, so a row with MBASE ``k`` describes generator
    ``k - 1`` of the returned IDs and a row with MBASE 0 no generator.

    Returns:
        The exported ``mpc`` struct and the generator IDs in tag order.
    """
    per_unit = pp_net.per_unit
    pp_net.per_unit = False
    gens = pp_net.get_generators(attributes=["rated_s"])
    ids = gens.index.tolist()
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
        mat_path = tmp.name
    try:
        if ids:
            tags = [float(k) for k in range(1, len(ids) + 1)]
            pp_net.update_generators(id=ids, rated_s=tags)
        pp_net.save(mat_path, format="MATPOWER")
        data = scipy.io.loadmat(mat_path, struct_as_record=True, squeeze_me=False)
    finally:
        if ids:
            pp_net.update_generators(id=ids, rated_s=gens["rated_s"].tolist())
        pp_net.per_unit = per_unit
        Path(mat_path).unlink()
    return data["mpc"][0, 0], ids
