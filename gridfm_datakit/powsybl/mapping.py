"""
ID-based O(n) mapping between gridfm_datakit and pypowsybl networks.

Background
----------
When ``to_powsybl()`` converts a gridfm :class:`~gridfm_datakit.network.Network`
to pypowsybl, the MATPOWER `.mat` file it writes is loaded by pypowsybl's
MATPOWER importer.  That importer assigns **deterministic string IDs** to every
element based on the bus indices that appear in the MATPOWER matrices:

+----------------+-------------------------------------------+
| Element        | pypowsybl ID pattern                      |
+================+===========================================+
| Bus            | ``VL-{bus_idx}_0``                        |
+----------------+-------------------------------------------+
| Load           | ``LOAD-{bus_idx}``                        |
+----------------+-------------------------------------------+
| Generator      | ``GEN-{bus_idx}`` (1st gen at bus),       |
|                | ``GEN-{bus_idx}#0`` (2nd), ``#1`` (3rd)… |
+----------------+-------------------------------------------+
| Line           | ``LINE-{f_bus}-{t_bus}`` (1st branch),    |
|                | ``LINE-{f_bus}-{t_bus}#0`` (2nd), …       |
+----------------+-------------------------------------------+
| Transformer    | ``TWT-{f_bus}-{t_bus}`` (1st),            |
|                | ``TWT-{f_bus}-{t_bus}#0`` (2nd), …        |
+----------------+-------------------------------------------+

where ``bus_idx``, ``f_bus``, and ``t_bus`` are the **0-based gridfm bus
indices** stored in the ``BUS_I``, ``F_BUS``, and ``T_BUS`` columns of the
MATPOWER matrices.

Because the mapping is already encoded in the IDs, there is no need for the
iterative O(n²) parameter-matching performed by
:class:`~gridfm_datakit.powsybl.network_mapper.NetworkMapper`.  A single pass
over the pypowsybl element tables is sufficient.

Public API
----------
:func:`build_p2g_maps`
    Build the three pypowsybl-to-gridfm maps given an already-converted pair
    of networks.

:func:`to_powsybl_with_mapping`
    Convert a gridfm Network to pypowsybl **and** return the maps in one call.
    This is the recommended entry point when the pypowsybl solver is used.
"""

import re
from collections import defaultdict
from typing import Dict, Tuple

from gridfm_datakit.network import Network
from gridfm_datakit.utils.idx_brch import F_BUS, T_BUS

from .api import check_powsybl_available
from .convert import to_powsybl


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
    """Return the ``(f_bus, t_bus)`` gridfm bus indices encoded in a branch ID.

    Parameters
    ----------
    pp_id:
        A pypowsybl branch ID such as ``'LINE-0-1'`` or ``'TWT-3-7#0'``.

    Raises
    ------
    ValueError
        If *pp_id* does not match the expected pattern.
    """
    m = _BRANCH_ID_RE.match(pp_id)
    if m is None:
        raise ValueError(
            f"Unexpected pypowsybl branch ID format: {pp_id!r}. "
            "Expected 'LINE-F-T[#k]' or 'TWT-F-T[#k]'.  "
            "Was the network produced by to_powsybl()?"
        )
    return int(m.group(1)), int(m.group(2))


def _parse_gen_bus(pp_gen_id: str) -> int:
    """Return the gridfm bus index encoded in a generator ID.

    Parameters
    ----------
    pp_gen_id:
        A pypowsybl generator ID such as ``'GEN-0'`` or ``'GEN-4#1'``.

    Raises
    ------
    ValueError
        If *pp_gen_id* does not match the expected pattern.
    """
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
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, int]]:
    """Build pypowsybl-to-gridfm ID maps in O(n) by parsing pypowsybl element IDs.

    When a gridfm :class:`~gridfm_datakit.network.Network` is converted to
    pypowsybl via :func:`~gridfm_datakit.powsybl.convert.to_powsybl`, pypowsybl
    encodes the original gridfm bus indices directly in all element IDs (see
    module docstring for the full naming convention).  This function reads those
    IDs and builds the three reverse maps needed to translate pypowsybl power
    flow results back into gridfm index space.

    The maps can be built once on the **base network** and then reused across
    all perturbed scenarios, because perturbations (load, generation, admittance,
    topology) preserve element identity and row ordering.

    Parameters
    ----------
    network:
        The gridfm_datakit Network that was passed to ``to_powsybl()`` to
        produce *pp_net*.  Used to build the ``(F_BUS, T_BUS) → row-index``
        lookup for branches and to validate completeness.
    pp_net:
        The pypowsybl network produced by ``to_powsybl(network)``.

    Returns
    -------
    map_bus_p2g : Dict[str, float]
        ``{pp_bus_id: gfm_bus_index}`` — pypowsybl bus string ID to the
        0-based gridfm bus index (the value stored in the ``BUS_I`` column,
        returned as ``float`` to match the convention expected by
        :func:`~gridfm_datakit.powsybl.preprocess_pf_res.preprocess_pp_pf_res`).
    map_branch_p2g : Dict[str, int]
        ``{pp_branch_id: gfm_branch_row}`` — pypowsybl branch string ID to the
        0-based row index in ``network.branches``.
    map_gen_p2g : Dict[str, int]
        ``{pp_gen_id: gfm_gen_row}`` — pypowsybl generator string ID to the
        0-based row index in ``network.gens``.

    Raises
    ------
    ValueError
        If any pypowsybl element ID does not match the expected naming pattern,
        if a ``(F_BUS, T_BUS)`` pair from a branch ID is not found in the gridfm
        branch matrix, or if one or more pypowsybl buses cannot be assigned a
        gridfm index (isolated bus with no adjacent branch, generator, or load).

    Notes
    -----
    **Generator ordering assumption**
        pypowsybl preserves the row ordering of the MATPOWER gen matrix when
        assigning generator IDs.  Therefore ``enumerate(pp_net.get_generators()
        .index)`` yields generators in the same order as ``network.gens`` rows,
        making the gen map a simple O(n) enumeration.

    **Parallel branch ordering**
        For parallel branches sharing the same ``(F_BUS, T_BUS)``, pypowsybl
        names the first branch ``LINE-F-T`` (no suffix) and subsequent ones
        ``LINE-F-T#0``, ``LINE-F-T#1``, etc.  This naming order matches the row
        order of parallel branches in ``network.branches``, so a per-pair counter
        correctly identifies the right gridfm row.

    Examples
    --------
    >>> from gridfm_datakit.network import load_net_from_pglib
    >>> from gridfm_datakit.powsybl.convert import to_powsybl
    >>> from gridfm_datakit.powsybl.mapping import build_p2g_maps
    >>>
    >>> net = load_net_from_pglib("case14_ieee")
    >>> pp_net = to_powsybl(net)
    >>> map_bus_p2g, map_branch_p2g, map_gen_p2g = build_p2g_maps(net, pp_net)
    """
    check_powsybl_available()

    # -------------------------------------------------------------------------
    # 1. Gen map — direct enumeration (row order is preserved by pypowsybl)
    # -------------------------------------------------------------------------
    # pypowsybl writes MATPOWER gen rows in the same order they were loaded,
    # so the k-th entry in get_generators() corresponds to row k in network.gens.
    map_gen_p2g: Dict[str, int] = {
        pp_gen_id: gfm_row
        for gfm_row, pp_gen_id in enumerate(pp_net.get_generators().index)
    }

    # -------------------------------------------------------------------------
    # 2. Branch map — parse IDs; use a per-(f,t) counter for parallel branches
    # -------------------------------------------------------------------------
    # Build a lookup from gfm: (f_bus_idx, t_bus_idx) → [row indices in order].
    # This handles parallel branches (multiple rows with the same F_BUS, T_BUS).
    gfm_branch_by_endpoints: Dict[Tuple[int, int], list] = defaultdict(list)
    for row in range(network.branches.shape[0]):
        f = int(network.branches[row, F_BUS])
        t = int(network.branches[row, T_BUS])
        gfm_branch_by_endpoints[(f, t)].append(row)

    # Counter: how many pp branches have already been assigned for each (f, t).
    # pypowsybl names them  LINE-f-t  (1st),  LINE-f-t#0  (2nd),  LINE-f-t#1  (3rd)…
    # get_branches() returns them in this lexicographic order, so incrementing
    # the counter per (f, t) pair correctly tracks the k-th parallel branch.
    parallel_counter: Dict[Tuple[int, int], int] = defaultdict(int)

    map_branch_p2g: Dict[str, int] = {}
    for pp_branch_id in pp_net.get_branches().index:
        f, t = _parse_branch_endpoints(pp_branch_id)
        endpoint_pair = (f, t)
        occurrence = parallel_counter[endpoint_pair]
        candidates = gfm_branch_by_endpoints.get(endpoint_pair)
        if candidates is None or occurrence >= len(candidates):
            n_found = len(candidates) if candidates else 0
            raise ValueError(
                f"Branch ID {pp_branch_id!r} encodes endpoint pair {endpoint_pair}, "
                f"but only {n_found} gridfm branch(es) share that (F_BUS, T_BUS) pair. "
                "Ensure pp_net was produced by to_powsybl() from the same network."
            )
        map_branch_p2g[pp_branch_id] = candidates[occurrence]
        parallel_counter[endpoint_pair] += 1

    # -------------------------------------------------------------------------
    # 3. Bus map — derived from branch bus1_id/bus2_id; fallback to gen/load IDs
    # -------------------------------------------------------------------------
    # Lines and transformers are queried separately so we can call
    # get_lines() / get_2_windings_transformers() which always expose
    # bus1_id and bus2_id, even in older pypowsybl versions where
    # get_branches() may not.
    map_bus_p2g: Dict[str, float] = {}

    # Primary source: every connected bus appears as an endpoint of at least
    # one branch.  The branch ID encodes the exact gridfm bus indices.
    for df in (pp_net.get_lines(), pp_net.get_2_windings_transformers()):
        for pp_id, row in df.iterrows():
            f, t = _parse_branch_endpoints(pp_id)
            map_bus_p2g[row["bus1_id"]] = float(f)
            map_bus_p2g[row["bus2_id"]] = float(t)

    # First fallback: isolated buses that have a generator but no branch.
    # Generator IDs encode the bus index directly.
    for pp_gen_id, row in pp_net.get_generators().iterrows():
        bus_pp_id = row["bus_id"]
        if bus_pp_id not in map_bus_p2g:
            map_bus_p2g[bus_pp_id] = float(_parse_gen_bus(pp_gen_id))

    # Second fallback: isolated buses that have a load but no branch and no gen.
    for pp_load_id, row in pp_net.get_loads().iterrows():
        bus_pp_id = row["bus_id"]
        if bus_pp_id not in map_bus_p2g:
            m = _LOAD_ID_RE.match(pp_load_id)
            if m is None:
                raise ValueError(
                    f"Unexpected pypowsybl load ID format: {pp_load_id!r}. "
                    "Expected 'LOAD-N'.  Was the network produced by to_powsybl()?"
                )
            map_bus_p2g[bus_pp_id] = float(int(m.group(1)))

    # Sanity check: every pypowsybl bus must be covered.
    pp_bus_ids = set(pp_net.get_buses().index)
    missing = pp_bus_ids - set(map_bus_p2g)
    if missing:
        raise ValueError(
            f"Could not determine the gridfm bus index for pypowsybl bus(es): {missing}. "
            "These buses are isolated (no adjacent branch, generator, or load). "
            "Ensure pp_net was produced by to_powsybl() from the same network."
        )

    return map_bus_p2g, map_branch_p2g, map_gen_p2g


def to_powsybl_with_mapping(
    network: Network,
    network_id: str = "network",
) -> Tuple[object, Dict[str, float], Dict[str, int], Dict[str, int]]:
    """Convert a gridfm Network to pypowsybl and build p2g ID maps in one call.

    This is the recommended entry point when the pypowsybl solver is used.
    Call it **once** on the base network at setup time; the returned maps can
    be reused for every perturbed scenario without recomputing the mapping.

    Parameters
    ----------
    network:
        The gridfm_datakit Network to convert.
    network_id:
        ID assigned to the created pypowsybl network (default: ``"network"``).

    Returns
    -------
    pp_net : pypowsybl.network.Network
        The converted pypowsybl network.
    map_bus_p2g : Dict[str, float]
        See :func:`build_p2g_maps`.
    map_branch_p2g : Dict[str, int]
        See :func:`build_p2g_maps`.
    map_gen_p2g : Dict[str, int]
        See :func:`build_p2g_maps`.

    Examples
    --------
    >>> from gridfm_datakit.network import load_net_from_pglib
    >>> from gridfm_datakit.powsybl.mapping import to_powsybl_with_mapping
    >>>
    >>> net = load_net_from_pglib("case14_ieee")
    >>> pp_net, map_bus_p2g, map_branch_p2g, map_gen_p2g = to_powsybl_with_mapping(net)
    >>>
    >>> # Reuse maps for every perturbed scenario:
    >>> # pp_perturbed = to_powsybl(perturbed_net)
    >>> # results = preprocess_pp_pf_res(
    >>> #     pp_perturbed, solve_time, metadata,
    >>> #     map_bus_p2g, map_branch_p2g, map_gen_p2g
    >>> # )
    """
    pp_net = to_powsybl(network, network_id=network_id)
    map_bus_p2g, map_branch_p2g, map_gen_p2g = build_p2g_maps(network, pp_net)
    return pp_net, map_bus_p2g, map_branch_p2g, map_gen_p2g
