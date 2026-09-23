"""Conversion of a gridfm_datakit Network to a lightsim2grid LSGrid."""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from gridfm_datakit.network import Network
from gridfm_datakit.utils.idx_brch import (
    BR_B,
    BR_R,
    BR_STATUS,
    BR_X,
    F_BUS,
    SHIFT,
    T_BUS,
    TAP,
)
from gridfm_datakit.utils.idx_bus import BS, BUS_I, BUS_TYPE, GS, PD, QD, REF, VA, VM
from gridfm_datakit.utils.idx_gen import GEN_BUS, GEN_STATUS, PG, VG

from .api import check_lightsim2grid_available, lightsim2grid_network
from .mapping import MappingL2G, build_l2g_maps


@dataclass
class ConvertedNetwork:
    """A lightsim2grid LSGrid together with its maps to the gridfm Network."""

    ls_net: Any  # lightsim2grid.network.LSGrid
    mapping_l2g: MappingL2G
    # what was last pushed to ls_net (see _snapshot), to push only the differences next time
    state: Dict[str, Any] = field(default_factory=dict, repr=False)


def to_lightsim2grid(net: Network) -> ConvertedNetwork:
    """Build a lightsim2grid LSGrid from the *current* state of ``net``.

    Loads, generator set points, statuses and branch parameters are all read
    from ``net.buses`` / ``net.gens`` / ``net.branches``, so the result is
    valid for one perturbed copy of the network only. Use
    :func:`update_lightsim2grid` to re-synchronise it with another one.

    Args:
        net: The network to convert.

    Returns:
        The LSGrid, its index maps to ``net`` and a snapshot of what it was built from.

    Raises:
        ImportError: If lightsim2grid is not available.
    """
    check_lightsim2grid_available()
    mpc = {
        "bus": net.buses,
        "gen": net.gens,
        "branch": net.branches,
        "baseMVA": float(net.baseMVA),
    }
    with warnings.catch_warnings():
        # e.g. BASE_KV == 0 everywhere: voltages are then reported in pu, which is what we want
        warnings.simplefilter("ignore")
        ls_net = lightsim2grid_network.init_from_matpower(mpc)

    mapping = build_l2g_maps(net)
    assert len(ls_net.get_lines()) == len(mapping.line_rows)
    assert len(ls_net.get_trafos()) == len(mapping.trafo_rows)
    assert len(ls_net.get_generators()) == net.gens.shape[0]
    assert ls_net.total_bus() == net.buses.shape[0]
    return ConvertedNetwork(
        ls_net=ls_net,
        mapping_l2g=mapping,
        state=_snapshot(net),
    )


def _structure(net: Network) -> tuple:
    """What can only be changed by rebuilding the LSGrid.

    Args:
        net: The network to read.

    Returns:
        Copies of the bus types and shunts, of the branch ends, taps and shifts, of the
        generator buses, of which buses have a load, and the in-service slack generators.
    """
    bus_type = np.zeros(net.buses.shape[0])
    bus_type[net.buses[:, BUS_I].astype(int)] = net.buses[:, BUS_TYPE]
    slack = np.flatnonzero(
        (net.gens[:, GEN_STATUS] > 0)
        & (bus_type[net.gens[:, GEN_BUS].astype(int)] == REF),
    )
    return (
        net.buses[:, [BUS_I, BUS_TYPE, GS, BS]].copy(),
        net.branches[:, [F_BUS, T_BUS, TAP, SHIFT]].copy(),
        net.gens[:, GEN_BUS].copy(),
        ((net.buses[:, PD] != 0) | (net.buses[:, QD] != 0)),  # which buses have a load
        slack,
    )


def _snapshot(net: Network) -> Dict[str, Any]:
    """Copy of everything of ``net`` that the LSGrid was built from.

    ``structure`` holds what can only be changed by rebuilding the LSGrid; the
    other entries are what :func:`update_lightsim2grid` can push in place.

    Args:
        net: The network the LSGrid was built from.

    Returns:
        The snapshot, as a dict of arrays.
    """
    structure = _structure(net)
    has_load = structure[3]
    return {
        "structure": structure,
        "load_rows": np.flatnonzero(has_load),
        "branch": net.branches[:, [BR_R, BR_X, BR_B, BR_STATUS]].copy(),
        "gen": net.gens[:, [GEN_STATUS, PG, VG]].copy(),
        "load": net.buses[has_load][:, [PD, QD]].copy(),
    }


def _same_structure(net: Network, state: Dict[str, Any]) -> bool:
    """Whether the LSGrid of ``state`` can be updated in place to match ``net``.

    Args:
        net: The network the LSGrid should match.
        state: The snapshot of the LSGrid (see :func:`_snapshot`).

    Returns:
        True if nothing that requires rebuilding the LSGrid changed.
    """
    return all(
        np.array_equal(a, b) for a, b in zip(_structure(net), state["structure"])
    )


def update_lightsim2grid(
    net: Network,
    converted: Optional[ConvertedNetwork] = None,
) -> ConvertedNetwork:
    """Return a lightsim2grid model in sync with ``net``.

    With no ``converted`` the LSGrid is built from scratch. Otherwise the
    LSGrid of ``converted`` is updated *in place* with what changed since it
    was last synchronised (branch parameters and statuses, generator statuses
    and set points, loads), which lets lightsim2grid keep its solver caches.
    It is rebuilt only if something it cannot update changed (topology, taps
    and shifts, shunts, bus types, which buses have a load, the slack
    generators). ``converted`` is modified and returned.

    Args:
        net: The network the LSGrid has to match.
        converted: The result of a previous call, or None to build the LSGrid.

    Returns:
        ``converted`` updated in place, or a new :class:`ConvertedNetwork` if the LSGrid
        had to be built or rebuilt.
    """
    if converted is None or not _same_structure(net, converted.state):
        return to_lightsim2grid(net)

    ls_net, mapping, old = converted.ls_net, converted.mapping_l2g, converted.state
    try:
        _update_in_place(ls_net, net, mapping, old)
    except AttributeError:
        # a released lightsim2grid without update_powerlines_parameters /
        # update_trafos_parameters: half-updated, nothing about the LSGrid can
        # be trusted anymore, fall back to rebuilding it from scratch
        return to_lightsim2grid(net)
    return converted


# lightsim2grid's change_* setters ignore a change of at most this (BaseConstants::_tol_equal_float)
_LS_TOL_EQUAL_FLOAT = 1e-7


def _set_exact(setter: Any, el_id: int, old: float, new: float) -> None:
    """Call ``setter(el_id, new)`` and make sure that ``new`` is what gets stored.

    A change smaller than ``_LS_TOL_EQUAL_FLOAT`` is silently dropped by the
    lightsim2grid setters, which would leave the LSGrid up to that far from
    ``net``, and make the data depend on which perturbations came before. Such a
    change is applied through a detour, in two steps that are both large enough.

    Args:
        setter: A lightsim2grid ``change_*`` method, taking an element id and a value.
        el_id: Id of the element, in the lightsim2grid ordering.
        old: Value currently stored in the LSGrid.
        new: Value to store.
    """
    if abs(new - old) <= _LS_TOL_EQUAL_FLOAT:
        setter(el_id, float(new) + 10 * _LS_TOL_EQUAL_FLOAT)
    setter(el_id, float(new))


def _update_in_place(
    ls_net: Any,
    net: Network,
    mapping: MappingL2G,
    old: Dict[str, Any],
) -> None:
    """Push into the LSGrid what changed in ``net`` since it was last synchronised.

    Only the parameters that :func:`_same_structure` allows to change are handled:
    branch impedances, admittances and statuses, generator statuses and set points,
    and loads. ``old`` is updated with the new values at the end, and is left
    untouched if an exception is raised before that (the caller then rebuilds).

    Args:
        ls_net: The lightsim2grid LSGrid to update.
        net: The network the LSGrid has to match.
        mapping: Index maps between ``net`` and ``ls_net``.
        old: Snapshot of what was last pushed to ``ls_net``, updated in place.
    """
    branch = net.branches[:, [BR_R, BR_X, BR_B, BR_STATUS]]

    # series impedance and charging admittance (the LSGrid has no per-element setter).
    # The update method is looked up only when needed: a released lightsim2grid does not
    # have it, and only a change of these parameters has to fall back to a rebuild.
    for rows, update_name, split_b in (
        (mapping.line_rows, "update_powerlines_parameters", True),
        (mapping.trafo_rows, "update_trafos_parameters", False),
    ):
        if rows.size and not np.array_equal(branch[rows, :3], old["branch"][rows, :3]):
            update = getattr(ls_net, update_name)
            r, x, b = branch[rows, 0], branch[rows, 1], 1j * branch[rows, 2]
            if split_b:  # a line: the charging is split in two halves
                update(r, x, b / 2, b / 2)
            else:  # a transformer: the total charging is given
                update(r, x, b)

    # branch statuses
    for rows, deactivate, reactivate in (
        (mapping.line_rows, ls_net.deactivate_powerline, ls_net.reactivate_powerline),
        (mapping.trafo_rows, ls_net.deactivate_trafo, ls_net.reactivate_trafo),
    ):
        for k in np.flatnonzero(branch[rows, 3] != old["branch"][rows, 3]):
            (reactivate if branch[rows[k], 3] != 0 else deactivate)(int(k))

    # generators
    gen = net.gens[:, [GEN_STATUS, PG, VG]]
    for i in np.flatnonzero(gen[:, 0] != old["gen"][:, 0]):
        (ls_net.reactivate_gen if gen[i, 0] > 0 else ls_net.deactivate_gen)(int(i))
    for i in np.flatnonzero(gen[:, 1] != old["gen"][:, 1]):
        _set_exact(ls_net.change_p_gen, int(i), old["gen"][i, 1], gen[i, 1])
    for i in np.flatnonzero(gen[:, 2] != old["gen"][:, 2]):
        _set_exact(ls_net.change_v_gen, int(i), old["gen"][i, 2], gen[i, 2])

    # loads (one per bus that has one)
    load = net.buses[old["load_rows"]][:, [PD, QD]]
    for k in np.flatnonzero(load[:, 0] != old["load"][:, 0]):
        _set_exact(ls_net.change_p_load, int(k), old["load"][k, 0], load[k, 0])
    for k in np.flatnonzero(load[:, 1] != old["load"][:, 1]):
        _set_exact(ls_net.change_q_load, int(k), old["load"][k, 1], load[k, 1])

    old["branch"] = branch.copy()
    old["gen"] = gen.copy()
    old["load"] = load.copy()


def initial_voltage(net: Network) -> np.ndarray:
    """Complex initial voltage (one entry per lightsim2grid bus) from ``net.buses``.

    Args:
        net: The network, whose bus voltage magnitudes (pu) and angles (degrees) are used.

    Returns:
        The complex voltages in pu, in the order of ``net.buses``.
    """
    return (net.buses[:, VM] * np.exp(1j * np.deg2rad(net.buses[:, VA]))).astype(
        complex,
    )
