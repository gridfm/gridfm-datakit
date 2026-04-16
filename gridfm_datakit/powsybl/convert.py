"""
PyPowSyBl Network Converter module.

Provides bidirectional conversion between pypowsybl and gridfm_datakit networks.

Conversion Architecture:
------------------------
from_powsybl(pp_net) → Network:
    1. Export pypowsybl network to temporary .mat file (native MATPOWER export)
    2. Load .mat file with scipy.io.loadmat
    3. Add gencost matrix (not included in pypowsybl export)
    4. Return gridfm_datakit Network

to_powsybl(network) → ConvertedNetwork:
    1. Restore original bus numbers via network.reverse_bus_index_mapping
    2. Build MATPOWER struct and save to temp .mat
    3. Load .mat with pypowsybl
    4. Build pypowsybl-to-gridfm maps (build_p2g_maps)
    5. Return ConvertedNetwork(pp_net, maps)

    Using reverse_bus_index_mapping ensures that any two networks sharing the
    same topology (e.g. a base network and its perturbations) produce identical
    pypowsybl element IDs, so maps computed on the base remain valid for all
    perturbed variants without recomputing.

Example:
--------
>>> import pypowsybl as pp
>>> from gridfm_datakit.powsybl.convert import from_powsybl, to_powsybl
>>> pp_net = pp.network.create_ieee14()
>>> gfm_net = from_powsybl(pp_net)
>>> result = to_powsybl(gfm_net)
>>> result.pp_net          # pypowsybl network
>>> result.map_bus_p2g     # bus ID → gfm index map
"""

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict

import numpy as np
import scipy.io

from gridfm_datakit.network import Network
from gridfm_datakit.utils.idx_bus import VMAX, VMIN
from gridfm_datakit.utils.idx_brch import F_BUS, T_BUS
from gridfm_datakit.utils.idx_gen import GEN_BUS
from gridfm_datakit.utils.idx_bus import BUS_I
from gridfm_datakit.utils.idx_cost import MODEL, STARTUP, SHUTDOWN, NCOST, COST, POLYNOMIAL

from .api import check_powsybl_available, pypowsybl


@dataclass
class ConversionOptions:
    """
    Configuration options for pypowsybl to gridfm_datakit conversion.

    Attributes:
        gen_costs: Generator cost coefficients. Either:
            - dict[str, tuple[float, ...]]: Per-generator costs keyed by index
            - tuple[float, ...]: Uniform costs for all generators
            Defaults to (0.0, 1.0, 0.0) for missing entries.
    """

    gen_costs: dict[str, tuple[float, ...]] | tuple[float, ...] | None = None


@dataclass
class ConvertedNetwork:
    """
    Result of converting a gridfm_datakit Network to a pypowsybl network.

    Bundles the pypowsybl network with the three pypowsybl-to-gridfm index
    maps so callers never need a separate build_p2g_maps call.  The maps are
    built once here and can be reused across all perturbed variants of the same
    base network because to_powsybl uses reverse_bus_index_mapping to produce
    consistent element IDs regardless of normalization history.

    Attributes
    ----------
    pp_net : pypowsybl.network.Network
        The converted pypowsybl network.
    map_bus_p2g : dict[str, float]
        ``{pp_bus_id: gfm_bus_index}`` — see build_p2g_maps.
    map_branch_p2g : dict[str, int]
        ``{pp_branch_id: gfm_branch_row}`` — see build_p2g_maps.
    map_gen_p2g : dict[str, int]
        ``{pp_gen_id: gfm_gen_row}`` — see build_p2g_maps.
    """

    pp_net: Any
    map_bus_p2g: Dict[str, float]
    map_branch_p2g: Dict[str, int]
    map_gen_p2g: Dict[str, int]


def _build_gencost_matrix(
    n_generators: int,
    gen_costs: dict[str, tuple[float, ...]] | tuple[float, ...] | None = None,
) -> np.ndarray:
    """
    Build the MATPOWER-style generator cost matrix.

    Creates an (n_generators x 7) matrix with polynomial cost functions.

    Columns: MODEL | STARTUP | SHUTDOWN | NCOST | c2 | c1 | c0

    Args:
        n_generators: Number of generators.
        gen_costs: Cost coefficients. Either dict[str, tuple] per-generator
                   or tuple for uniform costs. Defaults to (0.0, 1.0, 0.0).

    Returns:
        Generator cost matrix (n_generators x 7+).
    """
    DEFAULT_GEN_COSTS = (0.0, 1.0, 0.0)

    if gen_costs is None:
        gen_costs = {}

    if isinstance(gen_costs, tuple):
        uniform_costs = gen_costs
        gen_costs = {str(i): uniform_costs for i in range(n_generators)}

    max_coeffs = len(DEFAULT_GEN_COSTS)
    for costs in gen_costs.values():
        max_coeffs = max(max_coeffs, len(costs))

    gencost_matrix = np.zeros((n_generators, 4 + max_coeffs))

    for i in range(n_generators):
        costs = gen_costs.get(str(i), DEFAULT_GEN_COSTS)
        n_cost_coeffs = len(costs)

        gencost_matrix[i, MODEL] = POLYNOMIAL
        gencost_matrix[i, STARTUP] = 0.0
        gencost_matrix[i, SHUTDOWN] = 0.0
        gencost_matrix[i, NCOST] = n_cost_coeffs
        for j, coeff in enumerate(costs):
            gencost_matrix[i, COST + j] = coeff

    return gencost_matrix


def from_powsybl(
    pp_net,
    options: ConversionOptions | None = None,
) -> Network:
    """
    Convert a pypowsybl Network to a gridfm_datakit Network.

    Uses pypowsybl's native MATPOWER export, then adds gencost matrix.

    Steps:
        1. Export pypowsybl network to temp .mat file
        2. Load with scipy.io.loadmat
        3. Fix voltage limits if zero (default: 0.9-1.1 p.u.)
        4. Add gencost matrix
        5. Return gridfm_datakit Network

    Args:
        pp_net: A pypowsybl Network object.
        options: Conversion options (gen_costs). Defaults to None.

    Returns:
        A gridfm_datakit Network object.

    Raises:
        ValueError: If input is not a pypowsybl Network or has no buses.

    Example:
        >>> pp_net = pp.network.create_ieee14()
        >>> network = from_powsybl(pp_net)
    """
    check_powsybl_available()
    if not isinstance(pp_net, pypowsybl.network.Network):
        raise ValueError("Input must be a pypowsybl Network object")

    buses_df = pp_net.get_buses()
    if buses_df.empty:
        raise ValueError("pypowsybl network has no buses")

    if options is None:
        options = ConversionOptions()

    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
        mat_path = tmp.name

    try:
        pp_net.save(mat_path, format="MATPOWER")

        data = scipy.io.loadmat(mat_path, struct_as_record=True, squeeze_me=False)
        mpc_raw = data["mpc"][0, 0]

        version = str(mpc_raw["version"][0])
        baseMVA = float(mpc_raw["baseMVA"][0, 0])
        bus_matrix = mpc_raw["bus"]
        gen_matrix = mpc_raw["gen"]
        branch_matrix = mpc_raw["branch"]

    finally:
        Path(mat_path).unlink()

    for i in range(bus_matrix.shape[0]):
        if bus_matrix[i, VMAX] <= 0:
            bus_matrix[i, VMAX] = 1.1
        if bus_matrix[i, VMIN] <= 0:
            bus_matrix[i, VMIN] = 0.9

    n_generators = gen_matrix.shape[0]
    gencost_matrix = _build_gencost_matrix(n_generators, options.gen_costs)

    mpc = {
        "version": version,
        "baseMVA": baseMVA,
        "bus": bus_matrix,
        "gen": gen_matrix,
        "branch": branch_matrix,
        "gencost": gencost_matrix,
    }

    return Network(mpc)


def to_powsybl(
    network: Network,
    network_id: str = "network",
) -> ConvertedNetwork:
    """
    Convert a gridfm_datakit Network to a pypowsybl Network.

    Bus numbers in the MATPOWER intermediate file are restored from
    ``network.reverse_bus_index_mapping`` before the file is written.  This
    means any two networks that share the same topology (e.g. a base network
    and its perturbations) will always produce the same pypowsybl element IDs,
    so the maps returned here can be reused across all perturbed variants
    without recomputing.

    Args:
        network: A gridfm_datakit Network object.
        network_id: ID assigned to the created pypowsybl network.

    Returns:
        ConvertedNetwork with pp_net and the three p2g index maps.

    Example:
        >>> network = load_net_from_pglib("case14_ieee")
        >>> result = to_powsybl(network)
        >>> result.pp_net           # pypowsybl network
        >>> result.map_bus_p2g      # reusable for all perturbations of network
    """
    check_powsybl_available()

    rev = network.reverse_bus_index_mapping

    # Restore original bus numbers so element IDs are stable across
    # perturbations of the same base network.
    bus = network.buses.copy()
    bus[:, BUS_I] = [rev[int(i)] for i in bus[:, BUS_I]]

    gen = network.gens.copy()
    gen[:, GEN_BUS] = [rev[int(i)] for i in gen[:, GEN_BUS]]

    branch = network.branches.copy()
    branch[:, F_BUS] = [rev[int(i)] for i in branch[:, F_BUS]]
    branch[:, T_BUS] = [rev[int(i)] for i in branch[:, T_BUS]]

    tmp_dir = tempfile.mkdtemp()
    mat_path = Path(tmp_dir) / f"{network_id}.mat"

    try:
        mpc = {
            "version": "2",
            "baseMVA": np.array([[network.baseMVA]]),
            "bus": bus,
            "gen": gen,
            "branch": branch,
        }
        if hasattr(network, "gencosts") and network.gencosts is not None:
            mpc["gencost"] = network.gencosts

        scipy.io.savemat(mat_path, {"mpc": mpc})
        pp_net = pypowsybl.network.load(
            str(mat_path),
            {"matpower.import.ignore-base-voltage": "false"},
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    from .mapping import build_p2g_maps
    map_bus_p2g, map_branch_p2g, map_gen_p2g = build_p2g_maps(network, pp_net)

    return ConvertedNetwork(
        pp_net=pp_net,
        map_bus_p2g=map_bus_p2g,
        map_branch_p2g=map_branch_p2g,
        map_gen_p2g=map_gen_p2g,
    )
