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

to_powsybl(network) → pypowsybl.network.Network:
    1. Save Network to temp .m file
    2. Convert .m to .mat via scipy
    3. Load .mat with pypowsybl
    4. Return pypowsybl Network

Example:
--------
>>> import pypowsybl as pp
>>> from gridfm_datakit.powsybl.convert import from_powsybl, to_powsybl
>>> pp_net = pp.network.create_ieee14()
>>> gfm_net = from_powsybl(pp_net)
>>> pp_net_roundtrip = to_powsybl(gfm_net)
"""

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile

import numpy as np
import scipy.io

from gridfm_datakit.network import Network
from gridfm_datakit.utils.idx_bus import VMAX, VMIN
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
    # Default cost function: linear cost of $1/MWh with no fixed cost
    # Coefficients [0.0, 1.0, 0.0] = 0*P^2 + 1*P + 0 = P ($/hr)
    DEFAULT_GEN_COSTS = (0.0, 1.0, 0.0)

    if gen_costs is None:
        gen_costs = {}

    # Handle case where gen_costs is a tuple (same cost for all generators)
    if isinstance(gen_costs, tuple):
        uniform_costs = gen_costs
        gen_costs = {str(i): uniform_costs for i in range(n_generators)}

    # Determine max number of cost coefficients across all generators
    # This ensures all rows have the same width for the numpy array
    max_coeffs = len(DEFAULT_GEN_COSTS)
    for costs in gen_costs.values():
        max_coeffs = max(max_coeffs, len(costs))

    # Matrix width: 4 header columns + variable number of cost coefficients
    gencost_matrix = np.zeros((n_generators, 4 + max_coeffs))

    for i in range(n_generators):
        # Get per-generator costs or use default
        # Generator indices are stored as strings (e.g., "0", "1", "2")
        costs = gen_costs.get(str(i), DEFAULT_GEN_COSTS)
        n_cost_coeffs = len(costs)

        # MODEL = 2 means polynomial cost function
        gencost_matrix[i, MODEL] = POLYNOMIAL
        # Startup and shutdown costs (not used in continuous OPF)
        gencost_matrix[i, STARTUP] = 0.0
        gencost_matrix[i, SHUTDOWN] = 0.0
        # Number of polynomial coefficients
        gencost_matrix[i, NCOST] = n_cost_coeffs
        # Cost coefficients in descending order: c_n, c_{n-1}, ..., c_1, c_0
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

    # Check for empty network before attempting export
    buses_df = pp_net.get_buses()
    if buses_df.empty:
        raise ValueError("pypowsybl network has no buses")

    if options is None:
        options = ConversionOptions()

    # -------------------------------------------------------------------------
    # Step 1: Export pypowsybl network to temporary .mat file
    # -------------------------------------------------------------------------
    # pypowsybl can export to MATPOWER .mat format natively, which gives us
    # properly formatted bus, gen, and branch matrices without manual conversion
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
        mat_path = tmp.name

    try:
        pp_net.save(mat_path, format="MATPOWER")

        # -------------------------------------------------------------------------
        # Step 2: Load the .mat file with scipy
        # -------------------------------------------------------------------------
        # scipy.io.loadmat reads the MATPOWER mpc struct
        data = scipy.io.loadmat(mat_path, struct_as_record=True, squeeze_me=False)
        mpc_raw = data["mpc"][0, 0]

        # Extract arrays from the mpc struct
        version = str(mpc_raw["version"][0])
        baseMVA = float(mpc_raw["baseMVA"][0, 0])
        bus_matrix = mpc_raw["bus"]
        gen_matrix = mpc_raw["gen"]
        branch_matrix = mpc_raw["branch"]

    finally:
        Path(mat_path).unlink()

    # -------------------------------------------------------------------------
    # Step 2b: Fix voltage limits if pypowsybl exported 0 values
    # -------------------------------------------------------------------------
    # pypowsybl MATPOWER export sometimes sets VMAX/VMIN to 0, which is invalid
    # Set default values: VMIN=0.9, VMAX=1.1 (typical operational range)
    for i in range(bus_matrix.shape[0]):
        if bus_matrix[i, VMAX] <= 0:
            bus_matrix[i, VMAX] = 1.1
        if bus_matrix[i, VMIN] <= 0:
            bus_matrix[i, VMIN] = 0.9

    n_generators = gen_matrix.shape[0]

    # -------------------------------------------------------------------------
    # Step 3: Build gencost matrix with cost coefficients
    # -------------------------------------------------------------------------
    # pypowsybl's MATPOWER export doesn't include gencost, so we add it
    gencost_matrix = _build_gencost_matrix(n_generators, options.gen_costs)

    # -------------------------------------------------------------------------
    # Step 4: Assemble MATPOWER case structure
    # -------------------------------------------------------------------------
    # The mpc dict follows MATPOWER's case file convention
    mpc = {
        "version": version,       # MATPOWER case format version
        "baseMVA": baseMVA,       # System base apparent power (MW)
        "bus": bus_matrix,        # Bus data matrix (n_bus x 13)
        "gen": gen_matrix,        # Generator data matrix (n_gen x 21)
        "branch": branch_matrix,  # Branch data matrix (n_branch x 13)
        "gencost": gencost_matrix,  # Generator cost matrix (n_gen x 7+)
    }

    return Network(mpc)


def to_powsybl(
    network: Network,
    network_id: str = "network",
):
    """
    Convert a gridfm_datakit Network to a pypowsybl Network.

    Uses MATPOWER format as intermediate:
        1. Save Network to temp .m file
        2. Convert .m to .mat via scipy
        3. Load .mat with pypowsybl

    Args:
        network: A gridfm_datakit Network object.
        network_id: ID for the created pypowsybl network.

    Returns:
        A pypowsybl Network object.

    Example:
        >>> network = load_net_from_pglib("case14_ieee")
        >>> pp_net = to_powsybl(network)
    """
    check_powsybl_available()

    # Create temp directory and use network_id in filename
    # (pypowsybl uses filename as network id)
    tmp_dir = tempfile.mkdtemp()
    m_path = Path(tmp_dir) / f"{network_id}.m"
    mat_path = Path(tmp_dir) / f"{network_id}.mat"

    try:
        # Step 1: Save Network to .m file
        network.to_mpc(m_path)

        # Step 2: Convert .m to .mat (build mpc struct for scipy)
        # Use same format as to_mat_file for compatibility
        mpc = {
            "version": "2",
            "baseMVA": np.array([[network.baseMVA]]),
            "bus": network.buses,
            "gen": network.gens,
            "branch": network.branches,
        }
        if hasattr(network, "gencosts") and network.gencosts is not None:
            mpc["gencost"] = network.gencosts

        scipy.io.savemat(mat_path, {"mpc": mpc})

        # Step 3: Load .mat with pypowsybl
        pp_net = pypowsybl.network.load(mat_path, {"matpower.import.ignore-base-voltage": "false"})

    finally:
        # Clean up temp files and directory
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return pp_net
