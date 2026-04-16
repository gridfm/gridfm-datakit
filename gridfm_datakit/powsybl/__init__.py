"""
PowSyBl integration module for gridfm_datakit.

This module provides the bridge between the pypowsybl power system library and
gridfm_datakit's internal Network representation.  It exposes two primary entry
points for users:

* :func:`load_net` — load a network **from a file** into both pypowsybl and
  gridfm_datakit representations.
* :func:`convert_net` — convert an **existing** gridfm_datakit Network back to
  pypowsybl, capturing generator cost data in the returned metadata.

Why two representations?
-------------------------
pypowsybl is a rich power-system library with support for many file formats
(XIIDM, CGMES, PSS/E, UCTE, MATPOWER, …) and power-flow solvers.
gridfm_datakit's :class:`~gridfm_datakit.network.Network` is a compact
MATPOWER-style struct used internally for data generation (perturbations,
OPF/PF solving, dataset saving).  The two representations are complementary:
pypowsybl is the I/O and solver layer; gridfm_datakit is the data-generation
layer.

Generator costs
---------------
pypowsybl **cannot store or export generator cost functions**.  For this
reason, whenever a network is loaded via :func:`load_net`, the returned
gridfm_datakit Network is given **default** cost coefficients
``(c2=0, c1=1, c0=0)`` — a linear cost of $1/MWh with no fixed or quadratic
component.  These are intentionally neutral: they will not distort OPF results
but signal clearly that real costs have not been provided.

When the user later calls :func:`convert_net` on a Network that was built
through the generation pipeline, the generator costs currently stored in the
Network's ``gencost`` matrix are extracted and returned inside the
:class:`NetworkMetadata` object.  The caller is responsible for persisting
those costs separately if they need to survive a round-trip through pypowsybl.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

from gridfm_datakit.network import Network, load_net_from_file
from gridfm_datakit.utils.idx_cost import MODEL, NCOST, COST, POLYNOMIAL

from .api import check_powsybl_available, pypowsybl
from .convert import from_powsybl, to_powsybl, ConversionOptions
from .mapping import build_p2g_maps, to_powsybl_with_mapping


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

@dataclass
class NetworkMetadata:
    """
    Metadata that cannot be represented inside a pypowsybl network object.

    pypowsybl is a powerful library for modelling and simulating power systems,
    but it deliberately omits economic data such as generator cost functions.
    This dataclass is the designated holder for any such information that must
    travel alongside the pypowsybl network.

    Attributes
    ----------
    gen_costs : dict[str, tuple[float, ...]]
        Generator cost coefficients, one entry per generator.

        Keys are **string representations of the 0-based row index** of the
        generator in the gridfm_datakit Network's ``gens`` matrix (e.g.
        ``"0"``, ``"1"``, ``"2"``).

        Values are tuples of polynomial coefficients in **descending order of
        degree**: ``(c2, c1, c0)`` for a quadratic cost
        ``c2·P² + c1·P + c0``.  Fewer coefficients are allowed for lower-
        degree polynomials (e.g. ``(c1, c0)`` for linear).

        This dict is **empty** when a network is loaded via :func:`load_net`
        because pypowsybl does not carry cost data in the file formats it
        reads.  It is populated by :func:`convert_net` when converting a
        gridfm_datakit Network back to pypowsybl.

    extra : dict[str, Any]
        Free-form storage for any other metadata a caller wishes to attach.
        gridfm_datakit does not read or write this field itself.
    """

    gen_costs: Dict[str, Tuple[float, ...]] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedNetwork:
    """
    Container that bundles all representations of the same power network.

    Holding both representations together avoids the need to pass three
    separate objects through the pipeline and makes it explicit which
    pypowsybl network corresponds to which gridfm_datakit Network.

    Attributes
    ----------
    pp_net : pypowsybl.network.Network
        The pypowsybl network object.  This is the authoritative source for
        topology and physical parameters as seen by pypowsybl.
    gfm_net : Network
        The gridfm_datakit Network derived from (or used to produce) ``pp_net``.
        This is the object consumed by the data-generation pipeline.
    metadata : NetworkMetadata
        Data that pypowsybl cannot represent — primarily generator cost
        coefficients.  See :class:`NetworkMetadata` for details.
    """

    pp_net: Any          # pypowsybl.network.Network (typed as Any to avoid
                         # hard import when pypowsybl is not installed)
    gfm_net: Network
    metadata: NetworkMetadata


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_gen_costs(network: Network) -> Dict[str, Tuple[float, ...]]:
    """
    Extract polynomial generator cost coefficients from a gridfm_datakit Network.

    The gridfm_datakit Network stores generator costs as a MATPOWER-style
    ``gencost`` matrix.  Each row corresponds to one generator and has the
    layout::

        col 0  MODEL   — cost model type (2 = polynomial)
        col 1  STARTUP — startup cost in $/hr (not extracted)
        col 2  SHUTDOWN— shutdown cost in $/hr (not extracted)
        col 3  NCOST   — number of polynomial coefficients that follow
        col 4+ COST    — coefficients in descending degree order

    This function reads rows with ``MODEL == POLYNOMIAL`` and converts them
    to a plain Python dict so they can live in :class:`NetworkMetadata`.

    Parameters
    ----------
    network : Network
        The gridfm_datakit Network whose ``gencosts`` attribute is read.

    Returns
    -------
    dict[str, tuple[float, ...]]
        ``{"0": (c2, c1, c0), "1": ..., ...}`` — one entry per generator
        that has a polynomial cost function.  Generators with non-polynomial
        models (e.g. piecewise-linear) are silently skipped; callers that
        care about completeness should verify the returned dict length.
    """
    gen_costs: Dict[str, Tuple[float, ...]] = {}

    # Guard: some Network objects may not have been initialised with a gencost
    # matrix (e.g. networks created programmatically without setting gencosts).
    if not hasattr(network, "gencosts") or network.gencosts is None:
        return gen_costs

    gencost_matrix = network.gencosts

    for i in range(gencost_matrix.shape[0]):
        model = int(gencost_matrix[i, MODEL])
        ncost = int(gencost_matrix[i, NCOST])

        if model == POLYNOMIAL and ncost > 0:
            # Extract exactly ncost coefficients starting at column COST.
            # tuple() converts from numpy scalars to plain Python floats so
            # the dict can be safely serialised (e.g. to JSON).
            coeffs = tuple(float(gencost_matrix[i, COST + j]) for j in range(ncost))
            gen_costs[str(i)] = coeffs

    return gen_costs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_net(network_path: str) -> LoadedNetwork:
    """
    Load a power network from a file using pypowsybl and return both
    the pypowsybl representation and the derived gridfm_datakit Network.

    Supported file formats
    ----------------------
    All formats supported natively by pypowsybl are accepted: XIIDM, CGMES,
    PSS/E (.raw), UCTE (.uct), MATPOWER binary (.mat), and more.

    Additionally, **MATPOWER text files (.m)** are accepted even though
    pypowsybl cannot load them directly.  When a ``.m`` file is detected,
    this function converts it automatically:

    1. Load the ``.m`` file with ``gridfm_datakit.load_net_from_file`` to
       obtain a :class:`~gridfm_datakit.network.Network`.
    2. Convert that Network to pypowsybl with :func:`~.convert.to_powsybl`,
       which serialises it to a temporary MATPOWER ``.mat`` binary and loads
       it with pypowsybl.
    3. Proceed as for any other format (step 2 below).

    For all formats:

    1. pypowsybl loads the file into a pypowsybl network object.
    2. :func:`~.convert.from_powsybl` exports the pypowsybl network to a
       temporary MATPOWER ``.mat`` file, reads it back with scipy, and
       constructs a :class:`~gridfm_datakit.network.Network` — filling the
       ``gencost`` matrix with **default coefficients** ``(0.0, 1.0, 0.0)``
       because pypowsybl does not carry cost data.

    Why are gen_costs always defaulted?
    ------------------------------------
    pypowsybl does not store generator cost functions in any of the file
    formats it reads.  Even for MATPOWER ``.m`` files — which *do* contain a
    ``gencost`` block — there is no guarantee that the generator row order
    pypowsybl produces after parsing matches the row order in the ``.m`` file.
    Silently injecting costs from the file into the wrong generators would be
    worse than using neutral defaults, so this function always uses the
    default ``(0, 1, 0)`` cost and leaves ``metadata.gen_costs`` empty.

    Parameters
    ----------
    network_path : str
        Absolute or relative path to the network file.

    Returns
    -------
    LoadedNetwork
        A :class:`LoadedNetwork` with:

        * ``pp_net`` — the pypowsybl network object.
        * ``gfm_net`` — the corresponding gridfm_datakit Network with default
          generator cost coefficients.
        * ``metadata`` — a :class:`NetworkMetadata` whose ``gen_costs`` dict
          is **always empty** (see explanation above).

    Raises
    ------
    FileNotFoundError
        If the file at ``network_path`` does not exist.
    ImportError
        If pypowsybl is not installed.

    Examples
    --------
    Load an XIIDM file and run data generation:

    >>> from gridfm_datakit.powsybl import load_net
    >>> loaded = load_net("my_network.xiidm")
    >>> loaded.pp_net.get_buses()       # interact with pypowsybl
    >>> loaded.gfm_net.buses.shape      # use with gridfm data generation

    Load a MATPOWER text file (automatic .m → pypowsybl conversion):

    >>> loaded = load_net("case14.m")
    >>> loaded.gfm_net.gens.shape[0]
    5
    """
    check_powsybl_available()

    path = Path(network_path)
    if not path.is_file():
        raise FileNotFoundError(f"Network file not found: {network_path}")

    if path.suffix.lower() == ".m":
        # pypowsybl cannot load MATPOWER text (.m) files directly — it only
        # understands the binary MATPOWER (.mat) format.  We bridge the gap by
        # loading the .m file with gridfm_datakit (which uses matpowercaseframes)
        # and then converting the resulting Network object to pypowsybl via
        # to_powsybl(), which internally serialises it to a temporary .mat file.
        gfm_tmp_net = load_net_from_file(str(path))
        pp_net = to_powsybl(gfm_tmp_net)
    else:
        # For all other formats pypowsybl can handle the file directly.
        pp_net = pypowsybl.network.load(network_path)

    # Convert the pypowsybl network to a gridfm_datakit Network.
    # No gen_costs are passed, so from_powsybl injects default (0, 1, 0) costs.
    gfm_net = from_powsybl(pp_net)

    # metadata.gen_costs is intentionally left empty — see docstring.
    metadata = NetworkMetadata()

    return LoadedNetwork(pp_net=pp_net, gfm_net=gfm_net, metadata=metadata)


def convert_net(network: Network, network_id: str = "network") -> LoadedNetwork:
    """
    Convert a gridfm_datakit Network to a pypowsybl network and capture the
    generator costs in the returned metadata.

    This is the inverse of :func:`load_net`: where ``load_net`` goes from a
    file to gridfm_datakit, ``convert_net`` goes from gridfm_datakit back to
    pypowsybl.

    Because pypowsybl has no concept of generator cost functions, the costs
    stored in ``network.gencosts`` are extracted and placed in
    ``metadata.gen_costs`` so the caller can persist them independently
    (e.g. write them to a database, a JSON sidecar, or re-inject them the
    next time :func:`from_powsybl` is called via :class:`ConversionOptions`).

    Parameters
    ----------
    network : Network
        The gridfm_datakit Network to convert.  The ``gencosts`` attribute
        must be a valid MATPOWER-style cost matrix; if it is ``None`` or
        absent, ``metadata.gen_costs`` will be an empty dict.
    network_id : str, optional
        The ID string assigned to the created pypowsybl network.  pypowsybl
        uses the network ID as a human-readable label.  Defaults to
        ``"network"``.

    Returns
    -------
    LoadedNetwork
        A :class:`LoadedNetwork` with:

        * ``pp_net`` — a freshly created pypowsybl network equivalent to
          ``network``.  Note: pypowsybl does **not** store the generator costs.
        * ``gfm_net`` — the original ``network`` object passed in (stored as a
          reference, not a copy).
        * ``metadata`` — a :class:`NetworkMetadata` whose ``gen_costs`` dict
          contains the polynomial cost coefficients extracted from
          ``network.gencosts``.

    Raises
    ------
    ImportError
        If pypowsybl is not installed.

    Examples
    --------
    Convert back to pypowsybl after running the generation pipeline:

    >>> from gridfm_datakit.powsybl import load_net, convert_net
    >>> loaded = load_net("my_network.xiidm")
    >>> # ... run generate_power_flow_data(config) using loaded.gfm_net ...
    >>> result = convert_net(loaded.gfm_net)
    >>> result.pp_net.get_generators()   # inspect in pypowsybl
    >>> result.metadata.gen_costs        # retrieve cost data for persistence
    {'0': (0.0, 1.0, 0.0), '1': (0.0, 1.0, 0.0), ...}
    """
    check_powsybl_available()

    # Extract polynomial cost coefficients from the gencost matrix.
    # These are the costs that will be lost when we hand the network to
    # pypowsybl, so we capture them in metadata now.
    gen_costs = _extract_gen_costs(network)
    metadata = NetworkMetadata(gen_costs=gen_costs)

    # Build the pypowsybl network from the gridfm_datakit Network.
    pp_net = to_powsybl(network, network_id=network_id)

    return LoadedNetwork(pp_net=pp_net, gfm_net=network, metadata=metadata)


__all__ = [
    # Primary entry points
    "load_net",
    "convert_net",
    # Lower-level conversion primitives (for advanced use)
    "from_powsybl",
    "to_powsybl",
    # PF solver mapping utilities
    "build_p2g_maps",
    "to_powsybl_with_mapping",
    # Configuration and data classes
    "ConversionOptions",
    "LoadedNetwork",
    "NetworkMetadata",
]
