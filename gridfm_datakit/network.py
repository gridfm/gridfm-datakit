"""
Network module for power system data handling and MATPOWER case file operations.

This module provides functionality for loading, processing, and saving power system
networks in MATPOWER format, with support for non-continuous bus indexing.
"""

import os
import requests
from importlib import resources
import pandas as pd
from numpy import ones, conj, nonzero, any, exp, pi, hstack, real, int64
from scipy.sparse import csr_matrix
from matpowercaseframes import CaseFrames
from gridfm_datakit.utils.idx_bus import (
    BUS_I,
    BUS_TYPE,
    PD,
    QD,
    GS,
    BS,
    VM,
    VA,
    REF,
)
from gridfm_datakit.utils.idx_gen import GEN_BUS, GEN_STATUS, PG, QG
from gridfm_datakit.utils.idx_brch import (
    F_BUS,
    T_BUS,
    BR_STATUS,
    BR_R,
    BR_X,
    BR_B,
    TAP,
    SHIFT,
    BR_R_ASYM,
    BR_X_ASYM,
)
from gridfm_datakit.utils.idx_cost import NCOST
import warnings
import networkx as nx
import numpy as np
import copy
from typing import Dict, Tuple, Any


def numpy_to_matlab_matrix(array: np.ndarray, name: str) -> str:
    """Format a NumPy array as a MATLAB matrix assignment to mpc.<name>.

    Args:
        array: NumPy array to format as MATLAB matrix.
        name: Name of the matrix variable in MATLAB.

    Returns:
        String containing the MATLAB matrix assignment code.
    """
    lines = [f"mpc.{name} = ["]
    for row in array:
        formatted_row = "  ".join(f"{int(v) if v == int(v) else v}" for v in row)
        lines.append(f"    {formatted_row};")
    lines.append("];\n")
    return "\n".join(lines)


class Network:
    """Power system network representation with MATPOWER compatibility.

    This class handles power system networks loaded from MATPOWER case files,
    providing functionality for bus index mapping, power flow calculations,
    and data export. It automatically handles non-continuous bus indexing
    by mapping to continuous indices for internal processing.

    Attributes:
        mpc: Original MATPOWER case dictionary.
        baseMVA: Base MVA for the power system.
        buses: Bus data array with continuous indexing.
        gens: Generator data array with continuous indexing.
        branches: Branch data array with continuous indexing.
        gencosts: Generator cost data array.
        original_bus_indices: Original bus indices from MATPOWER file.
        bus_index_mapping: Mapping from original to continuous bus indices.
        reverse_bus_index_mapping: Mapping from continuous to original bus indices.
        ref_bus_idx: Index of the reference bus.
    """

    def __init__(self, mpc: Dict[str, Any]) -> None:
        """Initialize Network from MATPOWER case dictionary.

        Args:
            mpc: MATPOWER case dictionary containing bus, gen, branch, and gencost data.

        Raises:
            AssertionError: If generator buses are not in bus IDs or if there's not exactly one reference bus.
        """
        self.mpc = mpc
        self.baseMVA = self.mpc.get("baseMVA", 100)

        self.buses = self.mpc["bus"].copy()
        self.gens = self.mpc["gen"].copy()
        self.branches = self.mpc["branch"].copy()
        self.gencosts = self.mpc["gencost"].copy()

        # Store original bus indices before conversion (these are 1-based from MATPOWER)
        self.original_bus_indices = self.buses[:, BUS_I].astype(int).copy()

        # Create mapping from original bus indices to continuous indices (0, 1, 2, ..., n_bus-1)
        unique_bus_indices = np.unique(self.original_bus_indices)
        self.bus_index_mapping = {
            int(orig_idx): new_idx
            for new_idx, orig_idx in enumerate(unique_bus_indices)
        }
        self.reverse_bus_index_mapping = {
            new_idx: int(orig_idx)
            for orig_idx, new_idx in self.bus_index_mapping.items()
        }

        # Convert bus indices to continuous (0-based) for internal processing
        self.buses[:, BUS_I] = np.array(
            [self.bus_index_mapping[int(idx)] for idx in self.buses[:, BUS_I]],
        )
        self.gens[:, GEN_BUS] = np.array(
            [self.bus_index_mapping[int(idx)] for idx in self.gens[:, GEN_BUS]],
        )
        self.branches[:, F_BUS] = np.array(
            [self.bus_index_mapping[int(idx)] for idx in self.branches[:, F_BUS]],
        )
        self.branches[:, T_BUS] = np.array(
            [self.bus_index_mapping[int(idx)] for idx in self.branches[:, T_BUS]],
        )

        # assert all generator buses are in bus IDs
        assert np.all(np.isin(self.gens[:, GEN_BUS], self.buses[:, BUS_I])), (
            "All generator buses should be in bus IDs"
        )
        # assert all generators have the same number of cost coefficients
        assert np.all(self.gencosts[:, NCOST] == self.gencosts[:, NCOST][0]), (
            "All generators must have the same number of cost coefficients"
        )

        # assert only one reference bus
        assert np.sum(self.buses[:, BUS_TYPE] == REF) == 1, (
            "There should be exactly one reference bus"
        )
        self.ref_bus_idx = np.where(self.buses[:, BUS_TYPE] == REF)[0][0]

        self.check_single_connected_component()

    @property
    def idx_gens_in_service(self) -> np.ndarray:
        """Get indices of generators that are in service.

        Returns:
            Array of generator indices that are currently in service (status = 1).
        """
        return (np.where(self.gens[:, GEN_STATUS] == 1)[0]).astype(int)

    @property
    def idx_branches_in_service(self) -> np.ndarray:
        """Get indices of branches that are in service.

        Returns:
            Array of branch indices that are currently in service (status = 1).
        """
        return (np.where(self.branches[:, BR_STATUS] == 1)[0]).astype(int)

    @property
    def Pd(self) -> np.ndarray:
        """Get active power demand at all buses.

        Returns:
            Array of active power demand values for all buses.
        """
        return self.buses[:, PD]

    @Pd.setter
    def Pd(self, value: np.ndarray) -> None:
        """Set active power demand at all buses.

        Args:
            value: Array of active power demand values.
        """
        self.buses[:, PD] = value

    @property
    def Qd(self) -> np.ndarray:
        """Get reactive power demand at all buses.

        Returns:
            Array of reactive power demand values for all buses.
        """
        return self.buses[:, QD]

    @Qd.setter
    def Qd(self, value: np.ndarray) -> None:
        """Set reactive power demand at all buses.

        Args:
            value: Array of reactive power demand values.
        """
        self.buses[:, QD] = value

    @property
    def Pg_gen(self) -> np.ndarray:
        """Get active power generation at all generators.

        Returns:
            Array of active power generation values for all generators.
        """
        return self.gens[:, PG]

    @Pg_gen.setter
    def Pg_gen(self, value: np.ndarray) -> None:
        """Set active power generation at generators in service.

        Args:
            value: Array of active power generation values.
        """
        self.gens[self.idx_gens_in_service, PG] = value

    @property
    def Qg_gen(self) -> np.ndarray:
        """Get reactive power generation at all generators.

        Returns:
            Array of reactive power generation values for all generators.
        """
        return self.gens[:, QG]

    @Qg_gen.setter
    def Qg_gen(self, value: np.ndarray) -> None:
        """Set reactive power generation at generators in service.

        Args:
            value: Array of reactive power generation values.
        """
        self.gens[self.idx_gens_in_service, QG] = value

    @property
    def Vm(self) -> np.ndarray:
        """Get voltage magnitude at all buses.

        Returns:
            Array of voltage magnitude values for all buses.
        """
        return self.buses[:, VM]

    @Vm.setter
    def Vm(self, value: np.ndarray) -> None:
        """Set voltage magnitude at all buses.

        Args:
            value: Array of voltage magnitude values.
        """
        self.buses[:, VM] = value

    @property
    def Va(self) -> np.ndarray:
        """Get voltage angle at all buses.

        Returns:
            Array of voltage angle values for all buses.
        """
        return self.buses[:, VA]

    @Va.setter
    def Va(self, value: np.ndarray) -> None:
        """Set voltage angle at all buses.

        Args:
            value: Array of voltage angle values.
        """
        self.buses[:, VA] = value

    @property
    def Pg_bus(self) -> np.ndarray:
        """Get active power generation at all buses.

        Returns:
            Array of active power generation values for all buses.
        """
        return self.buses[:, PG]

    @Pg_bus.setter
    def Pg_bus(self, value: np.ndarray) -> None:
        """Set active power generation at buses (not allowed).

        Args:
            value: Array of active power generation values.

        Raises:
            ValueError: Power generation should be set at the generator level.
        """
        raise ValueError("Power generation should be set at the generator level")

    @property
    def Qg_bus(self) -> np.ndarray:
        """Get reactive power generation at all buses.

        Returns:
            Array of reactive power generation values for all buses.
        """
        return self.buses[:, QG]

    @Qg_bus.setter
    def Qg_bus(self, value: np.ndarray) -> None:
        """Set reactive power generation at buses (not allowed).

        Args:
            value: Array of reactive power generation values.

        Raises:
            ValueError: Power generation should be set at the generator level.
        """
        raise ValueError("Power generation should be set at the generator level")

    def deactivate_branches(self, idx_branches: np.ndarray) -> None:
        """Deactivate specified branches by setting their status to 0.

        Args:
            idx_branches: Array of branch indices to deactivate.

        Warns:
            UserWarning: If trying to deactivate branches that are already deactivated.
        """
        # throw warning if try deactivating branches that are already deactivated
        if not np.all(self.branches[idx_branches, BR_STATUS] == 1):
            warnings.warn(
                f"Trying to deactivate branches that are already deactivated: {idx_branches}",
            )
        self.branches[idx_branches, BR_STATUS] = 0

    def deactivate_gens(self, idx_gens: np.ndarray) -> None:
        """Deactivate specified generators by setting their status to 0.

        Args:
            idx_gens: Array of generator indices to deactivate.

        Warns:
            UserWarning: If trying to deactivate generators that are already deactivated.
        """
        # throw warning if try deactivate gens that are already deactivated
        if not np.all(self.gens[idx_gens, GEN_STATUS] == 1):
            warnings.warn(
                f"Trying to deactivate gens that are already deactivated: {idx_gens}",
            )
        self.gens[idx_gens, GEN_STATUS] = 0

    def check_single_connected_component(self) -> bool:
        """
        Check that the network forms a single connected component.

        Creates a NetworkX graph with buses as nodes and in-service branches as edges,
        then checks if there is exactly one connected component.

        Returns:
            bool: True if there is exactly one connected component, False otherwise
        """
        # Create NetworkX graph
        G = nx.Graph()

        # Add all buses as nodes
        n_buses = self.buses.shape[0]
        G.add_nodes_from(range(n_buses))

        # Add in-service branches as edges
        in_service_branches = self.idx_branches_in_service
        for branch_idx in in_service_branches:
            from_bus = int(self.branches[branch_idx, F_BUS])
            to_bus = int(self.branches[branch_idx, T_BUS])
            G.add_edge(from_bus, to_bus)

        # Find connected components
        connected_components = list(nx.connected_components(G))

        # Check if there is exactly one connected component
        if len(connected_components) == 1:
            return True
        else:
            return False

    def version(self) -> str:
        """Get the MATPOWER version from the MPC dictionary.

        Returns:
            MATPOWER version string, defaults to '2' if not specified.
        """
        return self.mpc.get("version", "2")

    def __eq__(self, other: Any) -> bool:
        """Structural equality: compare core fields and matrices with tolerance.

        Two Network objects are considered equal if their scalar attributes and
        all core matrices are numerically equal (within a small tolerance), and
        their bus index mappings agree.
        """
        if not isinstance(other, Network):
            return False

        # Compare simple scalars
        try:
            if self.version() != other.version():
                return False
            if not np.isclose(self.baseMVA, other.baseMVA, atol=1e-12, rtol=0):
                return False
            if int(self.ref_bus_idx) != int(other.ref_bus_idx):
                return False

            # Compare arrays with tolerance
            def arrays_close(a: np.ndarray, b: np.ndarray) -> bool:
                if a is None and b is None:
                    return True
                if (a is None) != (b is None):
                    return False
                if a.shape != b.shape:
                    return False
                # Use allclose for numeric matrices
                return np.allclose(a, b, atol=1e-12, rtol=0)

            if not arrays_close(self.buses, other.buses):
                return False
            if not arrays_close(self.gens, other.gens):
                return False
            if not arrays_close(self.branches, other.branches):
                return False
            if not arrays_close(self.gencosts, other.gencosts):
                return False
            if not arrays_close(self.original_bus_indices, other.original_bus_indices):
                return False

            # Compare mappings
            if self.bus_index_mapping != other.bus_index_mapping:
                return False
            if self.reverse_bus_index_mapping != other.reverse_bus_index_mapping:
                return False

            return True
        except Exception:
            return False

    def to_mpc(self, filename: str) -> None:
        """Convert network data to MATPOWER .m case file format.

        This method saves the network data to a MATPOWER case file, restoring
        the original bus indices for MATPOWER compatibility.

        Args:
            filename: Path where the MATPOWER case file should be saved.

        Raises:
            AssertionError: If bus, gen, or branch matrices don't have the required number of columns.
        """

        to_save = copy.deepcopy(self)
        # Restore original bus indices (1-based for MATPOWER)
        to_save.buses[:, BUS_I] = np.array(
            [self.reverse_bus_index_mapping[idx] for idx in to_save.buses[:, BUS_I]],
            dtype=int,
        )
        to_save.gens[:, GEN_BUS] = np.array(
            [self.reverse_bus_index_mapping[idx] for idx in to_save.gens[:, GEN_BUS]],
            dtype=int,
        )
        to_save.branches[:, F_BUS] = np.array(
            [self.reverse_bus_index_mapping[idx] for idx in to_save.branches[:, F_BUS]],
            dtype=int,
        )
        to_save.branches[:, T_BUS] = np.array(
            [self.reverse_bus_index_mapping[idx] for idx in to_save.branches[:, T_BUS]],
            dtype=int,
        )

        with open(filename, "w") as f:
            f.write("function mpc = case_from_dict\n")
            f.write("% Automatically generated MATPOWER case file\n\n")

            # version and baseMVA
            f.write(f"mpc.version = '{to_save.version()}';\n")
            f.write(f"mpc.baseMVA = {to_save.baseMVA};\n\n")

            # -------------------------
            # BUS matrix
            # -------------------------
            assert to_save.buses.ndim == 2, "mpc['bus'] must be a 2D array"
            assert to_save.buses.shape[1] >= 13, (
                f"mpc['bus'] has {to_save.buses.shape[1]} columns, expected ≥13"
            )
            f.write(
                "% Columns: BUS_I  BUS_TYPE  PD  QD  GS  BS  BUS_AREA  VM  VA  BASE_KV  ZONE  VMAX  VMIN\n",
            )
            f.write(numpy_to_matlab_matrix(to_save.buses, "bus"))

            # -------------------------
            # GEN matrix
            # -------------------------
            assert to_save.gens.ndim == 2, "mpc['gen'] must be a 2D array"
            assert to_save.gens.shape[1] >= 10, (
                f"mpc['gen'] has {to_save.gens.shape[1]} columns, expected minimum ≥10"
            )
            f.write(
                "% Columns: GEN_BUS  PG  QG  QMAX  QMIN  VG  MBASE  GEN_STATUS  PMAX  PMIN  "
                "PC1  PC2  QC1MIN  QC1MAX  QC2MIN  QC2MAX  RAMP_AGC  RAMP_10  RAMP_30  RAMP_Q  APF\n",
            )
            f.write(numpy_to_matlab_matrix(to_save.gens, "gen"))

            # -------------------------
            # BRANCH matrix (always 13 columns)
            # -------------------------
            assert to_save.branches.ndim == 2, "mpc['branch'] must be a 2D array"
            assert to_save.branches.shape[1] >= 13, (
                f"mpc['branch'] has {to_save.branches.shape[1]} columns, expected ≥13"
            )
            f.write(
                "% Columns: F_BUS  T_BUS  BR_R  BR_X  BR_B  RATE_A  RATE_B  RATE_C  TAP  SHIFT  BR_STATUS  ANGMIN  ANGMAX\n",
            )
            f.write(numpy_to_matlab_matrix(to_save.branches, "branch"))

            # -------------------------
            # GENCOST matrix
            # -------------------------
            if to_save.gencosts is not None:
                assert to_save.gencosts.ndim == 2, "mpc['gencost'] must be a 2D array"
                f.write(
                    "% Columns: MODEL  STARTUP  SHUTDOWN  NCOST  COST (coefficients or x-y pairs)\n",
                )
                f.write(numpy_to_matlab_matrix(to_save.gencosts, "gencost"))

        # print(f"MATPOWER case file saved as {filename}")


def load_net_from_file(network_path: str) -> Network:
    """Load a network from a MATPOWER file.

    Args:
        network_path: Path to the MATPOWER file (without extension).

    Returns:
        Network object containing the power network configuration.

    Raises:
        FileNotFoundError: If the network file doesn't exist.
        ValueError: If the file format is invalid.
    """
    # Load network using matpowercaseframes
    mpc_frames = CaseFrames(network_path)
    mpc = {
        key: mpc_frames.__getattribute__(key)
        if not isinstance(mpc_frames.__getattribute__(key), pd.DataFrame)
        else mpc_frames.__getattribute__(key).values
        for key in mpc_frames._attributes
    }

    return Network(mpc)


def load_net_from_pglib(grid_name: str) -> Network:
    """Load a power grid network from PGLib using matpowercaseframes.

    Downloads the network file if not locally available and loads it into a Network object.

    Args:
        grid_name: Name of the grid file without the prefix 'pglib_opf_'
                  (e.g., 'case14_ieee', 'case118_ieee').

    Returns:
        Network object containing the power network configuration.

    Raises:
        requests.exceptions.RequestException: If download fails.
        FileNotFoundError: If the file cannot be found after download.
        ValueError: If the file format is invalid.
    """
    # Construct file paths
    file_path = str(
        resources.files("gridfm_datakit.grids").joinpath(f"pglib_opf_{grid_name}.m"),
    )

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Download file if not exists
    if not os.path.exists(file_path):
        url = f"https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/pglib_opf_{grid_name}.m"
        response = requests.get(url)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            f.write(response.content)

    # Load network using matpowercaseframes
    mpc_frames = CaseFrames(file_path)
    mpc = {
        key: mpc_frames.__getattribute__(key)
        if not isinstance(mpc_frames.__getattribute__(key), pd.DataFrame)
        else mpc_frames.__getattribute__(key).values
        for key in mpc_frames._attributes
    }

    return Network(mpc)


def makeYbus(
    baseMVA: float,
    bus: np.ndarray,
    branch: np.ndarray,
) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
    """Build the bus admittance matrix and branch admittance matrices.

    Returns the full bus admittance matrix (i.e. for all buses) and the
    matrices Yf and Yt which, when multiplied by a complex voltage
    vector, yield the vector currents injected into each line from the
    "from" and "to" buses respectively of each line. Does appropriate
    conversions to p.u.

    Args:
        baseMVA: Base MVA for the power system.
        bus: Bus data array.
        branch: Branch data array.

    Returns:
        Tuple containing:
        - Ybus: Bus admittance matrix (sparse)
        - Yf: Branch admittance matrix for "from" buses (sparse)
        - Yt: Branch admittance matrix for "to" buses (sparse)
    """
    # constants
    nb = bus.shape[0]  # number of buses
    nl = branch.shape[0]  # number of lines

    # for each branch, compute the elements of the branch admittance matrix where
    #
    #      | If |   | Yff  Yft |   | Vf |
    #      |    | = |          | * |    |
    #      | It |   | Ytf  Ytt |   | Vt |
    #
    Ytt, Yff, Yft, Ytf = branch_vectors(branch, nl)
    # compute shunt admittance
    # if Psh is the real power consumed by the shunt at V = 1.0 p.u.
    # and Qsh is the reactive power injected by the shunt at V = 1.0 p.u.
    # then Psh - j Qsh = V * conj(Ysh * V) = conj(Ysh) = Gs - j Bs,
    # i.e. Ysh = Psh + j Qsh, so ...
    # vector of shunt admittances
    Ysh = (bus[:, GS] + 1j * bus[:, BS]) / baseMVA

    # build connection matrices
    f = real(branch[:, F_BUS]).astype(int64)  # list of "from" buses
    t = real(branch[:, T_BUS]).astype(int64)  # list of "to" buses
    # connection matrix for line & from buses
    Cf = csr_matrix((ones(nl), (range(nl), f)), (nl, nb))
    # connection matrix for line & to buses
    Ct = csr_matrix((ones(nl), (range(nl), t)), (nl, nb))

    # build Yf and Yt such that Yf * V is the vector of complex branch currents injected
    # at each branch's "from" bus, and Yt is the same for the "to" bus end
    i = hstack([range(nl), range(nl)])  # double set of row indices

    Yf = csr_matrix((hstack([Yff, Yft]), (i, hstack([f, t]))), (nl, nb))
    Yt = csr_matrix((hstack([Ytf, Ytt]), (i, hstack([f, t]))), (nl, nb))
    # Yf = spdiags(Yff, 0, nl, nl) * Cf + spdiags(Yft, 0, nl, nl) * Ct
    # Yt = spdiags(Ytf, 0, nl, nl) * Cf + spdiags(Ytt, 0, nl, nl) * Ct

    # build Ybus
    Ybus = Cf.T * Yf + Ct.T * Yt + csr_matrix((Ysh, (range(nb), range(nb))), (nb, nb))
    Ybus.sort_indices()
    Ybus.eliminate_zeros()

    return Ybus, Yf, Yt


def branch_vectors(
    branch: np.ndarray,
    nl: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute branch admittance vectors for Ybus construction.

    Args:
        branch: Branch data array.
        nl: Number of lines/branches.

    Returns:
        Tuple containing:
        - Ytt: Branch admittance matrix diagonal elements for "to" buses
        - Yff: Branch admittance matrix diagonal elements for "from" buses
        - Yft: Branch admittance matrix off-diagonal elements (from to to)
        - Ytf: Branch admittance matrix off-diagonal elements (to to from)
    """
    n_cols = branch.shape[1]
    stat = branch[:, BR_STATUS]  # ones at in-service branches
    Ysf = stat / (branch[:, BR_R] + 1j * branch[:, BR_X])  # series admittance
    if n_cols > BR_R_ASYM and (any(branch[:, BR_R_ASYM]) or any(branch[:, BR_X_ASYM])):
        Yst = stat / (
            (branch[:, BR_R] + branch[:, BR_R_ASYM])
            + 1j * (branch[:, BR_X] + branch[:, BR_X_ASYM])
        )  # series admittance
    else:
        Yst = Ysf
    Bc = stat * branch[:, BR_B]  # line charging susceptance
    tap = ones(nl)  # default tap ratio = 1
    i = nonzero(real(branch[:, TAP]))  # indices of non-zero tap ratios
    tap[i] = real(branch[i, TAP])  # assign non-zero tap ratios
    tap = tap * exp(1j * pi / 180 * branch[:, SHIFT])  # add phase shifters

    Ytt = Yst + 1j * Bc / 2
    Yff = (Ysf + 1j * Bc / 2) / (tap * conj(tap))
    Yft = -Ysf / conj(tap)
    Ytf = -Yst / tap
    return Ytt, Yff, Yft, Ytf


if __name__ == "__main__":
    network = load_net_from_pglib("case24_ieee_rts")
    network.to_mpc("tmp_case.m")
    print("MATPOWER case file saved as tmp_case.m")
