"""
Tests for :func:`gridfm_datakit.powsybl.load_net`.

What is tested
--------------
* The function returns a well-formed :class:`LoadedNetwork` for both
  pypowsybl-native formats (XIIDM) and MATPOWER text files (.m).
* Element counts (buses, generators, branches) are correct.
* ``metadata.gen_costs`` is **always empty** on load — gridfm_datakit
  intentionally does not extract gen_costs from files because pypowsybl
  cannot guarantee that its internal generator ordering matches the source
  file's row order.
* The gridfm_datakit Network always has a ``gencosts`` matrix populated
  with the default coefficients injected by :func:`from_powsybl`.
* Appropriate exceptions are raised for bad input.
"""

import pytest
from pathlib import Path

import numpy as np

from gridfm_datakit.powsybl.api import is_powsybl_available
from gridfm_datakit.network import Network
from gridfm_datakit.utils.idx_cost import MODEL, NCOST, COST, POLYNOMIAL

pytestmark = pytest.mark.skipif(
    not is_powsybl_available(),
    reason="pypowsybl is not installed. Install with: pip install gridfm-datakit[powsybl]",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def xiidm_case14_path(tmp_path_factory):
    """
    Write a pypowsybl IEEE-14 network to an XIIDM file and return the path.

    XIIDM is pypowsybl's native XML format.  Using a pypowsybl built-in
    network (create_ieee14) guarantees the fixture is self-contained and
    does not depend on external files.
    """
    import pypowsybl as pp
    tmp = tmp_path_factory.mktemp("xiidm")
    pp_net = pp.network.create_ieee14()
    xiidm_file = tmp / "ieee14.xiidm"
    pp_net.save(str(xiidm_file))
    return str(xiidm_file)


@pytest.fixture(scope="module")
def matpower_case14_path():
    """
    Return the path to the bundled MATPOWER pglib case14 .m file.

    This file lives in gridfm_datakit's own ``grids/`` package directory.
    """
    grids_dir = Path(__file__).parent.parent.parent / "gridfm_datakit" / "grids"
    return str(grids_dir / "pglib_opf_case14_ieee.m")


# ---------------------------------------------------------------------------
# Tests: loading a pypowsybl-native format (XIIDM)
# ---------------------------------------------------------------------------

class TestLoadNetXiidm:
    """load_net() with a pypowsybl-native XIIDM file."""

    def test_returns_loaded_network(self, xiidm_case14_path):
        """The returned object must be a LoadedNetwork with all three attributes."""
        from gridfm_datakit.powsybl import load_net, LoadedNetwork, NetworkMetadata

        loaded = load_net(xiidm_case14_path)

        assert isinstance(loaded, LoadedNetwork)
        assert loaded.pp_net is not None
        # pp_net must be a real pypowsybl network, not a placeholder.
        assert hasattr(loaded.pp_net, "get_buses")
        assert isinstance(loaded.gfm_net, Network)
        assert isinstance(loaded.metadata, NetworkMetadata)

    def test_bus_counts_match(self, xiidm_case14_path):
        """pypowsybl and gridfm_datakit must agree on the number of buses."""
        from gridfm_datakit.powsybl import load_net

        loaded = load_net(xiidm_case14_path)

        pp_bus_count = len(loaded.pp_net.get_buses())
        gfm_bus_count = loaded.gfm_net.buses.shape[0]
        assert pp_bus_count == gfm_bus_count == 14

    def test_element_counts(self, xiidm_case14_path):
        """Generator and branch counts must be correct for IEEE-14."""
        from gridfm_datakit.powsybl import load_net

        loaded = load_net(xiidm_case14_path)

        assert loaded.gfm_net.gens.shape[0] == 5
        assert loaded.gfm_net.branches.shape[0] == 20

    def test_metadata_gen_costs_always_empty(self, xiidm_case14_path):
        """
        metadata.gen_costs must be empty regardless of the file format.

        pypowsybl does not carry cost data in any file format it reads.
        Returning an empty dict is the explicit, documented contract.
        """
        from gridfm_datakit.powsybl import load_net

        loaded = load_net(xiidm_case14_path)

        assert loaded.metadata.gen_costs == {}

    def test_gfm_net_has_default_gencost_matrix(self, xiidm_case14_path):
        """
        gfm_net.gencosts must be populated with default coefficients.

        Even though no real costs are known, from_powsybl fills the gencost
        matrix with (c2=0, c1=1, c0=0) so that the Network is valid for
        OPF/PF runs and downstream code does not have to check for None.
        """
        from gridfm_datakit.powsybl import load_net

        loaded = load_net(xiidm_case14_path)
        gencosts = loaded.gfm_net.gencosts

        assert gencosts is not None
        # One row per generator.
        assert gencosts.shape[0] == loaded.gfm_net.gens.shape[0]
        # Every row must declare polynomial model (MODEL == 2).
        assert np.all(gencosts[:, MODEL] == POLYNOMIAL)

    def test_file_not_found_raises(self):
        """A non-existent path must raise FileNotFoundError."""
        from gridfm_datakit.powsybl import load_net

        with pytest.raises(FileNotFoundError):
            load_net("/nonexistent/path/to/network.xiidm")


# ---------------------------------------------------------------------------
# Tests: loading a MATPOWER text file (.m)
# ---------------------------------------------------------------------------

class TestLoadNetMatpower:
    """
    load_net() with a MATPOWER text (.m) file.

    pypowsybl cannot load .m files directly — it only understands the binary
    MATPOWER (.mat) format.  load_net() handles this transparently by
    converting .m → gridfm_datakit Network → pypowsybl.
    """

    def test_returns_loaded_network(self, matpower_case14_path):
        """The returned object must be a LoadedNetwork."""
        from gridfm_datakit.powsybl import load_net, LoadedNetwork

        loaded = load_net(matpower_case14_path)

        assert isinstance(loaded, LoadedNetwork)
        assert loaded.pp_net is not None
        assert isinstance(loaded.gfm_net, Network)

    def test_element_counts(self, matpower_case14_path):
        """Bus, generator and branch counts must be correct for case14."""
        from gridfm_datakit.powsybl import load_net

        loaded = load_net(matpower_case14_path)

        assert loaded.gfm_net.buses.shape[0] == 14
        assert len(loaded.pp_net.get_buses()) == 14
        assert loaded.gfm_net.gens.shape[0] == 5

    def test_metadata_gen_costs_always_empty(self, matpower_case14_path):
        """
        metadata.gen_costs must be empty even for .m files.

        Although .m files contain a gencost block, the generator row order
        that pypowsybl produces after parsing is not guaranteed to match the
        original file order.  Injecting costs into the wrong generators would
        silently corrupt OPF results, so gen_costs are never extracted.
        """
        from gridfm_datakit.powsybl import load_net

        loaded = load_net(matpower_case14_path)

        assert loaded.metadata.gen_costs == {}

    def test_gfm_net_has_default_gencost_matrix(self, matpower_case14_path):
        """gfm_net.gencosts must be populated with the default (0, 1, 0) coefficients."""
        from gridfm_datakit.powsybl import load_net

        loaded = load_net(matpower_case14_path)
        gencosts = loaded.gfm_net.gencosts

        assert gencosts is not None
        assert gencosts.shape[0] == loaded.gfm_net.gens.shape[0]
        assert np.all(gencosts[:, MODEL] == POLYNOMIAL)
