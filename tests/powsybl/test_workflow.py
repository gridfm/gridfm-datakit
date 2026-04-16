"""
End-to-end workflow tests: load_net → (gridfm pipeline) → convert_net.

What is tested
--------------
These tests exercise the full integration contract described in the module
docstring of ``gridfm_datakit.powsybl``:

1. A network file is loaded with :func:`load_net`.
2. The resulting :class:`LoadedNetwork` is verified to be correct.
3. :func:`convert_net` is called on the gridfm_datakit Network that came out
   of step 1, simulating what happens after the data-generation pipeline
   finishes.
4. The final :class:`LoadedNetwork` is verified — in particular that
   ``metadata.gen_costs`` now contains coefficients even though none were
   present after step 1.

The tests cover two file-format paths:

* **XIIDM** (pypowsybl-native XML format) — the common case when networks are
  authored or exported in pypowsybl's own format.
* **MATPOWER .m text** — the common case when networks come from academic
  repositories (PGLib, MATPOWER case library, etc.).  pypowsybl cannot load
  .m files directly; load_net handles the conversion automatically.

Gen-costs lifecycle
-------------------
After ``load_net``:
  - ``metadata.gen_costs`` is **empty** (pypowsybl carries no cost data).
  - ``gfm_net.gencosts`` contains the **default** coefficients (0·P²+1·P+0).

After ``convert_net(gfm_net)``:
  - ``metadata.gen_costs`` is **populated** from ``gfm_net.gencosts``.
  - The pp_net object does **not** carry costs — pypowsybl has no such concept.

This round-trip is the mechanism the caller uses to preserve costs: extract
them from metadata and persist them externally before handing the pp_net off
to a pypowsybl solver or exporter.
"""

import pytest
from pathlib import Path

import numpy as np

from gridfm_datakit.powsybl.api import is_powsybl_available
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
    Write pypowsybl's built-in IEEE-14 network to an XIIDM file.

    XIIDM is the native XML format for pypowsybl.  Using create_ieee14()
    keeps this fixture self-contained — no external files required.
    """
    import pypowsybl as pp
    tmp = tmp_path_factory.mktemp("workflow_xiidm")
    path = tmp / "ieee14.xiidm"
    pp.network.create_ieee14().save(str(path))
    return str(path)


@pytest.fixture(scope="module")
def matpower_case14_path():
    """
    Return the path to the bundled PGLib MATPOWER case14 .m file.

    This file is shipped with gridfm_datakit and contains a full MATPOWER
    case including a gencost block with real quadratic cost coefficients.
    """
    grids_dir = Path(__file__).parent.parent.parent / "gridfm_datakit" / "grids"
    return str(grids_dir / "pglib_opf_case14_ieee.m")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _assert_gencost_matrix_is_valid(gencosts, n_gens):
    """
    Assert that a gencost matrix looks like a properly initialised MATPOWER
    cost matrix.

    Checks performed:
    - Not None.
    - Has exactly ``n_gens`` rows.
    - Every row has MODEL == POLYNOMIAL (2).
    - Every row has NCOST >= 1 (at least one coefficient).
    """
    assert gencosts is not None, "gencosts matrix must not be None"
    assert gencosts.shape[0] == n_gens, (
        f"gencosts must have {n_gens} rows, got {gencosts.shape[0]}"
    )
    assert np.all(gencosts[:, MODEL] == POLYNOMIAL), (
        "Every generator must have MODEL == POLYNOMIAL (2)"
    )
    ncosts = gencosts[:, NCOST].astype(int)
    assert np.all(ncosts >= 1), "Every generator must have at least one cost coefficient"


# ---------------------------------------------------------------------------
# Workflow tests: XIIDM source
# ---------------------------------------------------------------------------

class TestWorkflowXiidm:
    """Full load_net → convert_net cycle starting from an XIIDM file."""

    def test_load_net_returns_valid_loaded_network(self, xiidm_case14_path):
        """
        load_net must return a LoadedNetwork with a live pypowsybl network, a
        valid gridfm Network, and an empty-costs metadata object.
        """
        from gridfm_datakit.powsybl import load_net, LoadedNetwork, NetworkMetadata

        loaded = load_net(xiidm_case14_path)

        assert isinstance(loaded, LoadedNetwork)
        assert hasattr(loaded.pp_net, "get_buses"), "pp_net must be a pypowsybl network"
        assert loaded.gfm_net is not None
        assert isinstance(loaded.metadata, NetworkMetadata)

    def test_load_net_metadata_gen_costs_empty(self, xiidm_case14_path):
        """
        metadata.gen_costs must be empty after load_net.

        pypowsybl does not carry cost data, so the contract is that the
        caller receives an empty dict and must supply costs separately if
        needed.
        """
        from gridfm_datakit.powsybl import load_net

        loaded = load_net(xiidm_case14_path)

        assert loaded.metadata.gen_costs == {}

    def test_load_net_gfm_net_has_default_gencosts(self, xiidm_case14_path):
        """
        gfm_net.gencosts must be populated with default coefficients.

        from_powsybl injects (0, 1, 0) for every generator so the Network
        is valid for OPF/PF runs without requiring the caller to supply costs
        first.
        """
        from gridfm_datakit.powsybl import load_net

        loaded = load_net(xiidm_case14_path)
        n_gens = loaded.gfm_net.gens.shape[0]

        _assert_gencost_matrix_is_valid(loaded.gfm_net.gencosts, n_gens)

    def test_convert_net_metadata_has_gen_costs(self, xiidm_case14_path):
        """
        After convert_net, metadata.gen_costs must be populated.

        convert_net extracts costs from gfm_net.gencosts (which at minimum
        contains the defaults from load_net) and places them in metadata so
        the caller can persist them independently of the pypowsybl network.
        """
        from gridfm_datakit.powsybl import load_net, convert_net

        loaded = load_net(xiidm_case14_path)
        result = convert_net(loaded.gfm_net)

        n_gens = loaded.gfm_net.gens.shape[0]
        assert len(result.metadata.gen_costs) == n_gens, (
            "metadata.gen_costs must have one entry per generator"
        )

    def test_convert_net_gen_costs_match_gencost_matrix(self, xiidm_case14_path):
        """
        The coefficients in metadata.gen_costs must exactly match those in
        gfm_net.gencosts — convert_net must not alter or re-order them.
        """
        from gridfm_datakit.powsybl import load_net, convert_net

        loaded = load_net(xiidm_case14_path)
        result = convert_net(loaded.gfm_net)

        gencosts = loaded.gfm_net.gencosts
        for gen_idx_str, coeffs in result.metadata.gen_costs.items():
            i = int(gen_idx_str)
            ncost = int(gencosts[i, NCOST])
            expected = tuple(float(gencosts[i, COST + j]) for j in range(ncost))
            assert coeffs == expected, (
                f"Generator {i}: metadata coefficients {coeffs} != "
                f"gencost matrix row {expected}"
            )

    def test_convert_net_pp_net_element_counts(self, xiidm_case14_path):
        """
        The pp_net produced by convert_net must have the same element counts
        as the gfm_net it was built from.
        """
        from gridfm_datakit.powsybl import load_net, convert_net

        loaded = load_net(xiidm_case14_path)
        result = convert_net(loaded.gfm_net)

        gfm = loaded.gfm_net
        assert len(result.pp_net.get_buses()) == gfm.buses.shape[0]
        assert len(result.pp_net.get_generators()) == gfm.gens.shape[0]

    def test_convert_net_stores_gfm_net_reference(self, xiidm_case14_path):
        """
        convert_net must store the original gfm_net as a reference (not a
        copy) so the caller can verify it came from the expected source.
        """
        from gridfm_datakit.powsybl import load_net, convert_net

        loaded = load_net(xiidm_case14_path)
        result = convert_net(loaded.gfm_net)

        assert result.gfm_net is loaded.gfm_net


# ---------------------------------------------------------------------------
# Workflow tests: MATPOWER .m source
# ---------------------------------------------------------------------------

class TestWorkflowMatpower:
    """
    Full load_net → convert_net cycle starting from a MATPOWER .m text file.

    load_net converts the .m file transparently (via gridfm_datakit +
    to_powsybl) before handing a pypowsybl network to from_powsybl.
    The rest of the workflow is identical to the XIIDM path.
    """

    def test_load_net_returns_valid_loaded_network(self, matpower_case14_path):
        """load_net must succeed on a .m file and return a valid LoadedNetwork."""
        from gridfm_datakit.powsybl import load_net, LoadedNetwork

        loaded = load_net(matpower_case14_path)

        assert isinstance(loaded, LoadedNetwork)
        assert loaded.pp_net is not None
        assert loaded.gfm_net is not None

    def test_load_net_metadata_gen_costs_empty(self, matpower_case14_path):
        """
        metadata.gen_costs must be empty even for .m files.

        The .m format contains a gencost block, but the generator row order
        that pypowsybl produces is not guaranteed to match the source file
        order.  Rather than silently misassign costs, load_net always returns
        empty metadata.gen_costs.
        """
        from gridfm_datakit.powsybl import load_net

        loaded = load_net(matpower_case14_path)

        assert loaded.metadata.gen_costs == {}

    def test_load_net_gfm_net_has_default_gencosts(self, matpower_case14_path):
        """gfm_net must have a valid gencost matrix with default coefficients."""
        from gridfm_datakit.powsybl import load_net

        loaded = load_net(matpower_case14_path)
        n_gens = loaded.gfm_net.gens.shape[0]

        _assert_gencost_matrix_is_valid(loaded.gfm_net.gencosts, n_gens)

    def test_convert_net_metadata_has_gen_costs(self, matpower_case14_path):
        """After convert_net, metadata.gen_costs must have one entry per generator."""
        from gridfm_datakit.powsybl import load_net, convert_net

        loaded = load_net(matpower_case14_path)
        result = convert_net(loaded.gfm_net)

        n_gens = loaded.gfm_net.gens.shape[0]
        assert len(result.metadata.gen_costs) == n_gens

    def test_gen_costs_round_trip_preserves_coefficients(self, matpower_case14_path):
        """
        Coefficients that survive the round-trip must be numerically identical
        to those stored in gfm_net.gencosts.

        This test specifically verifies that _extract_gen_costs in __init__.py
        reads the correct columns and does not truncate or reorder them.
        """
        from gridfm_datakit.powsybl import load_net, convert_net

        loaded = load_net(matpower_case14_path)
        result = convert_net(loaded.gfm_net)

        gencosts = loaded.gfm_net.gencosts
        for gen_idx_str, coeffs in result.metadata.gen_costs.items():
            i = int(gen_idx_str)
            ncost = int(gencosts[i, NCOST])
            expected = tuple(float(gencosts[i, COST + j]) for j in range(ncost))
            assert coeffs == expected, (
                f"Generator {i}: round-trip coefficients {coeffs} != "
                f"original {expected}"
            )

    def test_convert_net_network_id_is_forwarded(self, matpower_case14_path):
        """
        The network_id argument must be forwarded to pypowsybl so the
        caller can identify the created network object.
        """
        from gridfm_datakit.powsybl import load_net, convert_net

        loaded = load_net(matpower_case14_path)
        result = convert_net(loaded.gfm_net, network_id="case14_roundtrip")

        assert result.pp_net.id == "case14_roundtrip"
