"""
Optional exhaustive test for compare_pf_results across all PGLib OPF cases.
Enable by setting environment variable RUN_ALL_PGLIB=1 when invoking pytest.
"""

import os
import pytest
from gridfm_datakit.network import load_net_from_pglib
from gridfm_datakit.process.solvers import compare_pf_results
from gridfm_datakit.process.process_network import init_julia


# Skip this entire test file unless explicitly enabled
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_ALL_PGLIB") != "1",
    reason="Set RUN_ALL_PGLIB=1 to run exhaustive PGLib PF/OPF comparisons",
)


# Full list of PGLib OPF cases (names without the 'pglib_opf_' prefix)
PGLIB_CASES = [
    "case3_lmbd",
    "case5_pjm",
    "case14_ieee",
    "case24_ieee_rts",
    "case30_as",
    "case30_ieee",
    "case39_epri",
    "case57_ieee",
    "case60_c",
    "case73_ieee_rts",
    "case89_pegase",
    "case118_ieee",
    "case162_ieee_dtc",
    "case179_goc",
    "case197_snem",
    "case200_activ",
    "case300_ieee",
    "case500_goc",
    "case793_goc",
    "case1803_snem",
    "case1888_rte",
    "case1951_rte",
    "case2000_goc",
    "case2312_goc",
    "case2383wp_k",
    "case240_pserc",
    "case2736sp_k",
    "case2737sop_k",
    "case2742_goc",
    "case2746wop_k",
    "case2746wp_k",
    "case2848_rte",
    "case2853_sdet",
    "case2868_rte",
    "case2869_pegase",
    "case3012wp_k",
    "case3022_goc",
    "case3120sp_k",
    "case3375wp_k",
    "case3970_goc",
    "case4020_goc",
    "case4601_goc",
    "case4619_goc",
    "case4661_sdet",
    "case4837_goc",
    "case4917_goc",
    "case5658_epigrids",
    "case6468_rte",
    "case6470_rte",
    "case6495_rte",
    "case6515_rte",
    "case7336_epigrids",
    "case9241_pegase",
    "case9591_goc",
    "case10192_epigrids",
    "case10480_goc",
    "case13659_pegase",
    "case19402_goc",
    "case20758_epigrids",
    "case24464_goc",
    "case10000_goc",
    "case30000_goc",
    "case78484_epigrids",
]


class TestComparePF_OPF_Results_AllPGLib:
    """Exhaustive comparison of PF/OPF results for all PGLib cases (optional)."""

    @classmethod
    def setup_class(cls):
        """Initialize Julia once for all test parameters."""
        cls.jl = init_julia(max_iter=150)

    @pytest.mark.parametrize(
        "case_name,solver_type",
        [(c, s) for c in PGLIB_CASES for s in ("pf", "opf")],
    )
    def test_compare_results_all_pglib(self, case_name, solver_type):
        solver_name = "PF" if solver_type == "pf" else "OPF"
        print(f"\n[All-PGLib] Testing {solver_name} result comparison for {case_name}…")

        # Load network
        net = load_net_from_pglib(case_name)
        print(
            f"Loaded {case_name}: {net.buses.shape[0]} buses, {net.gens.shape[0]} gens, {net.branches.shape[0]} branches",
        )

        # Compare results
        results_match = compare_pf_results(net, self.jl, case_name, solver_type)

        assert results_match, (
            f"{solver_name} results from temp file don't match original case file for {case_name}"
        )
