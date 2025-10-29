"""
Test for compare_pf_results function
Tests that PF results from temporary files match results from original case files
"""

import pytest
from gridfm_datakit.network import load_net_from_pglib
from gridfm_datakit.process.solvers import compare_pf_results
from gridfm_datakit.process.process_network import init_julia


class TestComparePF_OPF_Results:
    """Test class for comparing PF and OPF results"""

    @classmethod
    def setup_class(cls):
        """Initialize Julia interface once for all tests"""
        cls.jl = init_julia()

    @pytest.mark.parametrize(
        "case_name,solver_type",
        [
            ("case24_ieee_rts", "pf"),
            ("case24_ieee_rts", "opf"),
            ("case57_ieee", "pf"),
            ("case57_ieee", "opf"),
            ("case118_ieee", "pf"),
            ("case118_ieee", "opf"),
            ("case300_ieee", "pf"),
            ("case300_ieee", "opf"),
            ("case2000_goc", "pf"),
            ("case2000_goc", "opf"),
            ("case10000_goc", "pf"),
            ("case10000_goc", "opf"),
        ],
    )
    def test_compare_results(self, case_name, solver_type):
        """Test that PF/OPF results from temp file match results from original case file"""
        solver_name = "PF" if solver_type == "pf" else "OPF"
        print(f"\nTesting {solver_name} result comparison for {case_name}...")

        # Load network
        net = load_net_from_pglib(case_name)
        print(
            f"Loaded {case_name}: {net.buses.shape[0]} buses, {net.gens.shape[0]} gens, {net.branches.shape[0]} branches",
        )

        # Compare results
        results_match = compare_pf_results(net, self.jl, case_name, solver_type)

        # Assert that results match
        assert results_match, (
            f"{solver_name} results from temp file don't match original case file for {case_name}"
        )

        print(f"{solver_name} result comparison passed for {case_name}")
