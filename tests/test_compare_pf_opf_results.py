"""
Test for compare_pf_results function
Tests that PF results from temporary files match results from original case files
"""

import pytest
from gridfm_datakit.network import load_net_from_pglib
from gridfm_datakit.process.solvers import compare_pf_results
from gridfm_datakit.process.process_network import init_julia
import os

SKIP_LARGE_GRIDS = os.getenv("SKIP_LARGE_GRIDS", "0") == "1"

# Small grids (<57 buses)
small_grids = [
    ("case24_ieee_rts", "pf", True),
    ("case24_ieee_rts", "pf", False),
    ("case24_ieee_rts", "opf", False),
]

# Large grids (>=57 buses)
large_grids = [
    ("case57_ieee", "pf", True),
    ("case57_ieee", "pf", False),
    ("case57_ieee", "opf", False),
    ("case118_ieee", "pf", True),
    ("case118_ieee", "pf", False),
    ("case118_ieee", "opf", False),
    ("case300_ieee", "pf", True),
    ("case300_ieee", "pf", False),
    ("case300_ieee", "opf", False),
    ("case2000_goc", "pf", True),
    ("case2000_goc", "pf", False),
    ("case2000_goc", "opf", False),
    # ("case10000_goc", "pf", True),
    # ("case10000_goc", "pf", False),
    # ("case10000_goc", "opf", False),
]

# Choose which list to parametrize
test_cases = small_grids if SKIP_LARGE_GRIDS else small_grids + large_grids


class TestComparePF_OPF_Results:
    """Test class for comparing PF and OPF results"""

    @classmethod
    def setup_class(cls):
        """Initialize Julia interface once for all tests"""
        cls.jl = init_julia(max_iter=150)

    @pytest.mark.parametrize("case_name,solver_type,fast", test_cases)
    def test_compare_results(self, case_name, solver_type, fast):
        """Test that PF/OPF results from temp file match results from original case file"""
        solver_name = "PF" if solver_type == "pf" else "OPF"
        print(f"\nTesting {solver_name} result comparison for {case_name}...")

        # Load network
        net = load_net_from_pglib(case_name)
        print(
            f"Loaded {case_name}: {net.buses.shape[0]} buses, {net.gens.shape[0]} gens, {net.branches.shape[0]} branches",
        )

        # Compare results
        results_match = compare_pf_results(net, self.jl, case_name, fast, solver_type)

        # Assert that results match
        assert results_match, (
            f"{solver_name} results from temp file don't match original case file for {case_name} with fast={fast}"
        )

        print(
            f"{solver_name} result comparison passed for {case_name} with fast={fast}",
        )
