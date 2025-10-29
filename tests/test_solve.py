"""
Minimal test for solve functions (OPF, PF preprocessing, PF post-processing)
Tests the complete workflow on several IEEE cases
"""

import pytest
from gridfm_datakit.network import load_net_from_pglib
from gridfm_datakit.process.solvers import run_opf, run_pf
from gridfm_datakit.process.process_network import (
    init_julia,
    pf_preprocessing,
    pf_post_processing,
)

import time


class TestSolve:
    """Test class for solve functions"""

    @classmethod
    def setup_class(cls):
        """Initialize Julia interface once for all tests"""
        cls.jl = init_julia()

    @pytest.mark.parametrize(
        "case_name",
        [
            # "case24_ieee_rts",
            # "case57_ieee",
            # "case118_ieee",
            # "case300_ieee",
            "case2000_goc",
            # "case30000_goc"
        ],
    )
    def test_complete_workflow(self, case_name):
        """Test complete workflow: OPF → PF post-processing → PF → PF post-processing on IEEE cases"""
        print(f"\nTesting {case_name}...")

        # Load network
        net = load_net_from_pglib(case_name)
        print(
            f"  Loaded {case_name}: {net.buses.shape[0]} buses, {net.gens.shape[0]} gens, {net.branches.shape[0]} branches",
        )

        # Step 1: Run OPF
        print("  Running OPF...")
        start_time = time.time()
        opf_result = run_opf(net, self.jl)
        assert str(opf_result["termination_status"]) == "LOCALLY_SOLVED"
        print("  OPF converged successfully")
        end_time = time.time()
        print(f"  OPF time2: {end_time - start_time} seconds")
        start_time = time.time()
        opf_result = run_opf(net, self.jl)
        assert str(opf_result["termination_status"]) == "LOCALLY_SOLVED"
        print("  OPF converged successfully")
        end_time = time.time()
        print(f"  OPF time: {end_time - start_time} seconds")
        # Step 2: PF post-processing on OPF results
        print("  PF post-processing OPF results...")
        opf_pf_data = pf_post_processing(net, opf_result, dcpf=False)
        assert "bus" in opf_pf_data
        assert "gen" in opf_pf_data
        assert "branch" in opf_pf_data
        assert "Y_bus" in opf_pf_data
        print(
            f"  OPF post-processing completed: bus={opf_pf_data['bus'].shape}, gen={opf_pf_data['gen'].shape}, branch={opf_pf_data['branch'].shape}",
        )

        # Step 3: Run PF
        net = pf_preprocessing(net, opf_result)
        start_time = time.time()
        print("  Running PF...")
        pf_result = run_pf(net, self.jl)
        assert str(pf_result["termination_status"]) == "True"
        end_time = time.time()
        print(f"  PF time: {end_time - start_time} seconds")
        print("  PF converged successfully")

        # Step 4: PF post-processing on PF results
        print("  PF post-processing PF results...")
        pf_pf_data = pf_post_processing(net, pf_result, dcpf=False)
        assert "bus" in pf_pf_data
        assert "gen" in pf_pf_data
        assert "branch" in pf_pf_data
        assert "Y_bus" in pf_pf_data
        print(
            f"  PF post-processing completed: bus={pf_pf_data['bus'].shape}, gen={pf_pf_data['gen'].shape}, branch={pf_pf_data['branch'].shape}",
        )

        # Verify data shapes match network dimensions
        assert pf_pf_data["bus"].shape[0] == net.buses.shape[0]
        assert pf_pf_data["gen"].shape[0] == net.gens.shape[0]
        assert pf_pf_data["branch"].shape[0] == net.branches.shape[0]

        print(f"  ✅ {case_name} completed successfully")
