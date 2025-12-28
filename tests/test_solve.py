"""
Minimal test for solve functions (OPF, PF preprocessing, PF post-processing)
Tests the complete workflow on several IEEE cases
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from gridfm_datakit.network import load_net_from_pglib
from gridfm_datakit.process.solvers import run_opf, run_pf, run_dcpf, run_dcopf
from gridfm_datakit.process.process_network import (
    init_julia,
    pf_preprocessing,
    pf_post_processing,
)
from gridfm_datakit.utils.idx_gen import GEN_BUS
from gridfm_datakit.utils.column_names import (
    BUS_COLUMNS,
    DC_BUS_COLUMNS,
    GEN_COLUMNS,
    DC_GEN_COLUMNS,
    BRANCH_COLUMNS,
    DC_BRANCH_COLUMNS,
    RUNTIME_COLUMNS,
    DC_RUNTIME_COLUMNS,
)

import time
from gridfm_datakit.utils.utils import n_scenario_per_partition


SKIP_LARGE_GRIDS = os.getenv("SKIP_LARGE_GRIDS", "0") == "1"


class TestSolve:
    """Test class for solve functions"""

    @classmethod
    def setup_class(cls):
        """Initialize Julia interface once for all tests"""
        cls.jl = init_julia(max_iter=100)

    @pytest.mark.parametrize(
        "case_name,pf_fast",
        [
            ("case24_ieee_rts", True),
            ("case24_ieee_rts", False),
            pytest.param(
                "case57_ieee",
                True,
                marks=pytest.mark.skipif(
                    SKIP_LARGE_GRIDS,
                    reason="Skipping large grids",
                ),
            ),
            pytest.param(
                "case57_ieee",
                False,
                marks=pytest.mark.skipif(
                    SKIP_LARGE_GRIDS,
                    reason="Skipping arge grids",
                ),
            ),
            pytest.param(
                "case118_ieee",
                True,
                marks=pytest.mark.skipif(
                    SKIP_LARGE_GRIDS,
                    reason="Skipping large grids",
                ),
            ),
            pytest.param(
                "case118_ieee",
                False,
                marks=pytest.mark.skipif(
                    SKIP_LARGE_GRIDS,
                    reason="Skipping arge grids",
                ),
            ),
            pytest.param(
                "case300_ieee",
                True,
                marks=pytest.mark.skipif(
                    SKIP_LARGE_GRIDS,
                    reason="Skipping large grids",
                ),
            ),
            pytest.param(
                "case300_ieee",
                False,
                marks=pytest.mark.skipif(
                    SKIP_LARGE_GRIDS,
                    reason="Skipping large grids",
                ),
            ),
            pytest.param(
                "case2000_goc",
                True,
                marks=pytest.mark.skipif(
                    SKIP_LARGE_GRIDS,
                    reason="Skipping large grids",
                ),
            ),
            pytest.param(
                "case2000_goc",
                False,
                marks=pytest.mark.skipif(
                    SKIP_LARGE_GRIDS,
                    reason="Skipping large grids",
                ),
            ),
            pytest.param(
                "case10000_goc",
                False,
                marks=pytest.mark.skipif(
                    SKIP_LARGE_GRIDS,
                    reason="Skipping large grids",
                ),
            ),  # fast PF not supported
        ],
    )
    def test_complete_workflow(self, case_name, pf_fast):
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
        opf_pf_data = pf_post_processing(0, net, opf_result, None, include_dc_res=False)
        assert "bus" in opf_pf_data
        assert "gen" in opf_pf_data
        assert "branch" in opf_pf_data
        assert "Y_bus" in opf_pf_data
        print(
            f"  OPF post-processing completed: bus={opf_pf_data['bus'].shape}, gen={opf_pf_data['gen'].shape}, branch={opf_pf_data['branch'].shape}",
        )

        # Step 3: Run PF
        net = pf_preprocessing(net, opf_result)
        pf_result = run_pf(net, self.jl, fast=pf_fast)
        start_time = time.time()
        print(f"  Running PF (fast={pf_fast})...")
        pf_result = run_pf(net, self.jl, fast=pf_fast)
        end_time = time.time()
        print(f"  PF time: {end_time - start_time} seconds")
        print("  PF converged successfully")

        # Step 4: PF post-processing on PF results
        print("  PF post-processing PF results...")
        pf_pf_data = pf_post_processing(0, net, pf_result, None, include_dc_res=False)
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

        print(f"{case_name} completed successfully")

    @pytest.mark.parametrize(
        "case_name",
        [
            "case24_ieee_rts",
            pytest.param(
                "case57_ieee",
                marks=pytest.mark.skipif(
                    SKIP_LARGE_GRIDS,
                    reason="Skipping large grids",
                ),
            ),
            pytest.param(
                "case118_ieee",
                marks=pytest.mark.skipif(
                    SKIP_LARGE_GRIDS,
                    reason="Skipping large grids",
                ),
            ),
        ],
    )
    def test_dc_opf_and_dc_pf_comprehensive(self, case_name):
        """Comprehensive test for DC OPF and DC PF functionality

        Tests:
        1. DC OPF - checks Vm=1.0, Va=0 at slack, qt/qf are NaN
        2. AC OPF + DC PF - checks Vm=1.0, Va=0 at slack, pg unchanged except slack
        3. Comparison of run_dcpf vs direct Julia compute_dc_pf
        """
        print(f"\nTesting DC OPF and DC PF comprehensively for {case_name}...")

        # Load network
        net = load_net_from_pglib(case_name)
        n_buses = net.buses.shape[0]
        print(
            f"  Loaded {case_name}: {net.buses.shape[0]} buses, {net.gens.shape[0]} gens, {net.branches.shape[0]} branches",
        )

        # ====== PART 1: Test DC OPF ======
        print("\n  ====== PART 1: Testing DC OPF ======")

        # Run DC OPF
        print("  Running DC OPF...")
        dcopf_result = run_dcopf(net, self.jl)
        assert str(dcopf_result["termination_status"]) == "LOCALLY_SOLVED", (
            "DC OPF should converge"
        )
        print("DC OPF converged successfully")
        opf_result = run_opf(net, self.jl)
        assert str(opf_result["termination_status"]) == "LOCALLY_SOLVED", (
            "OPF should converge"
        )
        print("OPF converged successfully")

        # Check Vm is 1.0 at all buses
        vm = [
            dcopf_result["solution"]["bus"][str(net.reverse_bus_index_mapping[i])]["vm"]
            for i in range(n_buses)
        ]
        assert np.isclose(vm, 1.0).all(), (
            "Vm should be 1.0 at all buses when solving with DCPF"
        )
        print("All buses have Vm = 1.0")

        # Check Va is 0 at slack bus
        va = [
            dcopf_result["solution"]["bus"][str(net.reverse_bus_index_mapping[i])]["va"]
            for i in range(n_buses)
        ]
        slack_bus_idx = net.ref_bus_idx
        assert np.isclose(va[slack_bus_idx], 0.0).all(), "Slack bus Va should be 0"
        print("Slack bus Va = 0")

        # Check qt and qf are NaN for all branches
        print("  Checking qt and qf are NaN for all branches...")
        for i in net.idx_branches_in_service:
            branch_key = str(i + 1)
            assert np.isnan(dcopf_result["solution"]["branch"][branch_key]["qf"]), (
                f"Branch {i + 1} qf should be NaN"
            )
            assert np.isnan(dcopf_result["solution"]["branch"][branch_key]["qt"]), (
                f"Branch {i + 1} qt should be NaN"
            )
        print(
            "qt and qf are NaN for all branches (DC OPF doesn't compute reactive power)",
        )

        # Post-process DC OPF results and ensure no NaNs in outputs
        print("  Post-processing DC OPF results and checking for NaNs...")
        dcopf_pf_data = pf_post_processing(
            0,
            net,
            opf_result,
            dcopf_result,
            include_dc_res=True,
        )
        for key in ["bus", "gen", "branch", "Y_bus"]:
            arr = dcopf_pf_data[key]
            assert not np.isnan(arr).any(), (
                f"NaN detected in {key} data after DC OPF post-processing"
            )
        print("No NaNs in DC OPF post-processed outputs")

        # ====== PART 2: Test AC OPF + DC PF ======
        print("\n  ====== PART 2: Testing AC OPF + DC PF ======")

        # Run AC OPF
        print("  Running AC OPF...")
        opf_result = run_opf(net, self.jl)
        assert str(opf_result["termination_status"]) == "LOCALLY_SOLVED", (
            "AC OPF should converge"
        )
        print("AC OPF converged successfully")

        # Get generator Pg from AC OPF (before PF)
        pg_before_pf = {}
        for i in net.idx_gens_in_service:
            pg_before_pf[i] = (
                opf_result["solution"]["gen"][str(i + 1)]["pg"] * net.baseMVA
            )

        # PF preprocessing
        print("  Running PF preprocessing...")
        net_pf = pf_preprocessing(net, opf_result)
        print("PF preprocessing completed")

        # Run DC PF
        print("  Running DC PF...")
        dcpf_result = run_dcpf(net_pf, self.jl, fast=False)
        assert str(dcpf_result["termination_status"]) == "LOCALLY_SOLVED", (
            "DC PF should converge"
        )
        print("DC PF converged successfully")

        # Check Vm is 1.0 at all buses
        vm = [
            dcopf_result["solution"]["bus"][str(net.reverse_bus_index_mapping[i])]["vm"]
            for i in range(n_buses)
        ]
        assert np.isclose(vm, 1.0).all(), (
            "Vm should be 1.0 at all buses when solving with DCPF"
        )
        print("All buses have Vm = 1.0")

        # Check Va is 0 at slack bus
        va = [
            dcopf_result["solution"]["bus"][str(net.reverse_bus_index_mapping[i])]["va"]
            for i in range(n_buses)
        ]
        slack_bus_idx = net.ref_bus_idx
        assert np.isclose(va[slack_bus_idx], 0.0).all(), "Slack bus Va should be 0"
        print("Slack bus Va = 0")

        # Check qt and qf are NaN for all branches
        print("  Checking qt and qf are NaN for all branches...")
        for i in net.idx_branches_in_service:
            branch_key = str(i + 1)
            assert np.isnan(dcopf_result["solution"]["branch"][branch_key]["qf"]), (
                f"Branch {i + 1} qf should be NaN"
            )
            assert np.isnan(dcopf_result["solution"]["branch"][branch_key]["qt"]), (
                f"Branch {i + 1} qt should be NaN"
            )
        print(
            "qt and qf are NaN for all branches (DC OPF doesn't compute reactive power)",
        )

        # Check pg has not changed at all gens except slack gen
        print("  Checking Pg unchanged except at slack gen...")
        slack_gen_indices = np.where(net_pf.gens[:, GEN_BUS] == net_pf.ref_bus_idx)[
            0
        ]  # GEN_BUS == ref_bus_idx

        for i in net_pf.idx_gens_in_service:
            gen_key = str(i + 1)
            pg_dc = dcpf_result["solution"]["gen"][gen_key]["pg"] * net_pf.baseMVA
            pg_before = pg_before_pf[i]
            if i not in slack_gen_indices:
                assert np.isclose(pg_dc, pg_before, atol=1e-3), (
                    f"Gen {i + 1} (non-slack): Pg should be unchanged. "
                    f"Before PF: {pg_before} MW, After DC PF: {pg_dc} MW"
                )
            else:
                print(
                    f"    Gen {i + 1} is slack gen - Pg allowed to change (DC PF: {pg_dc} MW, before PF: {pg_before} MW)",
                )
        print("Pg unchanged at all gens except slack gen")

        # Post-process DC PF results and ensure no NaNs in outputs
        print("  Post-processing DC PF results and checking for NaNs...")
        dcpf_pf_data = pf_post_processing(
            0,
            net_pf,
            opf_result,
            dcpf_result,
            include_dc_res=True,
        )
        for key in ["bus", "gen", "branch", "Y_bus"]:
            arr = dcpf_pf_data[key]
            assert not np.isnan(arr).any(), (
                f"NaN detected in {key} data after DC PF post-processing"
            )
        print("No NaNs in DC PF post-processed outputs")

        # ====== PART 3: Test DC PF fast ======
        # Run DC PF
        print("  Running DC PF...")
        dcpf_result = run_dcpf(net_pf, self.jl, fast=True)
        assert str(dcpf_result["termination_status"]) == "True", "DC PF should converge"
        print("DC PF converged successfully")

        # Check Vm is 1.0 at all buses
        vm = [
            dcopf_result["solution"]["bus"][str(net.reverse_bus_index_mapping[i])]["vm"]
            for i in range(n_buses)
        ]
        assert np.isclose(vm, 1.0).all(), (
            "Vm should be 1.0 at all buses when solving with DCPF"
        )
        print("All buses have Vm = 1.0")

        # Check Va is 0 at slack bus
        va = [
            dcopf_result["solution"]["bus"][str(net.reverse_bus_index_mapping[i])]["va"]
            for i in range(n_buses)
        ]
        slack_bus_idx = net.ref_bus_idx
        assert np.isclose(va[slack_bus_idx], 0.0).all(), "Slack bus Va should be 0"
        print("Slack bus Va = 0")

        # Check qt and qf are NaN for all branches
        print("  Checking qt and qf are NaN for all branches...")
        for i in net.idx_branches_in_service:
            branch_key = str(i + 1)
            assert np.isnan(dcopf_result["solution"]["branch"][branch_key]["qf"]), (
                f"Branch {i + 1} qf should be NaN"
            )
            assert np.isnan(dcopf_result["solution"]["branch"][branch_key]["qt"]), (
                f"Branch {i + 1} qt should be NaN"
            )
        print(
            "qt and qf are NaN for all branches (DC OPF doesn't compute reactive power)",
        )

        # Post-process DC PF results and ensure no NaNs in outputs
        print("  Post-processing DC PF results and checking for NaNs...")
        dcpf_pf_data = pf_post_processing(
            0,
            net_pf,
            opf_result,
            dcpf_result,
            include_dc_res=True,
        )
        for key in ["bus", "gen", "branch", "Y_bus"]:
            arr = dcpf_pf_data[key]
            assert not np.isnan(arr).any(), (
                f"NaN detected in {key} data after DC PF post-processing"
            )
        print("No NaNs in DC PF post-processed outputs")

        # Check pg has not changed at all gens except slack gen
        print("  Checking Pg unchanged except at slack gen...")
        slack_gen_indices = np.where(net_pf.gens[:, GEN_BUS] == net_pf.ref_bus_idx)[
            0
        ]  # GEN_BUS == ref_bus_idx

        for i in net_pf.idx_gens_in_service:
            pg_dc = dcpf_pf_data["gen"][
                i,
                len(GEN_COLUMNS) + DC_GEN_COLUMNS.index("p_mw_dc"),
            ]
            pg_before = pg_before_pf[i]
            if i not in slack_gen_indices:
                assert np.isclose(pg_dc, pg_before, atol=1e-3), (
                    f"Gen {i + 1} (non-slack): Pg should be unchanged. "
                    f"Before PF: {pg_before} MW, After DC PF: {pg_dc} MW"
                )
            else:
                print(
                    f"    Gen {i + 1} is slack gen - Pg allowed to change (DC PF: {pg_dc} MW, before PF: {pg_before} MW)",
                )
        print("Pg unchanged at all gens except slack gen")

    @pytest.mark.parametrize(
        "case_name",
        [
            ("case57_ieee"),
        ],
    )
    def test_dc_results_nan_when_nonconverged(self, case_name):
        """When DC PF and DC OPF do not converge, DC columns must be NaN.

        Force DC non-convergence by setting dc_max_iter=0 while keeping AC max_iter=150.
        """
        # init separate Julia instance with dc_max_iter=0
        jl_dc0 = init_julia(max_iter=150, dc_max_iter=0)

        # Load network
        net = load_net_from_pglib(case_name)

        # Run AC OPF (should converge with 150 iters)
        opf_result = run_opf(net, jl_dc0)
        assert str(opf_result["termination_status"]) == "LOCALLY_SOLVED"

        # DC OPF should not converge (dc_max_iter=0)
        dcopf_result = None
        try:
            dcopf_result = run_dcopf(net, jl_dc0)
            assert False, "DC OPF should not converge"
        except Exception as e:
            # check exception is runtime
            assert isinstance(e, RuntimeError)
            # check ITERATION_LIMIT in termination_status
            assert "ITERATION_LIMIT" in str(e)

        # Post-process with include_dc_res=True and verify DC columns are NaN
        pf_data_opf_mode = pf_post_processing(
            0,
            net,
            opf_result,
            dcopf_result,
            include_dc_res=True,
        )

        # Bus DC column (Va_dc) should be all NaN
        va_dc_idx = len(BUS_COLUMNS) + DC_BUS_COLUMNS.index("Va_dc")
        va_dc_col = pf_data_opf_mode["bus"][:, va_dc_idx]
        assert np.isnan(va_dc_col).all()

        # Gen DC column (p_mw_dc) should be NaN for gens in service
        p_mw_dc_idx = len(GEN_COLUMNS) + DC_GEN_COLUMNS.index("p_mw_dc")
        p_mw_dc_col = pf_data_opf_mode["gen"][:, p_mw_dc_idx]
        assert np.isnan(p_mw_dc_col[net.idx_gens_in_service]).all()

        # Branch DC columns (pf_dc, pt_dc) should be NaN for branches in service
        pf_dc_idx = len(BRANCH_COLUMNS) + DC_BRANCH_COLUMNS.index("pf_dc")
        pt_dc_idx = len(BRANCH_COLUMNS) + DC_BRANCH_COLUMNS.index("pt_dc")
        pf_dc_col = pf_data_opf_mode["branch"][:, pf_dc_idx]
        pt_dc_col = pf_data_opf_mode["branch"][:, pt_dc_idx]
        assert np.isnan(pf_dc_col[net.idx_branches_in_service]).all()
        assert np.isnan(pt_dc_col[net.idx_branches_in_service]).all()

        # runtime
        dc_idx = len(RUNTIME_COLUMNS) + DC_RUNTIME_COLUMNS.index("dc")
        runtime_col_opf = pf_data_opf_mode["runtime"][:, dc_idx]
        assert np.isnan(runtime_col_opf).all()
        print("DC columns are NaN when DC OPF and DC PF do not converge")

        # Also test PF mode: run AC PF (converges), DC PF (non-converged), then check DC NaNs
        net_pf = pf_preprocessing(net, opf_result)
        pf_result = run_pf(net_pf, jl_dc0, fast=True)
        assert (pf_result["termination_status"]) or (
            str(pf_result["termination_status"]) == "LOCALLY_SOLVED"
        )

        # DC OPF should not converge (dc_max_iter=0)
        dcpf_result = None
        try:
            dcpf_result = run_dcpf(net_pf, jl_dc0)
            assert False, "DC OPF should not converge"
        except Exception as e:
            # check exception is runtime
            assert isinstance(e, RuntimeError)
            # check ITERATION_LIMIT in termination_status
            assert "ITERATION_LIMIT" in str(e)

        pf_data_pf_mode = pf_post_processing(
            0,
            net_pf,
            pf_result,
            dcpf_result,
            include_dc_res=True,
        )

        va_dc_idx = len(BUS_COLUMNS) + DC_BUS_COLUMNS.index("Va_dc")
        va_dc_col_pf = pf_data_pf_mode["bus"][:, va_dc_idx]
        assert np.isnan(va_dc_col_pf).all()

        p_mw_dc_idx = len(GEN_COLUMNS) + DC_GEN_COLUMNS.index("p_mw_dc")
        p_mw_dc_col_pf = pf_data_pf_mode["gen"][:, p_mw_dc_idx]
        assert np.isnan(p_mw_dc_col_pf[net_pf.idx_gens_in_service]).all()

        pf_dc_idx = len(BRANCH_COLUMNS) + DC_BRANCH_COLUMNS.index("pf_dc")
        pt_dc_idx = len(BRANCH_COLUMNS) + DC_BRANCH_COLUMNS.index("pt_dc")
        pf_dc_col_pf = pf_data_pf_mode["branch"][:, pf_dc_idx]
        pt_dc_col_pf = pf_data_pf_mode["branch"][:, pt_dc_idx]
        assert np.isnan(pf_dc_col_pf[net_pf.idx_branches_in_service]).all()
        assert np.isnan(pt_dc_col_pf[net_pf.idx_branches_in_service]).all()

        # runtime
        dc_idx = len(RUNTIME_COLUMNS) + DC_RUNTIME_COLUMNS.index("dc")
        runtime_col_pf = pf_data_pf_mode["runtime"][:, dc_idx]
        assert np.isnan(runtime_col_pf).all()

        print("DC columns are NaN when DC PF and DC OPF do not converge")
        print(f"{case_name} completed successfully")

    # skip this test
    @pytest.mark.skip(
        reason="TODO: remove slow dc pf as fast works better in pretty much all cases",
    )
    def test_dcpf_fast_matches_slow(self):
        """Fast DC PF should match slow DC PF after post-processing."""
        # TODO: remove slow dc pf if all tests work with fast dc pf
        case_name = "case14_ieee"  # that test fails for case300 (and possibly other cases). compute_dc_pf and solve_dc_pf from powermodels
        # do not actually give the same solution
        print(f"\nTesting DC PF fast vs slow for {case_name}...")

        # Load network and run AC OPF
        net = load_net_from_pglib(case_name)
        opf_result = run_opf(net, self.jl)
        assert str(opf_result["termination_status"]) == "LOCALLY_SOLVED"

        # Prepare network for PF
        net_pf = pf_preprocessing(net, opf_result)

        # Run slow and fast DC PF
        print("  Running DC PF (slow)...")
        dcpf_slow = run_dcpf(net_pf, self.jl, fast=False)
        assert str(dcpf_slow["termination_status"]) == "LOCALLY_SOLVED"

        print("  Running DC PF (fast)...")
        dcpf_fast = run_dcpf(net_pf, self.jl, fast=True)
        assert str(dcpf_fast["termination_status"]) == "True"

        # Post-process and compare outputs
        print("  Post-processing and comparing outputs...")
        pf_data_slow = pf_post_processing(
            0,
            net_pf,
            opf_result,
            dcpf_slow,
            include_dc_res=True,
        )
        pf_data_fast = pf_post_processing(
            0,
            net_pf,
            opf_result,
            dcpf_fast,
            include_dc_res=True,
        )

        for key in ["bus", "gen", "branch", "Y_bus"]:
            arr_slow = pf_data_slow[key]
            arr_fast = pf_data_fast[key]
            assert arr_slow.shape == arr_fast.shape, f"Shape mismatch for key '{key}'"
            assert np.allclose(arr_slow, arr_fast, equal_nan=True), (
                f"Mismatch for key '{key}'"
            )

        print("Fast and slow DC PF results match after post-processing")

    @pytest.mark.parametrize(
        "case_name",
        [
            "case24_ieee_rts",
            pytest.param(
                "case57_ieee",
                marks=pytest.mark.skipif(
                    SKIP_LARGE_GRIDS,
                    reason="Skipping large grids",
                ),
            ),
            pytest.param(
                "case118_ieee",
                marks=pytest.mark.skipif(
                    SKIP_LARGE_GRIDS,
                    reason="Skipping large grids",
                ),
            ),
        ],
    )
    def test_dc_columns_nan_when_res_dc_none(self, case_name):
        """Test that when res_dc=None and include_dc_res=True, all DC columns are NaN,
        and that NaN values can be properly read from parquet after saving.

        This simulates the case where DC power flow did not converge (res_dc=None)
        but include_dc_res=True, so DC columns should be present but all NaN.
        """
        print(f"\nTesting DC columns are NaN when res_dc=None for {case_name}...")

        # Load network
        net = load_net_from_pglib(case_name)
        print(
            f"  Loaded {case_name}: {net.buses.shape[0]} buses, {net.gens.shape[0]} gens, {net.branches.shape[0]} branches",
        )

        # Run AC OPF to get a valid res
        print("  Running AC OPF...")
        opf_result = run_opf(net, self.jl)
        assert str(opf_result["termination_status"]) == "LOCALLY_SOLVED"
        print("  AC OPF converged successfully")

        # Call pf_post_processing with res_dc=None and include_dc_res=True
        print("  Post-processing with res_dc=None and include_dc_res=True...")
        pf_data = pf_post_processing(0, net, opf_result, None, include_dc_res=True)

        # Check that all DC columns are NaN in memory
        print("  Checking DC columns are NaN in memory...")

        # Bus DC columns: Va_dc, Pg_dc
        va_dc_idx = len(BUS_COLUMNS) + DC_BUS_COLUMNS.index("Va_dc")
        pg_dc_idx = len(BUS_COLUMNS) + DC_BUS_COLUMNS.index("Pg_dc")
        va_dc_col = pf_data["bus"][:, va_dc_idx]
        pg_dc_col = pf_data["bus"][:, pg_dc_idx]
        assert np.isnan(va_dc_col).all(), "Va_dc should be all NaN when res_dc=None"
        assert np.isnan(pg_dc_col).all(), "Pg_dc should be all NaN when res_dc=None"
        print("    Bus DC columns are all NaN")

        # Gen DC columns: p_mw_dc
        p_mw_dc_idx = len(GEN_COLUMNS) + DC_GEN_COLUMNS.index("p_mw_dc")
        p_mw_dc_col = pf_data["gen"][:, p_mw_dc_idx]
        assert np.isnan(p_mw_dc_col[net.idx_gens_in_service]).all(), (
            "p_mw_dc should be all NaN for in-service gens when res_dc=None"
        )
        print("    Gen DC columns are all NaN for in-service generators")

        # Branch DC columns: pf_dc, pt_dc
        pf_dc_idx = len(BRANCH_COLUMNS) + DC_BRANCH_COLUMNS.index("pf_dc")
        pt_dc_idx = len(BRANCH_COLUMNS) + DC_BRANCH_COLUMNS.index("pt_dc")
        pf_dc_col = pf_data["branch"][:, pf_dc_idx]
        pt_dc_col = pf_data["branch"][:, pt_dc_idx]
        assert np.isnan(pf_dc_col[net.idx_branches_in_service]).all(), (
            "pf_dc should be all NaN for in-service branches when res_dc=None"
        )
        assert np.isnan(pt_dc_col[net.idx_branches_in_service]).all(), (
            "pt_dc should be all NaN for in-service branches when res_dc=None"
        )
        print("    Branch DC columns are all NaN for in-service branches")

        # Runtime DC columns: dc
        dc_idx = len(RUNTIME_COLUMNS) + DC_RUNTIME_COLUMNS.index("dc")
        runtime_dc_col = pf_data["runtime"][:, dc_idx]
        assert np.isnan(runtime_dc_col).all(), (
            "dc runtime should be all NaN when res_dc=None"
        )
        print("    Runtime DC column is all NaN")

        # Now save to parquet and read back to verify NaN persistence
        print("  Saving to parquet and reading back to verify NaN persistence...")
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create DataFrames with proper column names
            bus_columns = BUS_COLUMNS + DC_BUS_COLUMNS
            gen_columns = GEN_COLUMNS + DC_GEN_COLUMNS
            branch_columns = BRANCH_COLUMNS + DC_BRANCH_COLUMNS
            runtime_columns = RUNTIME_COLUMNS + DC_RUNTIME_COLUMNS

            bus_df = pd.DataFrame(pf_data["bus"], columns=bus_columns)
            bus_df["bus"] = bus_df["bus"].astype("int64")
            bus_df.insert(0, "scenario", 0)

            gen_df = pd.DataFrame(pf_data["gen"], columns=gen_columns)
            gen_df["bus"] = gen_df["bus"].astype("int64")
            gen_df.insert(0, "scenario", 0)

            branch_df = pd.DataFrame(pf_data["branch"], columns=branch_columns)
            branch_df[["from_bus", "to_bus"]] = branch_df[
                ["from_bus", "to_bus"]
            ].astype("int64")
            branch_df.insert(0, "scenario", 0)

            runtime_df = pd.DataFrame(pf_data["runtime"], columns=runtime_columns)
            runtime_df.insert(0, "scenario", 0)

            # Save to partitioned parquet
            bus_path = os.path.join(tmpdir, "bus_data.parquet")
            gen_path = os.path.join(tmpdir, "gen_data.parquet")
            branch_path = os.path.join(tmpdir, "branch_data.parquet")
            runtime_path = os.path.join(tmpdir, "runtime_data.parquet")

            # Add partition column for scenario-based partitioning (n_scenario_per_partition scenarios per partition)
            bus_df["scenario_partition"] = (
                bus_df["scenario"] // n_scenario_per_partition
            ).astype("int64")
            gen_df["scenario_partition"] = (
                gen_df["scenario"] // n_scenario_per_partition
            ).astype("int64")
            branch_df["scenario_partition"] = (
                branch_df["scenario"] // n_scenario_per_partition
            ).astype(
                "int64",
            )
            runtime_df["scenario_partition"] = (
                runtime_df["scenario"] // n_scenario_per_partition
            ).astype(
                "int64",
            )

            bus_df.to_parquet(
                bus_path,
                partition_cols=["scenario_partition"],
                engine="pyarrow",
                index=False,
            )
            gen_df.to_parquet(
                gen_path,
                partition_cols=["scenario_partition"],
                engine="pyarrow",
                index=False,
            )
            branch_df.to_parquet(
                branch_path,
                partition_cols=["scenario_partition"],
                engine="pyarrow",
                index=False,
            )
            runtime_df.to_parquet(
                runtime_path,
                partition_cols=["scenario_partition"],
                engine="pyarrow",
                index=False,
            )

            # Read back from parquet
            bus_df_read = pd.read_parquet(bus_path, engine="pyarrow")
            gen_df_read = pd.read_parquet(gen_path, engine="pyarrow")
            branch_df_read = pd.read_parquet(branch_path, engine="pyarrow")
            runtime_df_read = pd.read_parquet(runtime_path, engine="pyarrow")

            # Verify NaN values are preserved after round-trip
            print("  Verifying NaN values are preserved after parquet round-trip...")

            # Bus DC columns
            assert bus_df_read["Va_dc"].isna().all(), (
                "Va_dc should remain all NaN after parquet round-trip"
            )
            assert bus_df_read["Pg_dc"].isna().all(), (
                "Pg_dc should remain all NaN after parquet round-trip"
            )
            print("    Bus DC columns remain all NaN after parquet round-trip")

            # Gen DC columns (check in-service gens)
            in_service_mask = gen_df_read["in_service"] == 1
            assert gen_df_read.loc[in_service_mask, "p_mw_dc"].isna().all(), (
                "p_mw_dc should remain all NaN for in-service gens after parquet round-trip"
            )
            print(
                "    Gen DC columns remain all NaN for in-service generators after parquet round-trip",
            )

            # Branch DC columns (check in-service branches)
            in_service_mask = branch_df_read["br_status"] == 1
            assert branch_df_read.loc[in_service_mask, "pf_dc"].isna().all(), (
                "pf_dc should remain all NaN for in-service branches after parquet round-trip"
            )
            assert branch_df_read.loc[in_service_mask, "pt_dc"].isna().all(), (
                "pt_dc should remain all NaN for in-service branches after parquet round-trip"
            )
            print(
                "    Branch DC columns remain all NaN for in-service branches after parquet round-trip",
            )

            # Runtime DC columns
            assert runtime_df_read["dc"].isna().all(), (
                "dc runtime should remain all NaN after parquet round-trip"
            )
            print("    Runtime DC column remains all NaN after parquet round-trip")

        print(
            f"  {case_name} completed successfully: NaN values persist through parquet save/load",
        )


if __name__ == "__main__":
    test = TestSolve()
    # set up julia interface
    os.makedirs("solver_log", exist_ok=True)
    test.jl = init_julia(max_iter=300, solver_log_dir="solver_log")
    test.test_dcpf_fast_matches_slow()
