"""
Verification tests to check network operations consistency with PowerModels.
Converted from scripts/verify_network.py to run under pytest.
"""

import tempfile
import os
from juliacall import Main as jl
from gridfm_datakit.network import (
    load_net_from_pglib,
    F_BUS,
    T_BUS,
    GEN_BUS,
)
from gridfm_datakit.process.process_network import init_julia


def verify_deactivated_branches():
    """Check that deactivated branches don't appear in PowerModels results."""
    print("\n1. Testing branch deactivation...")

    case_name = "case24_ieee_rts"  # because case300 doesnt converge for PF
    net = load_net_from_pglib(case_name)

    # Deactivate a branch and save to a temporary .m file
    branch_to_deactivate = 2
    net.deactivate_branches([branch_to_deactivate])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    try:
        net.to_mpc(tmp_path)

        # Parse with PowerModels and check br_status
        parse = jl.PowerModels.parse_file(tmp_path)
        deactivated_branch_pm_idx = str(
            branch_to_deactivate + 1,
        )  # 1-indexed in PowerModels

        # Check that br_status is 0 in parsed file
        assert parse["branch"][deactivated_branch_pm_idx]["br_status"] == 0, (
            f"Branch {branch_to_deactivate} should have br_status = 0"
        )

        print(
            f"   ✓ Deactivated branch {branch_to_deactivate} has br_status = 0 in parsed file",
        )

        # Solve with PowerModels
        result = jl.run_opf(tmp_path)

        # Check branch doesn't appear in results
        assert deactivated_branch_pm_idx not in result["solution"]["branch"], (
            f"Branch {branch_to_deactivate} should not appear in results"
        )

        print(
            f"   ✓ Deactivated branch {branch_to_deactivate} doesn't appear in results",
        )

        result = jl.run_pf(tmp_path)
        assert (
            str(result["solution"]["branch"][deactivated_branch_pm_idx]["pf"]) == "nan"
        ), f"Branch {branch_to_deactivate} should have NaN flow after pf"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def verify_branch_direction():
    """Check that branch direction matches between net and PowerModels."""
    print("\n2. Testing branch direction consistency...")

    case_name = "case300_ieee"
    net = load_net_from_pglib(case_name)

    # Parse with PowerModels
    parse = jl.PowerModels.parse_file(f"gridfm_datakit/grids/pglib_opf_{case_name}.m")

    active_branches = net.idx_branches_in_service
    for branch_idx in active_branches:
        pm_branch_data = parse["branch"][str(branch_idx + 1)]
        net_f_bus = int(net.branches[branch_idx, F_BUS])
        net_t_bus = int(net.branches[branch_idx, T_BUS])
        pm_f_bus = pm_branch_data["f_bus"]
        pm_t_bus = pm_branch_data["t_bus"]

        assert net.reverse_bus_index_mapping[net_f_bus] == pm_f_bus, (
            f"From bus mismatch at branch {branch_idx}"
        )
        assert net.reverse_bus_index_mapping[net_t_bus] == pm_t_bus, (
            f"To bus mismatch at branch {branch_idx}"
        )

    print("   ✓ Branch directions match between Network and PowerModels")


def verify_deactivated_generators():
    """Check that deactivated generators don't appear in PowerModels results."""
    print("\n3. Testing generator deactivation...")

    case_name = "case24_ieee_rts"
    net = load_net_from_pglib(case_name)

    # Deactivate a generator and save to a temporary .m file
    gen_to_deactivate = 3
    net.deactivate_gens([gen_to_deactivate])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    try:
        net.to_mpc(tmp_path)

        # Parse with PowerModels and check gen_status
        parse = jl.PowerModels.parse_file(tmp_path)
        deactivated_gen_pm_idx = str(gen_to_deactivate + 1)  # 1-indexed in PowerModels

        # Check that gen_status is 0 in parsed file
        assert parse["gen"][deactivated_gen_pm_idx]["gen_status"] == 0, (
            f"Generator {gen_to_deactivate} should have gen_status = 0"
        )

        print(
            f"   ✓ Deactivated generator {gen_to_deactivate} has gen_status = 0 in parsed file",
        )

        # Solve with PowerModels
        result = jl.run_opf(tmp_path)

        # Check generator doesn't appear in results
        assert deactivated_gen_pm_idx not in result["solution"]["gen"], (
            f"Generator {gen_to_deactivate} should not appear in results"
        )

        print(
            f"   ✓ Deactivated generator {gen_to_deactivate} doesn't appear in results",
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def verify_generator_bus_assignment():
    """Check that generator bus assignments match between net and PowerModels."""
    print("\n4. Testing generator bus assignment...")

    case_name = "case24_ieee_rts"
    net = load_net_from_pglib(case_name)

    # Parse with PowerModels
    parse = jl.PowerModels.parse_file(f"gridfm_datakit/grids/pglib_opf_{case_name}.m")

    # Check each generator's bus assignment
    for gen_idx in range(net.gens.shape[0]):
        pm_gen_idx = str(gen_idx + 1)  # 1-indexed in PowerModels
        pm_gen_data = parse["gen"][pm_gen_idx]

        # Get Network's bus index for this generator (using reverse mapping)
        net_gen_bus_idx = int(net.gens[gen_idx, GEN_BUS])
        net_gen_bus_original = net.reverse_bus_index_mapping[net_gen_bus_idx]

        # Get PowerModels bus index
        pm_gen_bus = pm_gen_data["gen_bus"]

        assert net_gen_bus_original == pm_gen_bus, (
            f"Generator {gen_idx}: Network bus {net_gen_bus_original} != PowerModels bus {pm_gen_bus}"
        )

    print(f"   ✓ All {net.gens.shape[0]} generators have matching bus assignments")


class TestVerifyNetwork:
    @classmethod
    def setup_class(cls):
        # Initialize Julia so PowerModels functions are available via juliacall Main
        init_julia(max_iter=150)

    def test_deactivated_branches(self):
        verify_deactivated_branches()

    def test_branch_direction(self):
        verify_branch_direction()

    def test_deactivated_generators(self):
        verify_deactivated_generators()

    def test_generator_bus_assignment(self):
        verify_generator_bus_assignment()
