"""
Data validation functions for power flow data integrity and consistency checks.

This module contains all validation functions extracted from test_data_validation.py
to provide comprehensive validation of generated power flow data.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Iterable
from gridfm_datakit.utils.power_balance import compute_branch_admittances
from gridfm_datakit.utils.column_names import (
    BUS_COLUMNS,
    DC_BUS_COLUMNS,
    BRANCH_COLUMNS,
    DC_BRANCH_COLUMNS,
    GEN_COLUMNS,
    DC_GEN_COLUMNS,
    YBUS_COLUMNS,
    RUNTIME_COLUMNS,
    DC_RUNTIME_COLUMNS,
)
import copy
from gridfm_datakit.utils.power_balance import (
    compute_branch_powers_vectorized,
    compute_bus_balance,
)
from gridfm_datakit.utils.utils import get_num_scenarios
from gridfm_datakit.utils.utils import read_partitions, n_scenario_per_partition


def _validate_partition_structure(
    file_paths: Dict[str, str],
    total_scenarios: int,
) -> None:
    """
    for each partitioned parquet directory (n_scenario_per_partition scenarios/partition),
    check that:
      - All partitions except the last contain 100 unique scenarios.
      - Last partition contains (total_scenarios % 100) unique scenarios (unless perfectly divided, then 100).
      - For each partition, minimum and maximum scenario numbers are as expected.
      - The entire span covers scenarios from 0 to total_scenarios-1.
      - Prints results for each check.
    """
    print("Validating partition structure (easy checks)...")
    partitions = (
        total_scenarios + n_scenario_per_partition - 1
    ) // n_scenario_per_partition  # ceiling division

    for k in range(partitions):
        expected_first = k * n_scenario_per_partition
        expected_last = min((k + 1) * n_scenario_per_partition, total_scenarios) - 1
        expect_count = expected_last - expected_first + 1

        # Construct partition path
        partition_path = os.path.join(file_paths["bus_data"], f"scenario_partition={k}")

        # Read all parquet files in this partition folder
        part = pd.read_parquet(partition_path, columns=["scenario"], engine="pyarrow")[
            "scenario"
        ]
        unique_scenarios = part.unique()
        min_scenario, max_scenario = np.min(unique_scenarios), np.max(unique_scenarios)
        if len(unique_scenarios) != expect_count:
            raise AssertionError(
                f"Partition {k}: found {len(unique_scenarios)} scenarios, expected {expect_count}",
            )

        if min_scenario != expected_first or max_scenario != expected_last:
            raise AssertionError(
                f"Partition {k}: scenario range is {min_scenario}-{max_scenario}, expected {expected_first}-{expected_last}",
            )

        print(
            f"  ✓ Partition {k:>2}: {len(unique_scenarios)} scenarios ({min_scenario}-{max_scenario}) OK",
        )


def validate_generated_data(
    file_paths: Dict[str, str],
    mode: str,
    sn_mva: float,
    n_partitions: int = 0,
) -> bool:
    """Run all validation tests on the generated data.

    Args:
        file_paths: Dictionary containing paths to data files (bus_data, branch_data, gen_data, y_bus_data).
        mode: Operating mode ("opf" or "pf").
        n_partitions: Number of partitions to sample for validation (0 for all partitions).
        sn_mva: Base MVA used to scale power quantities

    Returns:
        True if all validations pass.

    Raises:
        AssertionError: If any validation fails.
    """
    # Get total scenarios from metadata
    data_dir = os.path.dirname(file_paths["bus_data"])
    total_scenarios = get_num_scenarios(data_dir)

    # Step 1: Validate partition structure on ALL partitions
    # print("Step 1: Validating partition structure on all partitions...")
    # _validate_partition_structure(file_paths, total_scenarios)

    # Calculate number of partitions (n_scenario_per_partition scenarios per partition)
    num_partitions = (
        total_scenarios + n_scenario_per_partition - 1
    ) // n_scenario_per_partition

    # Sample partitions
    if n_partitions > 0:
        n_partitions_to_sample = n_partitions
        sampled_partitions = sorted(
            np.random.choice(
                num_partitions,
                size=min(n_partitions_to_sample, num_partitions),
                replace=False,
            ),
        )
        max_scenarios_to_validate = len(sampled_partitions) * n_scenario_per_partition
        print(
            f"Step 2: Running core validations on {len(sampled_partitions)} sampled partitions (up to {max_scenarios_to_validate} scenarios) out of {num_partitions} total",
        )
    else:
        sampled_partitions = list(range(num_partitions))
        print(
            f"Step 2: Running core validations on all {num_partitions} partitions ({total_scenarios} total scenarios)",
        )

    bus_data = read_partitions(file_paths["bus_data"], sampled_partitions)
    branch_data = read_partitions(file_paths["branch_data"], sampled_partitions)
    gen_data = read_partitions(file_paths["gen_data"], sampled_partitions)
    y_bus_data = read_partitions(file_paths["y_bus_data"], sampled_partitions)
    runtime_data = (
        read_partitions(file_paths["runtime_data"], sampled_partitions)
        if "runtime_data" in file_paths
        else None
    )
    if runtime_data is None:
        print("No runtime data found, skipping runtime data validation")
    else:
        print(f"Runtime data found: {runtime_data.shape}")

    generated_data = {
        "bus_data": bus_data,
        "branch_data": branch_data,
        "gen_data": gen_data,
        "y_bus_data": y_bus_data,
        "runtime_data": runtime_data,
        "mode": mode,
        "file_paths": file_paths,
    }

    # Run core validations on sampled partitions
    try:
        validate_scenario_indexing_consistency(generated_data)
    except Exception as e:
        raise AssertionError(f"Scenario indexing consistency validation failed: {e}")

    try:
        validate_bus_indexing_consistency(generated_data)
    except Exception as e:
        raise AssertionError(f"Bus indexing consistency validation failed: {e}")

    try:
        validate_data_completeness(generated_data)
    except Exception as e:
        raise AssertionError(f"Data completeness validation failed: {e}")

    try:
        validate_dc_columns_consistency(generated_data)
    except Exception as e:
        raise AssertionError(f"DC columns consistency validation failed: {e}")

    # Check voltage angles are within [-180, 180]
    try:
        validate_voltage_angles_within_bounds(generated_data)
    except Exception as e:
        raise AssertionError(f"Voltage angles validation failed: {e}")
    # Run Y-Bus Consistency Tests
    try:
        validate_ybus_diagonal_consistency(generated_data)
    except Exception as e:
        raise AssertionError(f"Y-bus diagonal consistency validation failed: {e}")

    # Run Branch Constraint Tests
    try:
        validate_deactivated_lines_zero_admittance(generated_data)
    except Exception as e:
        raise AssertionError(
            f"Deactivated lines zero admittance validation failed: {e}",
        )

    try:
        validate_admittance_calculations(generated_data)
    except Exception as e:
        raise AssertionError(
            f"Admittance calculations validation failed: {e}",
        )

    try:
        validate_computed_vs_stored_power_flows(generated_data, sn_mva)
    except Exception as e:
        raise AssertionError(f"Computed vs stored power flows validation failed: {e}")

    try:
        validate_tap_not_zero(generated_data)
    except Exception as e:
        raise AssertionError(f"Tap not zero validation failed: {e}")

    # Run branch loading validation for both OPF and PF modes
    # In OPF mode: asserts loading <= 1.01
    # In PF mode: computes statistics without asserting
    try:
        validate_branch_loading_opf_mode(generated_data)
    except Exception as e:
        if mode == "opf":
            raise AssertionError(f"Branch loading OPF mode validation failed: {e}")
        else:
            print(f"    Branch loading computation encountered errors: {e}")

    # Run Generator Constraint Tests
    try:
        validate_deactivated_generators_zero_output(generated_data)
    except Exception as e:
        raise AssertionError(
            f"Deactivated generators zero output validation failed: {e}",
        )

    try:
        validate_generator_limits(generated_data)
    except Exception as e:
        raise AssertionError(f"Generator limits validation failed: {e}")

    # Run OPF mode Constraints
    if mode == "opf":
        try:
            validate_voltage_magnitude_limits_opf_mode(generated_data)
        except Exception as e:
            raise AssertionError(
                f"Voltage magnitude limits OPF mode validation failed: {e}",
            )
        try:
            validate_branch_angle_difference_opf_mode(generated_data)
        except Exception as e:
            raise AssertionError(
                f"Branch angle difference limits OPF mode validation failed: {e}",
            )

    # Run Power Balance Tests
    try:
        validate_bus_generation_consistency(generated_data)
    except Exception as e:
        raise AssertionError(f"Bus generation consistency validation failed: {e}")

    # DC bus generation consistency (if DC fields present)
    try:
        validate_bus_generation_consistency_dc(generated_data)
    except Exception as e:
        raise AssertionError(f"Bus generation DC consistency validation failed: {e}")

    # Check Pg and Pg_dc match at slack nodes in PF mode
    if mode == "pf":
        try:
            validate_non_slack_pg_consistency(generated_data)
        except Exception as e:
            raise AssertionError(f"Slack Pg consistency validation failed: {e}")

    try:
        validate_power_balance_equations(generated_data, sn_mva)
    except Exception as e:
        raise AssertionError(f"Power balance equations validation failed: {e}")

    return True


def validate_ybus_diagonal_consistency(generated_data: Dict[str, pd.DataFrame]) -> None:
    """Test Y-bus diagonal consistency with bus and branch data (vectorized)."""
    bus_data = generated_data["bus_data"]
    branch_data = generated_data["branch_data"]
    y_bus_data = generated_data["y_bus_data"]

    scenarios = bus_data["scenario"].unique()
    total_buses = len(bus_data)
    print(
        f"    Y-bus diagonal consistency: validating {total_buses} bus entries across {len(scenarios)} scenarios",
    )

    # Aggregate Yff contributions by (scenario, from_bus)
    yff_sum = (
        branch_data.groupby(["scenario", "from_bus"], as_index=False)
        .agg({"Yff_r": "sum", "Yff_i": "sum"})
        .rename(columns={"from_bus": "bus", "Yff_r": "yff_sum_g", "Yff_i": "yff_sum_b"})
    )

    # Aggregate Ytt contributions by (scenario, to_bus)
    ytt_sum = (
        branch_data.groupby(["scenario", "to_bus"], as_index=False)
        .agg({"Ytt_r": "sum", "Ytt_i": "sum"})
        .rename(columns={"to_bus": "bus", "Ytt_r": "ytt_sum_g", "Ytt_i": "ytt_sum_b"})
    )

    # Prepare bus data with (scenario, bus) as key
    bus_keyed = bus_data[["scenario", "bus", "GS", "BS"]].copy()
    bus_keyed["scenario"] = bus_keyed["scenario"].astype(int)
    bus_keyed["bus"] = bus_keyed["bus"].astype(int)

    # Merge all contributions
    expected = (
        bus_keyed.merge(yff_sum, on=["scenario", "bus"], how="left")
        .merge(ytt_sum, on=["scenario", "bus"], how="left")
        .fillna(
            {"yff_sum_g": 0.0, "yff_sum_b": 0.0, "ytt_sum_g": 0.0, "ytt_sum_b": 0.0},
        )
    )

    # Compute expected G and B
    expected["expected_g"] = (
        expected["GS"] + expected["yff_sum_g"] + expected["ytt_sum_g"]
    )
    expected["expected_b"] = (
        expected["BS"] + expected["yff_sum_b"] + expected["ytt_sum_b"]
    )

    # Get actual G and B from y_bus_data (diagonal entries only)
    ybus_diagonal = y_bus_data[(y_bus_data["index1"] == y_bus_data["index2"])][
        ["scenario", "index1", "G", "B"]
    ].rename(columns={"index1": "bus"})
    ybus_diagonal["scenario"] = ybus_diagonal["scenario"].astype(int)
    ybus_diagonal["bus"] = ybus_diagonal["bus"].astype(int)

    # Merge expected with actual
    comparison = expected.merge(
        ybus_diagonal,
        on=["scenario", "bus"],
        how="left",
        suffixes=("", "_actual"),
    )

    # Vectorized comparison
    g_diff = np.abs(comparison["expected_g"] - comparison["G"])
    b_diff = np.abs(comparison["expected_b"] - comparison["B"])

    tolerance = 1e-6
    g_mismatches = comparison[g_diff >= tolerance]
    b_mismatches = comparison[b_diff >= tolerance]

    if len(g_mismatches) > 0:
        raise AssertionError(f"G mismatches: {g_mismatches}")
    if len(b_mismatches) > 0:
        raise AssertionError(f"B mismatches: {b_mismatches}")


def validate_deactivated_lines_zero_admittance(
    generated_data: Dict[str, pd.DataFrame],
) -> None:
    """Test that deactivated lines have zero power flows and admittances."""
    branch_data = generated_data["branch_data"]
    deactivated_branches = branch_data[branch_data["br_status"] == 0]

    print(
        f"    Deactivated lines zero admittance: validating {len(deactivated_branches)} deactivated branches",
    )
    if not deactivated_branches.empty:
        assert (deactivated_branches["pf"] == 0).all(), (
            "Deactivated branches should have zero pf"
        )
        assert (deactivated_branches["qf"] == 0).all(), (
            "Deactivated branches should have zero qf"
        )
        assert (deactivated_branches["pt"] == 0).all(), (
            "Deactivated branches should have zero pt"
        )
        assert (deactivated_branches["qt"] == 0).all(), (
            "Deactivated branches should have zero qt"
        )
        assert (deactivated_branches["Yff_r"] == 0).all(), (
            "Deactivated branches should have zero Yff_r"
        )
        assert (deactivated_branches["Yff_i"] == 0).all(), (
            "Deactivated branches should have zero Yff_i"
        )
        assert (deactivated_branches["Yft_r"] == 0).all(), (
            "Deactivated branches should have zero Yft_r"
        )
        assert (deactivated_branches["Yft_i"] == 0).all(), (
            "Deactivated branches should have zero Yft_i"
        )
        assert (deactivated_branches["Ytf_r"] == 0).all(), (
            "Deactivated branches should have zero Ytf_r"
        )
        assert (deactivated_branches["Ytf_i"] == 0).all(), (
            "Deactivated branches should have zero Ytf_i"
        )
        assert (deactivated_branches["Ytt_r"] == 0).all(), (
            "Deactivated branches should have zero Ytt_r"
        )
        assert (deactivated_branches["Ytt_i"] == 0).all(), (
            "Deactivated branches should have zero Ytt_i"
        )

    print("    Deactivated lines zero admittance: OK")


def validate_voltage_angles_within_bounds(
    generated_data: Dict[str, pd.DataFrame],
) -> None:
    """
    Validate that all bus voltage angles (Va, and Va_dc if present)
    are within [-180, 180] degrees for both PF and OPF scenarios.

    Raises:
        AssertionError: if any voltage angle is out of bounds.
    """
    bus_data = generated_data["bus_data"]
    scenarios = bus_data["scenario"].unique()
    print(
        f"    Voltage angles bounds check: validating {len(bus_data)} bus entries across {len(scenarios)} scenarios",
    )

    def _check_angle(col: str) -> None:
        values = bus_data[col]
        within_bounds = values.isna() | (
            (values >= -180 - 1e-6) & (values <= 180 + 1e-6)
        )
        if not within_bounds.all():
            out_of_bounds = bus_data.loc[~within_bounds, ["scenario", "bus", col]]
            raise AssertionError(
                f"{col} angles out of bounds [-180, 180]:\n{out_of_bounds}",
            )

    # AC voltage angles
    _check_angle("Va")

    # DC voltage angles (if present)
    if "Va_dc" in bus_data.columns:
        print("    DC voltage angles (Va_dc) present, performing check for Va_dc")
        _check_angle("Va_dc")
    else:
        print("    No DC voltage angles (Va_dc) present, skipping check for Va_dc")

    print("    Voltage angles bounds check: OK")


def validate_admittance_calculations(
    generated_data: Dict[str, pd.DataFrame],
) -> None:
    """Test that branch admittances in branch_data match calculated values from branch parameters (vectorized).

    Validates the admittance matrix equations for both AC lines and transformers:
        Yff = (y_series + y_sh_f) / t2  (where t2 = |tap|^2)
        Yft = -y_series / tap.conjugate()
        Ytf = -y_series / tap
        Ytt = y_series + y_sh_t

    For AC lines, tap = 1.0 and shift = 0.0, so t2 = 1.0.
    For transformers, tap and shift are non-trivial values.

    where y_series = 1/(r + jx) and y_sh_f, y_sh_t are shunt admittances.
    Compares calculated admittances with stored values in branch_data.
    """
    branch_data = generated_data["branch_data"]

    # Validate all active branches (both AC lines and transformers)
    active_branches = branch_data[branch_data["br_status"] == 1].copy()

    if active_branches.empty:
        print("    Admittance calculations: no active branches to validate")
        return

    print(
        f"    Admittance calculations: validating {len(active_branches)} active branches",
    )

    tolerance = 1e-8

    # Vectorized extraction of branch parameters (convert shift from degrees to radians)
    r = active_branches["r"].to_numpy()
    x = active_branches["x"].to_numpy()
    b = active_branches["b"].to_numpy()
    tap_mag = active_branches["tap"].to_numpy()
    shift_deg = active_branches["shift"].to_numpy()
    shift_rad = np.deg2rad(shift_deg)

    # Skip branches with zero impedance
    valid_mask = np.abs(r + 1j * x) >= 1e-10

    if not valid_mask.any():
        print("    Admittance calculations: OK")
        return

    # Filter to valid branches
    active_branches_valid = active_branches[valid_mask].copy()
    r = r[valid_mask]
    x = x[valid_mask]
    b = b[valid_mask]
    tap_mag = tap_mag[valid_mask]
    shift_rad = shift_rad[valid_mask]

    # Calculate expected admittances using fully vectorized function
    Yff_expected, Yft_expected, Ytf_expected, Ytt_expected = compute_branch_admittances(
        r=r,
        x=x,
        b=b,
        tap_mag=tap_mag,
        shift=shift_rad,
    )

    # Extract stored admittances from branch_data
    Yff_stored = (
        active_branches_valid["Yff_r"].to_numpy()
        + 1j * active_branches_valid["Yff_i"].to_numpy()
    )
    Yft_stored = (
        active_branches_valid["Yft_r"].to_numpy()
        + 1j * active_branches_valid["Yft_i"].to_numpy()
    )
    Ytf_stored = (
        active_branches_valid["Ytf_r"].to_numpy()
        + 1j * active_branches_valid["Ytf_i"].to_numpy()
    )
    Ytt_stored = (
        active_branches_valid["Ytt_r"].to_numpy()
        + 1j * active_branches_valid["Ytt_i"].to_numpy()
    )

    # Vectorized comparison (all differences at once)
    Yff_diff = np.abs(Yff_expected - Yff_stored)
    Yft_diff = np.abs(Yft_expected - Yft_stored)
    Ytf_diff = np.abs(Ytf_expected - Ytf_stored)
    Ytt_diff = np.abs(Ytt_expected - Ytt_stored)

    # Find any mismatches
    has_mismatch = (
        (Yff_diff >= tolerance)
        | (Yft_diff >= tolerance)
        | (Ytf_diff >= tolerance)
        | (Ytt_diff >= tolerance)
    )

    if has_mismatch.any():
        # Get first mismatch for error message
        mismatch_idx = np.where(has_mismatch)[0][0]
        scenario = int(active_branches_valid.iloc[mismatch_idx]["scenario"])
        fb = int(active_branches_valid.iloc[mismatch_idx]["from_bus"])
        tb = int(active_branches_valid.iloc[mismatch_idx]["to_bus"])

        error_details = []
        if Yff_diff[mismatch_idx] >= tolerance:
            error_details.append(
                f"Yff: expected {Yff_expected[mismatch_idx]}, got {Yff_stored[mismatch_idx]}, diff={Yff_diff[mismatch_idx]:.2e}",
            )
        if Yft_diff[mismatch_idx] >= tolerance:
            error_details.append(
                f"Yft: expected {Yft_expected[mismatch_idx]}, got {Yft_stored[mismatch_idx]}, diff={Yft_diff[mismatch_idx]:.2e}",
            )
        if Ytf_diff[mismatch_idx] >= tolerance:
            error_details.append(
                f"Ytf: expected {Ytf_expected[mismatch_idx]}, got {Ytf_stored[mismatch_idx]}, diff={Ytf_diff[mismatch_idx]:.2e}",
            )
        if Ytt_diff[mismatch_idx] >= tolerance:
            error_details.append(
                f"Ytt: expected {Ytt_expected[mismatch_idx]}, got {Ytt_stored[mismatch_idx]}, diff={Ytt_diff[mismatch_idx]:.2e}",
            )

        error_msg = (
            f"Scenario {scenario}, Branch {fb}->{tb}: {', '.join(error_details)}"
        )
        total_mismatches = has_mismatch.sum()
        if total_mismatches > 1:
            error_msg += f" ({total_mismatches} total mismatches)"

        raise AssertionError(f"Admittance calculations failed: {error_msg}")

    print("    Admittance calculations: OK")


def validate_computed_vs_stored_power_flows(
    generated_data: Dict[str, pd.DataFrame],
    sn_mva: float,
) -> None:
    """Test that computed power flows match stored power flows."""

    print(
        f"    Validate computed vs stored power flows: validating {len(generated_data['branch_data'])} branches across {len(generated_data['branch_data']['scenario'].unique())} scenarios",
    )

    pf, qf, pt, qt = compute_branch_powers_vectorized(
        generated_data["branch_data"],
        generated_data["bus_data"],
        dc=False,
        sn_mva=sn_mva,
    )
    computed_flows = pd.DataFrame(
        {
            "pf": pf,
            "qf": qf,
            "pt": pt,
            "qt": qt,
            "scenario": generated_data["branch_data"]["scenario"],
            "from_bus": generated_data["branch_data"]["from_bus"],
            "to_bus": generated_data["branch_data"]["to_bus"],
        },
        index=generated_data["branch_data"].index,
    )

    flows_data = generated_data["branch_data"][
        ["pf", "qf", "pt", "qt", "scenario", "from_bus", "to_bus"]
    ]
    mismatch = ~np.isclose(computed_flows, flows_data, atol=1e-2, rtol=1e-3)
    # TODO investigate why atol has to be so large, especially for pf delta
    if mismatch.any():
        raise AssertionError(
            f"Computed power flows do not match stored power flows, stored: \n{flows_data[mismatch]}, computed: \n{computed_flows[mismatch]}",
        )

    print("    Computed vs stored power flows: OK")


def validate_tap_not_zero(generated_data: Dict[str, pd.DataFrame]) -> None:
    """Test that transformer tap ratio is not zero for active branches."""
    branch_data = generated_data["branch_data"]
    active_branches = branch_data[branch_data["br_status"] == 1]

    print(
        f"    Tap not zero: validating {len(active_branches)} active branches",
    )
    if not active_branches.empty:
        zero_tap_branches = active_branches[active_branches["tap"] == 0]
        assert len(zero_tap_branches) == 0, (
            f"Active branches should not have zero tap ratio. "
            f"Found {len(zero_tap_branches)} branches with tap=0"
        )

    print("    Tap not zero: OK")


def validate_branch_loading_opf_mode(generated_data: Dict[str, pd.DataFrame]) -> None:
    """Test branch loading limits in OPF mode, compute loading statistics in PF mode."""
    bus_data = generated_data["bus_data"]
    branch_data = generated_data["branch_data"]

    scenarios = bus_data["scenario"].unique()

    # Filter to active, rated branches
    rated_branches = branch_data[
        (branch_data["br_status"] == 1) & (branch_data["rate_a"] > 0)
    ].copy()

    mode_label = "opf" if generated_data["mode"] == "opf" else "pf"
    print(
        f"    Branch loading limits ({mode_label} mode): validating {len(rated_branches)} rated branches across {len(scenarios)} scenarios",
    )

    # Vectorized computation of loading
    # Compute apparent power: S = sqrt(P^2 + Q^2)
    s_from = np.sqrt(
        rated_branches["pf"].to_numpy() ** 2 + rated_branches["qf"].to_numpy() ** 2,
    )
    s_to = np.sqrt(
        rated_branches["pt"].to_numpy() ** 2 + rated_branches["qt"].to_numpy() ** 2,
    )
    rate_a = rated_branches["rate_a"].to_numpy()

    # Loading = max(S_from, S_to) / rate_a
    loading = np.maximum(s_from, s_to) / rate_a

    # Identify binding and overloaded branches
    binding_mask = loading >= 0.99
    overload_mask = loading > 1.01

    binding_loadings = loading[binding_mask]
    n_binding = len(binding_loadings)
    n_overloads = overload_mask.sum()

    # In OPF mode, assert no overloads
    if generated_data["mode"] == "opf":
        if n_overloads > 0:
            overloaded_idx = np.where(overload_mask)[0]
            overload_info = rated_branches.iloc[overloaded_idx[0]]
            raise AssertionError(
                f"Scenario {int(overload_info['scenario'])}, "
                f"Branch {int(overload_info['from_bus'])}->{int(overload_info['to_bus'])}: "
                f"Loading {loading[overloaded_idx[0]]:.3f} exceeds 1.01 in OPF mode",
            )

    print(
        f"    Binding loading constraints (>= 0.99): {n_binding} branches",
    )
    if generated_data["mode"] == "pf":
        print(f"    Overloaded branches (> 1.0): {n_overloads} branches")
        print("    Branch loading limits (PF mode): statistics computed")
    else:
        print("    Branch loading limits (OPF mode): OK")


def validate_deactivated_generators_zero_output(
    generated_data: Dict[str, pd.DataFrame],
) -> None:
    """Test that deactivated generators have zero output."""
    gen_data = generated_data["gen_data"]
    deactivated_gens = gen_data[gen_data["in_service"] == 0]

    print(
        f"    Deactivated generators zero output: validating {len(deactivated_gens)} deactivated generators",
    )
    if not deactivated_gens.empty:
        assert (deactivated_gens["p_mw"] == 0).all(), (
            "Deactivated generators should have zero p_mw"
        )
        assert (deactivated_gens["q_mvar"] == 0).all(), (
            "Deactivated generators should have zero q_mvar"
        )

        if "p_mw_dc" in deactivated_gens.columns:
            # zero or nan (if no solution was found)
            assert (
                (deactivated_gens["p_mw_dc"] == 0)
                | (deactivated_gens["p_mw_dc"].isna())
            ).all(), "Deactivated generators should have zero p_mw_dc or be NaN"

    print("    Deactivated generators zero output: OK")


def validate_generator_limits(generated_data: Dict[str, pd.DataFrame]) -> None:
    """Test that generator outputs respect their limits."""
    gen_data = generated_data["gen_data"]
    gen_data = gen_data[gen_data["in_service"] == 1]
    # keep only the ones with limits for p_mw
    filtered_gens = gen_data[
        gen_data["max_p_mw"].notna() & gen_data["min_p_mw"].notna()
    ]

    if generated_data["mode"] == "pf":
        filtered_gens = filtered_gens[filtered_gens["is_slack_gen"] == 0]

    print(
        f"    Generator limits: validating {len(filtered_gens)} active generators (mode: {generated_data['mode']})",
    )

    # Count binding P limits
    binding_p_min = 0
    binding_p_max = 0
    if not filtered_gens.empty:
        p_within_limits = (
            filtered_gens["p_mw"] >= filtered_gens["min_p_mw"] - 1e-2
        ) & (filtered_gens["p_mw"] <= filtered_gens["max_p_mw"] + 1e-2)

        # Check for binding minimum limits
        p_at_min = (filtered_gens["p_mw"] <= filtered_gens["min_p_mw"] + 1e-2) & (
            filtered_gens["p_mw"] >= filtered_gens["min_p_mw"] - 1e-2
        )
        binding_p_min = p_at_min.sum()

        # Check for binding maximum limits
        p_at_max = (filtered_gens["p_mw"] <= filtered_gens["max_p_mw"] + 1e-2) & (
            filtered_gens["p_mw"] >= filtered_gens["max_p_mw"] - 1e-2
        )
        binding_p_max = p_at_max.sum()

        assert p_within_limits.all(), (
            f"Generator active power should be within limits, current: \n{filtered_gens.loc[~p_within_limits, ['bus', 'p_mw']]}, \nmax: \n{filtered_gens.loc[~p_within_limits, ['bus', 'max_p_mw']]}"
        )

    # Count binding Q limits (only in OPF mode)
    binding_q_min = 0
    binding_q_max = 0
    if generated_data["mode"] == "opf":
        filtered_gens_q = filtered_gens[
            filtered_gens["max_q_mvar"].notna() & filtered_gens["min_q_mvar"].notna()
        ]
        q_within_limits = (
            filtered_gens_q["q_mvar"] >= filtered_gens_q["min_q_mvar"] - 1e-2
        ) & (filtered_gens_q["q_mvar"] <= filtered_gens_q["max_q_mvar"] + 1e-2)

        # Check for binding minimum limits
        q_at_min = (
            filtered_gens_q["q_mvar"] <= filtered_gens_q["min_q_mvar"] + 1e-2
        ) & (filtered_gens_q["q_mvar"] >= filtered_gens_q["min_q_mvar"] - 1e-2)
        binding_q_min = q_at_min.sum()

        # Check for binding maximum limits
        q_at_max = (
            filtered_gens_q["q_mvar"] <= filtered_gens_q["max_q_mvar"] + 1e-2
        ) & (filtered_gens_q["q_mvar"] >= filtered_gens_q["max_q_mvar"] - 1e-2)
        binding_q_max = q_at_max.sum()

        assert q_within_limits.all(), (
            f"Generator reactive power should be within limits, expected: {filtered_gens_q.loc[~q_within_limits, ['bus', 'q_mvar']]}, actual: {filtered_gens_q.loc[~q_within_limits, ['bus', 'q_mvar']]}, max: {filtered_gens_q.loc[~q_within_limits, ['bus', 'max_q_mvar']]}"
        )

    print(
        f"    Binding P limits: {binding_p_min} at minimum, {binding_p_max} at maximum",
    )
    if generated_data["mode"] == "opf":
        print(
            f"    Binding Q limits: {binding_q_min} at minimum, {binding_q_max} at maximum",
        )
    print("    Generator limits: OK")


def validate_voltage_magnitude_limits_opf_mode(
    generated_data: Dict[str, pd.DataFrame],
) -> None:
    """Test voltage magnitude limits in OPF mode."""
    if generated_data["mode"] != "opf":
        print("    Voltage magnitude limits: skipped (not in OPF mode)")
        return

    bus_data = generated_data["bus_data"]
    print(
        f"    Voltage magnitude limits (OPF mode): validating {len(bus_data)} bus voltage entries",
    )
    vm_within_limits = (bus_data["Vm"] >= bus_data["min_vm_pu"] - 1e-6) & (
        bus_data["Vm"] <= bus_data["max_vm_pu"] + 1e-6
    )
    assert vm_within_limits.all(), "Voltage magnitudes should be within limits"
    print("    Voltage magnitude limits (OPF mode): OK")


def validate_branch_angle_difference_opf_mode(
    generated_data: Dict[str, pd.DataFrame],
) -> None:
    """Validate branch angle difference limits in OPF mode (vectorized).

    For each active branch, the difference in bus voltage angles must respect
    the branch angle limits [angmin, angmax].
    """
    if generated_data["mode"] != "opf":
        print("    Branch angle difference limits: skipped (not in OPF mode)")
        return

    bus_data = generated_data["bus_data"]
    branch_data = generated_data["branch_data"]

    # Filter to active branches only
    active_branches = branch_data[branch_data["br_status"] == 1].copy()

    if active_branches.empty:
        print("    Branch angle difference limits (OPF mode): OK")
        return

    scenarios = active_branches["scenario"].unique()
    print(
        f"    Branch angle difference limits (OPF mode): validating across {len(scenarios)} scenarios",
    )

    # Merge branch data with bus voltage angles
    bus_angles = bus_data[["scenario", "bus", "Va"]].copy()

    # Merge for from_bus
    merged_from = active_branches.merge(
        bus_angles.rename(columns={"bus": "from_bus", "Va": "Va_from"}),
        on=["scenario", "from_bus"],
        how="left",
    )

    # Merge for to_bus
    merged = merged_from.merge(
        bus_angles.rename(columns={"bus": "to_bus", "Va": "Va_to"}),
        on=["scenario", "to_bus"],
        how="left",
    )

    # Calculate angle differences and normalize to [-180, 180]
    delta = (merged["Va_from"] - merged["Va_to"]).to_numpy()
    delta = delta % 360.0  # Normalize to [0, 360)
    delta = np.where(delta > 180.0, delta - 360.0, delta)  # Convert to [-180, 180)

    # Check limits with tolerance
    tolerance = 1e-6
    min_violations = delta < (merged["ang_min"].to_numpy() - tolerance)
    max_violations = delta > (merged["ang_max"].to_numpy() + tolerance)

    violations = min_violations | max_violations

    if violations.any():
        # Report first violation
        viol_idx = np.where(violations)[0][0]
        scenario = int(merged.iloc[viol_idx]["scenario"])
        fb = int(merged.iloc[viol_idx]["from_bus"])
        tb = int(merged.iloc[viol_idx]["to_bus"])
        delta_val = delta[viol_idx]
        angmin = merged.iloc[viol_idx]["ang_min"]
        angmax = merged.iloc[viol_idx]["ang_max"]

        if delta_val < angmin - tolerance:
            raise AssertionError(
                f"Scenario {scenario}, Branch {fb}->{tb}: angle diff {delta_val:.3f} < angmin {angmin}",
            )
        else:
            raise AssertionError(
                f"Scenario {scenario}, Branch {fb}->{tb}: angle diff {delta_val:.3f} > angmax {angmax}",
            )

    print("    Branch angle difference limits (OPF mode): OK")


def validate_bus_generation_consistency(
    generated_data: Dict[str, pd.DataFrame],
) -> None:
    """Test that Pg in bus data equals sum of generators at each bus (vectorized)."""
    bus_data = generated_data["bus_data"]
    gen_data = generated_data["gen_data"]

    scenarios = bus_data["scenario"].unique()
    print(
        f"    Bus generation consistency: validating {len(bus_data)} bus entries across {len(scenarios)} scenarios",
    )

    # Aggregate generator outputs by (scenario, bus)
    gen_sum = (
        gen_data.groupby(["scenario", "bus"], as_index=False)
        .agg({"p_mw": "sum", "q_mvar": "sum"})
        .rename(columns={"p_mw": "pg_gen_sum", "q_mvar": "qg_gen_sum"})
    )

    # Prepare bus data with (scenario, bus) as key
    bus_keyed = bus_data[["scenario", "bus", "Pg", "Qg"]].copy()
    bus_keyed["scenario"] = bus_keyed["scenario"].astype(int)
    bus_keyed["bus"] = bus_keyed["bus"].astype(int)

    # Merge bus data with generator sums
    comparison = bus_keyed.merge(gen_sum, on=["scenario", "bus"], how="left").fillna(
        {"pg_gen_sum": 0.0, "qg_gen_sum": 0.0},
    )

    # Vectorized comparison
    tolerance = 1e-6
    pg_diff = np.abs(comparison["Pg"] - comparison["pg_gen_sum"])
    qg_diff = np.abs(comparison["Qg"] - comparison["qg_gen_sum"])

    pg_mismatches = comparison[pg_diff >= tolerance]
    qg_mismatches = comparison[qg_diff >= tolerance]

    if len(pg_mismatches) > 0:
        raise AssertionError(f"Pg mismatches: {pg_mismatches}")
    if len(qg_mismatches) > 0:
        raise AssertionError(f"Qg mismatches: {qg_mismatches}")

    print("    Bus generation consistency: OK")


def validate_bus_generation_consistency_dc(
    generated_data: Dict[str, pd.DataFrame],
) -> None:
    """Test that Pg_dc in bus data equals sum of DC generator outputs at each bus (vectorized).

    Skips if required DC columns are not present.
    """
    bus_data = generated_data["bus_data"]
    gen_data = generated_data["gen_data"]

    # Require presence of DC columns
    if ("Pg_dc" not in bus_data.columns) or ("p_mw_dc" not in gen_data.columns):
        print("    Bus generation DC consistency: skipped (no DC columns)")
        return

    scenarios = bus_data["scenario"].unique()
    print(
        f"    Bus generation DC consistency: validating {len(bus_data)} bus entries across {len(scenarios)} scenarios",
    )

    # Aggregate DC generator outputs by (scenario, bus)
    gen_sum_dc = (
        gen_data.groupby(["scenario", "bus"], as_index=False)
        .agg({"p_mw_dc": "sum"})
        .rename(columns={"p_mw_dc": "pg_dc_gen_sum"})
    )

    # Prepare bus data with (scenario, bus) as key
    bus_keyed = bus_data[["scenario", "bus", "Pg_dc"]].copy()
    bus_keyed["scenario"] = bus_keyed["scenario"].astype(int)
    bus_keyed["bus"] = bus_keyed["bus"].astype(int)

    # Merge bus data with generator sums
    comparison = bus_keyed.merge(gen_sum_dc, on=["scenario", "bus"], how="left").fillna(
        {"pg_dc_gen_sum": 0.0},
    )

    # Filter out NaN Pg_dc rows (e.g., OPF mode without DC)
    comparison = comparison[comparison["Pg_dc"].notna()]

    if len(comparison) == 0:
        print("    Bus generation DC consistency: OK")
        return

    # Vectorized comparison
    tolerance = 1e-6
    pg_dc_diff = np.abs(comparison["Pg_dc"] - comparison["pg_dc_gen_sum"])

    pg_dc_mismatches = comparison[pg_dc_diff >= tolerance]

    if len(pg_dc_mismatches) > 0:
        raise AssertionError(f"Pg_dc mismatches: {pg_dc_mismatches}")

    print("    Bus generation DC consistency: OK")


def validate_dc_columns_consistency(generated_data: Dict[str, pd.DataFrame]) -> None:
    """Test that if any DC column has NaN for a scenario, all DC columns are NaN for that scenario (vectorized).

    This ensures that when DC power flow doesn't converge, all DC data is consistently NaN,
    not just some columns.
    """
    bus_data = generated_data["bus_data"]
    branch_data = generated_data["branch_data"]
    gen_data = generated_data["gen_data"]
    runtime_data = generated_data["runtime_data"]

    # check if there is dc data:
    if "Va_dc" not in bus_data.columns:
        print("    DC columns consistency: skipped (no Va_dc column)")
        return

    # compute scenarios that have nan for each DC column
    scenarios_with_nan = {}
    for col in DC_BUS_COLUMNS:
        scenarios_with_nan[col] = set(
            bus_data[bus_data[col].isna()]["scenario"].unique(),
        )
    for col in DC_BRANCH_COLUMNS:
        scenarios_with_nan[col] = set(
            branch_data[branch_data[col].isna()]["scenario"].unique(),
        )
    for col in DC_GEN_COLUMNS:
        scenarios_with_nan[col] = set(
            gen_data[gen_data[col].isna()]["scenario"].unique(),
        )
    if runtime_data is not None:
        for col in DC_RUNTIME_COLUMNS:
            scenarios_with_nan[col] = set(
                runtime_data[runtime_data[col].isna()]["scenario"].unique(),
            )

    # check all values of scenarios_with_nan are the same
    key_0 = list(scenarios_with_nan.keys())[0]
    for col in scenarios_with_nan:
        if scenarios_with_nan[col] != scenarios_with_nan[key_0]:
            # find the difference
            difference = (
                scenarios_with_nan[col]
                .difference(scenarios_with_nan[key_0])
                .union(scenarios_with_nan[key_0].difference(scenarios_with_nan[col]))
            )
            raise AssertionError(
                f"DC columns consistency: scenarios with nan for {col} are not the same, difference: {sorted([int(i) for i in difference])}",
            )

    print(
        "number of scenarios with nan: ",
        len(
            scenarios_with_nan[key_0].union(
                scenarios_with_nan[key_0].difference(scenarios_with_nan[col]),
            ),
        ),
    )
    print("    DC columns consistency: OK")


def validate_non_slack_pg_consistency(generated_data: Dict[str, pd.DataFrame]) -> None:
    """Test that Pg and Pg_dc are the same for all buses except slack in PF mode.

    In power flow mode, the dispatch (generator outputs) should not change between
    AC and DC power flow solutions. The slack generator may differ because it adjusts
    to balance the system, but all other buses should have Pg == Pg_dc.

    Skips if required DC columns are not present or if not in PF mode.
    """
    if generated_data["mode"] != "pf":
        print("    Slack Pg consistency: skipped (not in PF mode)")
        return

    bus_data = generated_data["bus_data"]

    # Require presence of DC columns
    if "Pg_dc" not in bus_data.columns:
        print("    Slack Pg consistency: skipped (no Pg_dc column)")
        return

    scenarios = bus_data["scenario"].unique()

    # Filter to non-slack buses
    non_slack_buses = bus_data[bus_data["REF"] == 0].copy()
    print(
        f"    Non-slack Pg consistency: validating {len(non_slack_buses)} non-slack bus entries across {len(scenarios)} scenarios",
    )

    # Vectorized comparison: Pg vs Pg_dc for non-slack buses

    pg = non_slack_buses["Pg"].to_numpy()
    pg_dc = non_slack_buses["Pg_dc"].to_numpy()

    # Create mask for valid comparisons (exclude NaN Pg_dc)
    valid_mask = ~np.isnan(pg_dc)

    tolerance = 1e-6
    mismatch_mask = np.abs(pg - pg_dc) >= tolerance
    violations_mask = valid_mask & mismatch_mask

    if violations_mask.any():
        # Get first violation for error message
        violation_idx = np.where(violations_mask)[0][0]
        violation_row = non_slack_buses.iloc[violation_idx]
        raise AssertionError(
            f"Scenario {int(violation_row['scenario'])}, Bus {int(violation_row['bus'])}: "
            f"Pg and Pg_dc mismatch (dispatch should not change in PF mode), "
            f"Pg: {violation_row['Pg']}, Pg_dc: {violation_row['Pg_dc']}",
        )

    print("    Non-slack Pg consistency: OK")


def validate_power_balance_equations(
    generated_data: Dict[str, pd.DataFrame],
    sn_mva: float,
) -> None:
    """Test power balance equations (Kirchhoff's Current Law)."""
    bus_data = generated_data["bus_data"]
    branch_data = generated_data["branch_data"]

    scenarios = bus_data["scenario"].unique()
    print(
        f"    Power balance equations (Kirchhoff's Law): validating {len(bus_data)} bus entries across {len(scenarios)} scenarios",
    )

    power_balance_ac = compute_bus_balance(
        bus_data,
        branch_data,
        branch_data[["pf", "qf", "pt", "qt"]],
        False,
        sn_mva=sn_mva,
    )
    not_close_zero = ~np.isclose(0.0, power_balance_ac["P_mis_ac"], atol=1e-3)
    # TODO investigate why atol has to be so large
    if not_close_zero.any():
        raise AssertionError(
            f"Power balance equations (Kirchhoff's Law) do not hold, mismatches: {power_balance_ac[not_close_zero]}",
        )

    print("    Power balance equations (Kirchhoff's Law): OK")


def validate_scenario_indexing_consistency(
    generated_data: Dict[str, pd.DataFrame],
) -> None:
    """Test that scenario indices are consistent across all data files."""
    bus_data = generated_data["bus_data"]
    branch_data = generated_data["branch_data"]
    gen_data = generated_data["gen_data"]
    y_bus_data = generated_data["y_bus_data"]

    bus_scenarios = set(bus_data["scenario"].unique())
    branch_scenarios = set(branch_data["scenario"].unique())
    gen_scenarios = set(gen_data["scenario"].unique())
    ybus_scenarios = set(y_bus_data["scenario"].unique())

    print(
        f"    Scenario indexing consistency: validating {len(bus_scenarios)} scenarios across 4 data files",
    )

    assert bus_scenarios == branch_scenarios == gen_scenarios == ybus_scenarios, (
        "All data files should contain the same set of scenario indices"
    )

    print("    Scenario indexing consistency: OK")


def validate_bus_indexing_consistency(generated_data: Dict[str, pd.DataFrame]) -> None:
    """Test that bus indices are consistent across data files."""
    bus_data = generated_data["bus_data"]
    branch_data = generated_data["branch_data"]
    gen_data = generated_data["gen_data"]

    bus_indices = set(bus_data["bus"].unique())
    branch_bus_indices = set(branch_data["from_bus"].unique()) | set(
        branch_data["to_bus"].unique(),
    )
    gen_bus_indices = set(gen_data["bus"].unique())

    print(
        f"    Bus indexing consistency: validating {len(bus_indices)} buses across 3 data files",
    )

    assert gen_bus_indices.issubset(bus_indices), (
        "All generator buses should exist in bus data"
    )
    assert branch_bus_indices.issubset(bus_indices), (
        "All branch endpoint buses should exist in bus data"
    )

    print("    Bus indexing consistency: OK")


def _require_columns(df: pd.DataFrame, name: str, required: Iterable[str]) -> None:
    required = copy.deepcopy(required)  # deepcopy to avoid modifying the original list
    if "load_scenario_idx" in required:
        required.remove("load_scenario_idx")
    missing = set(required) - set(df.columns)
    assert not missing, f"{name}: missing required columns {sorted(missing)}"


def _check_no_nan(df: pd.DataFrame, name: str, required: Iterable[str]) -> None:
    required = copy.deepcopy(required)  # deepcopy to avoid modifying the original list
    if "load_scenario_idx" in required:
        required.remove("load_scenario_idx")
    if df[required].isna().any().any():
        for col in required:
            assert not df[col].isna().any(), (
                f"{name}: column '{col}' contains NaN values"
            )


def validate_data_completeness(generated_data: Dict[str, pd.DataFrame]) -> None:
    """Test that all required columns are present and contain no NaN values."""
    bus_data = generated_data["bus_data"]
    branch_data = generated_data["branch_data"]
    gen_data = generated_data["gen_data"]
    y_bus_data = generated_data["y_bus_data"]
    runtime_data = generated_data["runtime_data"]

    total_entries = len(bus_data) + len(branch_data) + len(gen_data) + len(y_bus_data)
    print(
        f"    Data completeness: validating {total_entries} total entries across 4 data files",
    )

    # 1) Ensure 'scenario' column exists everywhere
    for name, df in [
        ("Bus data", bus_data),
        ("Branch data", branch_data),
        ("Generator data", gen_data),
        ("Y-bus data", y_bus_data),
    ]:
        assert "scenario" in df.columns, f"{name} should have scenario column"

    if runtime_data is not None:
        assert "scenario" in runtime_data.columns, (
            "Runtime data should have scenario column"
        )

    dc = True if "Va_dc" in bus_data.columns else False

    # 2) Check required columns exist and contain no NaN values
    _require_columns(
        bus_data,
        "Bus data",
        BUS_COLUMNS + DC_BUS_COLUMNS if dc else BUS_COLUMNS,
    )
    _require_columns(
        branch_data,
        "Branch data",
        BRANCH_COLUMNS + DC_BRANCH_COLUMNS if dc else BRANCH_COLUMNS,
    )
    _require_columns(
        gen_data,
        "Generator data",
        GEN_COLUMNS + DC_GEN_COLUMNS if dc else GEN_COLUMNS,
    )
    _require_columns(y_bus_data, "Y-bus data", YBUS_COLUMNS)

    _check_no_nan(bus_data, "Bus data", BUS_COLUMNS)
    _check_no_nan(branch_data, "Branch data", BRANCH_COLUMNS)
    _check_no_nan(gen_data, "Generator data", GEN_COLUMNS)
    _check_no_nan(y_bus_data, "Y-bus data", YBUS_COLUMNS)
    if runtime_data is not None:
        _check_no_nan(runtime_data, "Runtime data", RUNTIME_COLUMNS)

    # 3) Non-emptiness
    assert len(bus_data) > 0, "Bus data should not be empty"
    assert len(branch_data) > 0, "Branch data should not be empty"
    assert len(gen_data) > 0, "Generator data should not be empty"
    assert len(y_bus_data) > 0, "Y-bus data should not be empty"

    print("    Data completeness: OK (all required columns present and NaN-free)")


if __name__ == "__main__":
    # debug with data_out/case24_ieee_rts/raw
    file_paths = {
        "bus_data": "data_out/case24_ieee_rts/raw/bus_data.parquet",
        "branch_data": "data_out/case24_ieee_rts/raw/branch_data.parquet",
        "gen_data": "data_out/case24_ieee_rts/raw/gen_data.parquet",
        "y_bus_data": "data_out/case24_ieee_rts/raw/y_bus_data.parquet",
        "runtime_data": "data_out/case24_ieee_rts/raw/runtime_data.parquet",
    }
    validate_generated_data(file_paths, "pf", 100.0, n_scenarios=10)
    print("Validation completed successfully!")
