"""
Data validation functions for power flow data integrity and consistency checks.

This module contains all validation functions extracted from test_data_validation.py
to provide comprehensive validation of generated power flow data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Iterable
from gridfm_datakit.utils.column_names import (
    BUS_COLUMNS,
    BRANCH_COLUMNS,
    GEN_COLUMNS,
    YBUS_COLUMNS,
)


def validate_generated_data(
    file_paths: Dict[str, str],
    mode: str,
    n_scenarios: int = 0,
) -> bool:
    """Run all validation tests on the generated data.

    Args:
        file_paths: Dictionary containing paths to data files (bus_data, branch_data, gen_data, y_bus_data).
        mode: Operating mode ("opf" or "pf").
        n_scenarios: Number of scenarios to sample for validation (0 for all scenarios).

    Returns:
        True if all validations pass.

    Raises:
        AssertionError: If any validation fails.
    """
    # Load the generated data
    bus_data = pd.read_parquet(file_paths["bus_data"], engine="fastparquet")
    branch_data = pd.read_parquet(file_paths["branch_data"], engine="fastparquet")
    gen_data = pd.read_parquet(file_paths["gen_data"], engine="fastparquet")
    y_bus_data = pd.read_parquet(file_paths["y_bus_data"], engine="fastparquet")

    generated_data = {
        "bus_data": bus_data,
        "branch_data": branch_data,
        "gen_data": gen_data,
        "y_bus_data": y_bus_data,
        "mode": mode,
        "file_paths": file_paths,
    }

    try:
        validate_scenario_indexing_consistency(generated_data)
    except Exception as e:
        raise AssertionError(f"Scenario indexing consistency validation failed: {e}")

    # Run Data Integrity Tests
    try:
        validate_bus_indexing_consistency(generated_data)
    except Exception as e:
        raise AssertionError(f"Bus indexing consistency validation failed: {e}")

    try:
        validate_data_completeness(generated_data)
    except Exception as e:
        raise AssertionError(f"Data completeness validation failed: {e}")

    # Sample scenarios if n_scenarios is provided to avoid too long validation times
    if n_scenarios > 0:
        max_scenarios = len(generated_data["bus_data"]["scenario"].unique())
        sampled_scenarios = np.random.choice(
            generated_data["bus_data"]["scenario"].unique(),
            size=min(n_scenarios, max_scenarios),
            replace=False,
        )
        generated_data["bus_data"] = generated_data["bus_data"][
            generated_data["bus_data"]["scenario"].isin(sampled_scenarios)
        ]
        generated_data["branch_data"] = generated_data["branch_data"][
            generated_data["branch_data"]["scenario"].isin(sampled_scenarios)
        ]
        generated_data["gen_data"] = generated_data["gen_data"][
            generated_data["gen_data"]["scenario"].isin(sampled_scenarios)
        ]
        generated_data["y_bus_data"] = generated_data["y_bus_data"][
            generated_data["y_bus_data"]["scenario"].isin(sampled_scenarios)
        ]
        print(
            f"Sampled {len(sampled_scenarios)} scenarios for validation out of {max_scenarios}",
        )

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
        validate_computed_vs_stored_power_flows(generated_data)
    except Exception as e:
        raise AssertionError(f"Computed vs stored power flows validation failed: {e}")

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

    try:
        validate_power_balance_equations(generated_data)
    except Exception as e:
        raise AssertionError(f"Power balance equations validation failed: {e}")

    return True


def validate_ybus_diagonal_consistency(generated_data: Dict[str, pd.DataFrame]) -> None:
    """Test Y-bus diagonal consistency with bus and branch data."""
    bus_data = generated_data["bus_data"]
    branch_data = generated_data["branch_data"]
    y_bus_data = generated_data["y_bus_data"]

    scenarios = bus_data["scenario"].unique()
    total_buses = len(bus_data)
    print(
        f"    Y-bus diagonal consistency: validating {total_buses} bus entries across {len(scenarios)} scenarios",
    )
    for scenario in scenarios:
        bus_scenario = bus_data[bus_data["scenario"] == scenario]
        branch_scenario = branch_data[branch_data["scenario"] == scenario]
        ybus_scenario = y_bus_data[y_bus_data["scenario"] == scenario]

        for _, bus_row in bus_scenario.iterrows():
            bus_idx = int(bus_row["bus"])
            gs_shunt = bus_row["GS"]
            bs_shunt = bus_row["BS"]

            from_branches = branch_scenario[branch_scenario["from_bus"] == bus_idx]
            yff_sum_g = from_branches["Yff_r"].sum()
            yff_sum_b = from_branches["Yff_i"].sum()

            to_branches = branch_scenario[branch_scenario["to_bus"] == bus_idx]
            ytt_sum_g = to_branches["Ytt_r"].sum()
            ytt_sum_b = to_branches["Ytt_i"].sum()

            expected_g = gs_shunt + yff_sum_g + ytt_sum_g
            expected_b = bs_shunt + yff_sum_b + ytt_sum_b

            diagonal_entry = ybus_scenario[
                (ybus_scenario["index1"] == bus_idx)
                & (ybus_scenario["index2"] == bus_idx)
            ]

            if not diagonal_entry.empty:
                actual_g = diagonal_entry["G"].iloc[0]
                actual_b = diagonal_entry["B"].iloc[0]

                assert abs(expected_g - actual_g) < 1e-6, (
                    f"Scenario {scenario}, Bus {bus_idx}: G mismatch. Expected: {expected_g}, Actual: {actual_g}"
                )
                assert abs(expected_b - actual_b) < 1e-6, (
                    f"Scenario {scenario}, Bus {bus_idx}: B mismatch. Expected: {expected_b}, Actual: {actual_b}"
                )


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


def validate_computed_vs_stored_power_flows(
    generated_data: Dict[str, pd.DataFrame],
) -> None:
    """Test that computed power flows match stored power flows."""
    bus_data = generated_data["bus_data"]
    branch_data = generated_data["branch_data"]

    scenarios = bus_data["scenario"].unique()
    active_branches = branch_data[branch_data["br_status"] == 1]
    print(
        f"    Computed vs stored power flows: validating {len(active_branches)} active branches across {len(scenarios)} scenarios",
    )
    for scenario in scenarios:
        bus_scenario = bus_data[bus_data["scenario"] == scenario]
        branch_scenario = branch_data[branch_data["scenario"] == scenario]
        active_branches = branch_scenario[branch_scenario["br_status"] == 1]

        if active_branches.empty:
            continue

        Vm = bus_scenario.set_index("bus")["Vm"].to_dict()
        Va_deg = bus_scenario.set_index("bus")["Va"].to_dict()
        Va_rad = {bus: np.radians(angle) for bus, angle in Va_deg.items()}

        for _, branch in active_branches.iterrows():
            from_bus = int(branch["from_bus"])
            to_bus = int(branch["to_bus"])

            Vf_mag = Vm[from_bus]
            Vt_mag = Vm[to_bus]
            Vf_ang = Va_rad[from_bus]
            Vt_ang = Va_rad[to_bus]

            Vf_r = Vf_mag * np.cos(Vf_ang)
            Vf_i = Vf_mag * np.sin(Vf_ang)
            Vt_r = Vt_mag * np.cos(Vt_ang)
            Vt_i = Vt_mag * np.sin(Vt_ang)

            Yff_r, Yff_i = branch["Yff_r"], branch["Yff_i"]
            Yft_r, Yft_i = branch["Yft_r"], branch["Yft_i"]

            If_r = Yff_r * Vf_r - Yff_i * Vf_i + Yft_r * Vt_r - Yft_i * Vt_i
            If_i = Yff_r * Vf_i + Yff_i * Vf_r + Yft_r * Vt_i + Yft_i * Vt_r

            Pf_computed = (Vf_r * If_r + Vf_i * If_i) * 100.0  # Convert to MW
            Qf_computed = (Vf_i * If_r - Vf_r * If_i) * 100.0  # Convert to MVAr

            tolerance = 1e-3
            assert abs(Pf_computed - branch["pf"]) < tolerance, (
                f"Scenario {scenario}, Branch {from_bus}->{to_bus}: Pf mismatch: {Pf_computed} != {branch['pf']} | "
                f"Yff_r: {Yff_r}, Yff_i: {Yff_i}, Yft_r: {Yft_r}, Yft_i: {Yft_i}"
            )
            assert abs(Qf_computed - branch["qf"]) < tolerance, (
                f"Scenario {scenario}, Branch {from_bus}->{to_bus}: Qf mismatch: {Qf_computed} != {branch['qf']} | "
                f"Yff_r: {Yff_r}, Yff_i: {Yff_i}, Yft_r: {Yft_r}, Yft_i: {Yft_i}"
            )

    print("    Computed vs stored power flows: OK")


def validate_branch_loading_opf_mode(generated_data: Dict[str, pd.DataFrame]) -> None:
    """Test branch loading limits in OPF mode, compute loading statistics in PF mode."""
    bus_data = generated_data["bus_data"]
    branch_data = generated_data["branch_data"]

    scenarios = bus_data["scenario"].unique()
    rated_branches = branch_data[
        (branch_data["br_status"] == 1) & (branch_data["rate_a"] > 0)
    ]

    mode_label = "opf" if generated_data["mode"] == "opf" else "pf"
    print(
        f"    Branch loading limits ({mode_label} mode): validating {len(rated_branches)} rated branches across {len(scenarios)} scenarios",
    )

    # Track binding constraints and overloads
    binding_loadings = []
    overloaded_branches = []

    for scenario in scenarios:
        branch_scenario = branch_data[branch_data["scenario"] == scenario]
        active_branches = branch_scenario[
            (branch_scenario["br_status"] == 1) & (branch_scenario["rate_a"] > 0)
        ]

        if active_branches.empty:
            continue

        for _, branch in active_branches.iterrows():
            from_bus = int(branch["from_bus"])
            to_bus = int(branch["to_bus"])
            rate_a = branch["rate_a"]

            # Get power flow values from branch data
            pf = branch["pf"]  # Real power from
            qf = branch["qf"]  # Reactive power from
            pt = branch["pt"]  # Real power to
            qt = branch["qt"]  # Reactive power to

            # Compute apparent power squared: S^2 = P^2 + Q^2
            s_from_sq = pf**2 + qf**2
            s_to_sq = pt**2 + qt**2

            # Loading is the ratio of apparent power to rate_a
            # PowerModels constraint: p[f_idx]^2 + q[f_idx]^2 <= rate_a^2
            loading_f = np.sqrt(s_from_sq) / rate_a if rate_a > 0 else 0
            loading_t = np.sqrt(s_to_sq) / rate_a if rate_a > 0 else 0
            loading = max(loading_f, loading_t)

            # Track if loading is binding (within 1% of limit)
            if loading >= 0.99:
                binding_loadings.append(loading)

            # Track overloads (loading > 1.0)
            if loading > 1.01:
                overloaded_branches.append((scenario, from_bus, to_bus, loading))

            # Only assert loading limits in OPF mode
            if generated_data["mode"] == "opf":
                assert loading <= 1.01, (
                    f"Scenario {scenario}, Branch {from_bus}->{to_bus}: "
                    f"Loading {loading:.3f} exceeds 1.01 in OPF mode"
                )

    print(
        f"    Binding loading constraints (>= 0.99): {len(binding_loadings)} branches",
    )
    if generated_data["mode"] == "pf":
        print(f"    Overloaded branches (> 1.0): {len(overloaded_branches)} branches")
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
    """Validate branch angle difference limits in OPF mode.

    For each active branch, the difference in bus voltage angles must respect
    the branch angle limits [angmin, angmax].
    """
    if generated_data["mode"] != "opf":
        print("    Branch angle difference limits: skipped (not in OPF mode)")
        return

    bus_data = generated_data["bus_data"]
    branch_data = generated_data["branch_data"]
    scenarios = bus_data["scenario"].unique()
    print(
        f"    Branch angle difference limits (OPF mode): validating across {len(scenarios)} scenarios",
    )

    for scenario in scenarios:
        bus_scenario = bus_data[bus_data["scenario"] == scenario]
        branch_scenario = branch_data[branch_data["scenario"] == scenario]
        active_branches = branch_scenario[branch_scenario["br_status"] == 1]

        if active_branches.empty:
            continue

        va_deg = bus_scenario.set_index("bus")["Va"].to_dict()

        for _, br in active_branches.iterrows():
            shift = br["shift"]
            if shift != 0.0:
                continue  # skip branches with phase shift
            fb = int(br["from_bus"])
            tb = int(br["to_bus"])
            angmin = br["ang_min"]
            angmax = br["ang_max"]

            if fb not in va_deg or tb not in va_deg:
                continue

            delta = va_deg[fb] - va_deg[tb]

            # Normalize delta into [-180, 180] for robust comparison
            while delta > 180.0:
                delta -= 360.0
            while delta < -180.0:
                delta += 360.0

            assert delta >= angmin - 1e-6, (
                f"Scenario {scenario}, Branch {fb}->{tb}: angle diff {delta:.3f} < angmin {angmin}"
            )
            assert delta <= angmax + 1e-6, (
                f"Scenario {scenario}, Branch {fb}->{tb}: angle diff {delta:.3f} > angmax {angmax}"
            )

    print("    Branch angle difference limits (OPF mode): OK")


def validate_bus_generation_consistency(
    generated_data: Dict[str, pd.DataFrame],
) -> None:
    """Test that Pg in bus data equals sum of generators at each bus."""
    bus_data = generated_data["bus_data"]
    gen_data = generated_data["gen_data"]

    scenarios = bus_data["scenario"].unique()
    print(
        f"    Bus generation consistency: validating {len(bus_data)} bus entries across {len(scenarios)} scenarios",
    )
    for scenario in scenarios:
        bus_scenario = bus_data[bus_data["scenario"] == scenario]
        gen_scenario = gen_data[gen_data["scenario"] == scenario]
        active_gens = gen_scenario[gen_scenario["in_service"] == 1]

        for _, bus_row in bus_scenario.iterrows():
            bus_idx = int(bus_row["bus"])
            pg_bus = bus_row["Pg"]
            qg_bus = bus_row["Qg"]

            bus_gens = active_gens[active_gens["bus"] == bus_idx]
            pg_gen_sum = bus_gens["p_mw"].sum()
            qg_gen_sum = bus_gens["q_mvar"].sum()

            tolerance = 1e-6
            assert abs(pg_bus - pg_gen_sum) < tolerance, (
                f"Scenario {scenario}, Bus {bus_idx}: Pg mismatch",
                f"Pg_bus: {pg_bus}, Pg_gen_sum: {pg_gen_sum}",
            )
            assert abs(qg_bus - qg_gen_sum) < tolerance, (
                f"Scenario {scenario}, Bus {bus_idx}: Qg mismatch",
                f"Qg_bus: {qg_bus}, Qg_gen_sum: {qg_gen_sum}",
            )

    print("    Bus generation consistency: OK")


def validate_power_balance_equations(generated_data: Dict[str, pd.DataFrame]) -> None:
    """Test power balance equations (Kirchhoff's Current Law)."""
    bus_data = generated_data["bus_data"]
    branch_data = generated_data["branch_data"]

    scenarios = bus_data["scenario"].unique()
    print(
        f"    Power balance equations (Kirchhoff's Law): validating {len(bus_data)} bus entries across {len(scenarios)} scenarios",
    )
    for scenario in scenarios:
        bus_scenario = bus_data[bus_data["scenario"] == scenario]
        branch_scenario = branch_data[branch_data["scenario"] == scenario]
        active_branches = branch_scenario[branch_scenario["br_status"] == 1]

        for _, bus_row in bus_scenario.iterrows():
            bus_idx = int(bus_row["bus"])

            pg = bus_row["Pg"]
            pd = bus_row["Pd"]
            qg = bus_row["Qg"]
            qd = bus_row["Qd"]
            gs = bus_row["GS"]
            bs = bus_row["BS"]
            vm = bus_row["Vm"]

            p_shunt = (
                -gs * 100 * vm**2
            )  # MINUS SIGN BECAUSE Gs, shunt conductance (MW DEMANDED at V = 1.0 p.u.) https://matpower.org/docs/ref/matpower6.0/idx_bus.html
            q_shunt = bs * 100 * vm**2

            net_p_injection = pg - pd + p_shunt
            net_q_injection = qg - qd + q_shunt

            from_branches = active_branches[active_branches["from_bus"] == bus_idx]
            p_from_sum = from_branches["pf"].sum()
            q_from_sum = from_branches["qf"].sum()

            to_branches = active_branches[active_branches["to_bus"] == bus_idx]
            p_to_sum = to_branches["pt"].sum()
            q_to_sum = to_branches["qt"].sum()

            total_p_flow = p_from_sum + p_to_sum
            total_q_flow = q_from_sum + q_to_sum

            tolerance = 1e-3 if generated_data["mode"] == "pf" else 1e-2

            p_balance_error = abs(net_p_injection - total_p_flow)
            q_balance_error = abs(net_q_injection - total_q_flow)

            assert p_balance_error < tolerance, (
                f"Scenario {scenario}, Bus {bus_idx}: Active power balance violation, expected: {net_p_injection}, actual: {total_p_flow}"
            )
            assert q_balance_error < tolerance, (
                f"Scenario {scenario}, Bus {bus_idx}: Reactive power balance violation, expected: {net_q_injection}, actual: {total_q_flow}"
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
    missing = set(required) - set(df.columns)
    assert not missing, f"{name}: missing required columns {sorted(missing)}"


def _check_no_nan(df: pd.DataFrame, name: str, required: Iterable[str]) -> None:
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

    # 2) Check required columns exist and contain no NaN values
    _require_columns(bus_data, "Bus data", BUS_COLUMNS)
    _require_columns(branch_data, "Branch data", BRANCH_COLUMNS)
    _require_columns(gen_data, "Generator data", GEN_COLUMNS)
    _require_columns(y_bus_data, "Y-bus data", YBUS_COLUMNS)

    _check_no_nan(bus_data, "Bus data", BUS_COLUMNS)
    _check_no_nan(branch_data, "Branch data", BRANCH_COLUMNS)
    _check_no_nan(gen_data, "Generator data", GEN_COLUMNS)
    _check_no_nan(y_bus_data, "Y-bus data", YBUS_COLUMNS)

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
    }
    validate_generated_data(file_paths, "pf", n_scenarios=10)
    print("Validation completed successfully!")
