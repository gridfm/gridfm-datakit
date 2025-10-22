"""
Data validation functions for power flow data integrity and consistency checks.

This module contains all validation functions extracted from test_data_validation.py
to provide comprehensive validation of generated power flow data.
"""

import pandas as pd
import numpy as np


def validate_generated_data(file_paths, mode, n_scenarios=0):
    """
    Run all validation tests on the generated data.
    Returns True if all validations pass, raises AssertionError if any fail.
    """
    # Load the generated data
    bus_data = pd.read_csv(file_paths["bus_data"])
    branch_data = pd.read_csv(file_paths["branch_data"])
    gen_data = pd.read_csv(file_paths["gen_data"])
    y_bus_data = pd.read_csv(file_paths["y_bus_data"])

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

    # Run Data Integrity Tests
    try:
        validate_bus_indexing_consistency(generated_data)
    except Exception as e:
        raise AssertionError(f"Bus indexing consistency validation failed: {e}")

    try:
        validate_data_completeness(generated_data)
    except Exception as e:
        raise AssertionError(f"Data completeness validation failed: {e}")

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

    if mode == "secure":
        try:
            validate_branch_loading_secure_mode(generated_data)
        except Exception as e:
            raise AssertionError(f"Branch loading secure mode validation failed: {e}")

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

    # Run Secure Mode Constraints
    if mode == "secure":
        try:
            validate_voltage_magnitude_limits_secure_mode(generated_data)
        except Exception as e:
            raise AssertionError(
                f"Voltage magnitude limits secure mode validation failed: {e}",
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


def validate_ybus_diagonal_consistency(generated_data):
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


def validate_deactivated_lines_zero_admittance(generated_data):
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


def validate_computed_vs_stored_power_flows(generated_data):
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


def validate_branch_loading_secure_mode(generated_data):
    """Test branch loading limits in secure mode."""
    if generated_data["mode"] != "secure":
        print("    Branch loading limits: skipped (not in secure mode)")
        return

    bus_data = generated_data["bus_data"]
    branch_data = generated_data["branch_data"]

    scenarios = bus_data["scenario"].unique()
    rated_branches = branch_data[
        (branch_data["br_status"] == 1) & (branch_data["rate_a"] > 0)
    ]
    print(
        f"    Branch loading limits (secure mode): validating {len(rated_branches)} rated branches across {min(2, len(scenarios))} scenarios",
    )
    for scenario in scenarios[:2]:  # Test first 2 scenarios for performance
        bus_scenario = bus_data[bus_data["scenario"] == scenario]
        branch_scenario = branch_data[branch_data["scenario"] == scenario]
        active_branches = branch_scenario[
            (branch_scenario["br_status"] == 1) & (branch_scenario["rate_a"] > 0)
        ]

        if active_branches.empty:
            continue

        Vm = bus_scenario.set_index("bus")["Vm"].to_dict()
        Va_deg = bus_scenario.set_index("bus")["Va"].to_dict()
        Va_rad = {bus: np.radians(angle) for bus, angle in Va_deg.items()}

        for _, branch in active_branches.iterrows():
            from_bus = int(branch["from_bus"])
            to_bus = int(branch["to_bus"])
            rate_a = branch["rate_a"]

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
            Ytf_r, Ytf_i = branch["Ytf_r"], branch["Ytf_i"]
            Ytt_r, Ytt_i = branch["Ytt_r"], branch["Ytt_i"]

            If_r = Yff_r * Vf_r - Yff_i * Vf_i + Yft_r * Vt_r - Yft_i * Vt_i
            If_i = Yff_r * Vf_i + Yff_i * Vf_r + Yft_r * Vt_i + Yft_i * Vt_r
            It_r = Ytf_r * Vf_r - Ytf_i * Vf_i + Ytt_r * Vt_r - Ytt_i * Vt_i
            It_i = Ytf_r * Vf_i + Ytf_i * Vf_r + Ytt_r * Vt_i + Ytt_i * Vt_r

            If_mag_sq_pu = If_r**2 + If_i**2
            It_mag_sq_pu = It_r**2 + It_i**2

            sn_mva = 100.0
            If_ka_baseKV = If_mag_sq_pu * sn_mva**2
            It_ka_baseKV = It_mag_sq_pu * sn_mva**2

            loading_f = If_ka_baseKV / (rate_a**2) if rate_a > 0 else 0
            loading_t = It_ka_baseKV / (rate_a**2) if rate_a > 0 else 0
            loading = np.sqrt(max(loading_f, loading_t))

            assert loading <= 1.01, (
                f"Scenario {scenario}, Branch {from_bus}->{to_bus}: "
                f"Loading {loading:.3f} exceeds 1.01 in secure mode"
            )

    print("    Branch loading limits (secure mode): OK")


def validate_deactivated_generators_zero_output(generated_data):
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


def validate_generator_limits(generated_data):
    """Test that generator outputs respect their limits."""
    gen_data = generated_data["gen_data"]
    gen_data = gen_data[gen_data["in_service"] == 1]
    # keep only the ones with limits for p_mw
    filtered_gens = gen_data[
        gen_data["max_p_mw"].notna() & gen_data["min_p_mw"].notna()
    ]

    if generated_data["mode"] == "unsecure":
        filtered_gens = filtered_gens[filtered_gens["is_ext_grid"] == 0]

    print(
        f"    Generator limits: validating {len(filtered_gens)} active generators (mode: {generated_data['mode']})",
    )
    if not filtered_gens.empty:
        p_within_limits = (
            filtered_gens["p_mw"] >= filtered_gens["min_p_mw"] - 1e-2
        ) & (filtered_gens["p_mw"] <= filtered_gens["max_p_mw"] + 1e-2)
        assert p_within_limits.all(), (
            f"Generator active power should be within limits, expected: {filtered_gens.loc[~p_within_limits, ['bus', 'p_mw']]}, actual: {filtered_gens.loc[~p_within_limits, ['bus', 'p_mw']]}, max: {filtered_gens.loc[~p_within_limits, ['bus', 'max_p_mw']]}"
        )

    if generated_data["mode"] == "secure":
        filtered_gens = filtered_gens[
            filtered_gens["max_q_mvar"].notna() & filtered_gens["min_q_mvar"].notna()
        ]
        q_within_limits = (
            filtered_gens["q_mvar"] >= filtered_gens["min_q_mvar"] - 1e-2
        ) & (filtered_gens["q_mvar"] <= filtered_gens["max_q_mvar"] + 1e-2)
        assert q_within_limits.all(), (
            f"Generator reactive power should be within limits, expected: {filtered_gens.loc[~q_within_limits, ['bus', 'q_mvar']]}, actual: {filtered_gens.loc[~q_within_limits, ['bus', 'q_mvar']]}, max: {filtered_gens.loc[~q_within_limits, ['bus', 'max_q_mvar']]}"
        )

    print("    Generator limits: OK")


def validate_voltage_magnitude_limits_secure_mode(generated_data):
    """Test voltage magnitude limits in secure mode."""
    if generated_data["mode"] != "secure":
        print("    Voltage magnitude limits: skipped (not in secure mode)")
        return

    bus_data = generated_data["bus_data"]
    print(
        f"    Voltage magnitude limits (secure mode): validating {len(bus_data)} bus voltage entries",
    )
    vm_within_limits = (bus_data["Vm"] >= bus_data["min_vm_pu"] - 1e-6) & (
        bus_data["Vm"] <= bus_data["max_vm_pu"] + 1e-6
    )
    assert vm_within_limits.all(), "Voltage magnitudes should be within limits"
    print("    Voltage magnitude limits (secure mode): OK")


def validate_bus_generation_consistency(generated_data):
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


def validate_power_balance_equations(generated_data):
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

            tolerance = 1e-3 if generated_data["mode"] == "unsecure" else 1e-2

            p_balance_error = abs(net_p_injection - total_p_flow)
            q_balance_error = abs(net_q_injection - total_q_flow)

            assert p_balance_error < tolerance, (
                f"Scenario {scenario}, Bus {bus_idx}: Active power balance violation, expected: {net_p_injection}, actual: {total_p_flow}"
            )
            assert q_balance_error < tolerance, (
                f"Scenario {scenario}, Bus {bus_idx}: Reactive power balance violation, expected: {net_q_injection}, actual: {total_q_flow}"
            )

    print("    Power balance equations (Kirchhoff's Law): OK")


def validate_scenario_indexing_consistency(generated_data):
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


def validate_bus_indexing_consistency(generated_data):
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


def validate_data_completeness(generated_data):
    """Test that all required columns are present and no critical data is missing."""
    bus_data = generated_data["bus_data"]
    branch_data = generated_data["branch_data"]
    gen_data = generated_data["gen_data"]
    y_bus_data = generated_data["y_bus_data"]

    total_entries = len(bus_data) + len(branch_data) + len(gen_data) + len(y_bus_data)
    print(
        f"    Data completeness: validating {total_entries} total entries across 4 data files",
    )

    assert "scenario" in bus_data.columns, "Bus data should have scenario column"
    assert "scenario" in branch_data.columns, "Branch data should have scenario column"
    assert "scenario" in gen_data.columns, "Generator data should have scenario column"
    assert "scenario" in y_bus_data.columns, "Y-bus data should have scenario column"

    assert not bus_data["bus"].isna().any(), "Bus indices should not be missing"
    assert not bus_data["Vm"].isna().any(), "Voltage magnitudes should not be missing"
    assert not branch_data["from_bus"].isna().any(), (
        "Branch from_bus should not be missing"
    )
    assert not branch_data["to_bus"].isna().any(), "Branch to_bus should not be missing"
    assert not gen_data["bus"].isna().any(), (
        "Generator bus indices should not be missing"
    )

    assert len(bus_data) > 0, "Bus data should not be empty"
    assert len(branch_data) > 0, "Branch data should not be empty"
    assert len(gen_data) > 0, "Generator data should not be empty"
    assert len(y_bus_data) > 0, "Y-bus data should not be empty"

    print("    Data completeness: OK")


if __name__ == "__main__":
    # debug with data_out/case24_ieee_rts/raw
    file_paths = {
        "bus_data": "data_out/case24_ieee_rts/raw/bus_data.csv",
        "branch_data": "data_out/case24_ieee_rts/raw/branch_data.csv",
        "gen_data": "data_out/case24_ieee_rts/raw/gen_data.csv",
        "y_bus_data": "data_out/case24_ieee_rts/raw/y_bus_data.csv",
    }
    validate_generated_data(file_paths, "unsecure", n_scenarios=10)
    print("Validation completed successfully!")
