# Output CSV column definitions used by save/generate pipeline
GEN_COLUMNS = [
    "load_scenario_idx",
    "idx",
    "bus",
    "p_mw",
    "q_mvar",
    "min_p_mw",
    "max_p_mw",
    "min_q_mvar",
    "max_q_mvar",
    "cp0_eur",
    "cp1_eur_per_mw",
    "cp2_eur_per_mw2",
    "in_service",
    "is_slack_gen",
]

DC_GEN_COLUMNS = [
    "p_mw_dc",
]

BUS_COLUMNS = [
    "load_scenario_idx",
    "bus",
    "Pd",
    "Qd",
    "Pg",
    "Qg",
    "Vm",
    "Va",
    "PQ",
    "PV",
    "REF",
    "vn_kv",
    "min_vm_pu",
    "max_vm_pu",
    "GS",
    "BS",
]

DC_BUS_COLUMNS = ["Va_dc", "Pg_dc"]

BRANCH_COLUMNS = [
    "load_scenario_idx",
    "idx",
    "from_bus",
    "to_bus",
    "pf",
    "qf",
    "pt",
    "qt",
    "r",
    "x",
    "b",
    "Yff_r",
    "Yff_i",
    "Yft_r",
    "Yft_i",
    "Ytf_r",
    "Ytf_i",
    "Ytt_r",
    "Ytt_i",
    "tap",
    "shift",
    "ang_min",
    "ang_max",
    "rate_a",
    "br_status",
]

DC_BRANCH_COLUMNS = [
    "pf_dc",
    "pt_dc",
]

YBUS_COLUMNS = [
    "load_scenario_idx",
    "index1",
    "index2",
    "G",
    "B",
]

RUNTIME_COLUMNS = [
    "load_scenario_idx",
    "ac",
]

DC_RUNTIME_COLUMNS = [
    "dc",
]
