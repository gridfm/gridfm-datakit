# Global variables

# bus types index used in matlab case files
PQ = 1
PV = 2
REF = 3

# Output CSV column definitions used by save/generate pipeline
GEN_COLUMNS = [
    "bus",
    "et",
    "element",
    "p_mw",
    "q_mvar",
    "min_p_mw",
    "max_p_mw",
    "min_q_mvar",
    "max_q_mvar",
    "cp0_eur",
    "cp1_eur_per_mw",
    "cp2_eur_per_mw2",
    "is_gen",
    "is_sgen",
    "is_ext_grid",
    "in_service",
]

BUS_COLUMNS = [
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

DC_BUS_COLUMNS = ["Vm_dc", "Va_dc"]

BRANCH_COLUMNS = [
    "from_bus",
    "to_bus",
    "pf",
    "qf",
    "pt",
    "qt",
    "Yff_r",
    "Yff_i",
    "Yft_r",
    "Yft_i",
    "Ytf_r",
    "Ytf_i",
    "Ytt_r",
    "Ytt_i",
    "tap",
    "ang_min",
    "ang_max",
    "rate_a",
    "br_status",
]
