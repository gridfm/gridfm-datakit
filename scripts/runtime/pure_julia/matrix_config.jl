"""Shared configuration for the complete pure-Julia benchmark matrix."""

const MATRIX_MODES = ("pf", "dcpf", "opf", "dcopf")
const MATRIX_SCOPES = ("small", "large")
const MATRIX_SETUPS = Dict(
    "setup1" => "cached-base",
    "setup2" => "per-solve-load",
)

const PROCESS_START = 24
const PROCESS_STEP = 16
const PROCESS_STOP = 216
const INIT_TIMEOUT_S = 900

function gridfm_data_base()
    path = strip(get(ENV, "GRIDFM_DATA_BASE", ""))
    isempty(path) && error(
        "GRIDFM_DATA_BASE is not set. Download " *
        "https://huggingface.co/datasets/gridfm/reproducibility-powermodels-setup2 " *
        "and export GRIDFM_DATA_BASE to that directory. " *
        "Expected: \$GRIDFM_DATA_BASE/{pf,opf}/<network>/powermodels/" *
        "scenario_*_corrected.json",
    )
    return path
end

const MATRIX_NETWORKS = (
    (network = "case14_ieee", count = 4_000_000, scope = "small", pf_fast = true),
    (network = "case30_ieee", count = 3_000_000, scope = "small", pf_fast = true),
    (network = "case57_ieee", count = 2_000_000, scope = "small", pf_fast = true),
    (network = "case118_ieee", count = 2_000_000, scope = "small", pf_fast = true),
    (network = "case500_goc", count = 500_000, scope = "small", pf_fast = true),
    (network = "case2000_goc", count = 50_000, scope = "large", pf_fast = false),
    (network = "case10000_goc", count = 10_000, scope = "large", pf_fast = false),
)

dispatch_batch_size(scope) = scope == "small" ? 32 : 1

function networks_for_scope(scope)
    scope in MATRIX_SCOPES || error("scope must be small or large: $scope")
    return filter(config -> config.scope == scope, MATRIX_NETWORKS)
end

function setup_name(setup)
    haskey(MATRIX_SETUPS, setup) || error("setup must be setup1 or setup2: $setup")
    return MATRIX_SETUPS[setup]
end
