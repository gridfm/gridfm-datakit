#!/usr/bin/env julia

include(joinpath(@__DIR__, "matrix_config.jl"))

const SCRIPT_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(SCRIPT_DIR, "..", "..", ".."))
const BENCHMARK = joinpath(SCRIPT_DIR, "benchmark_distributed.jl")
const GRIDS_DIR = joinpath(REPO_ROOT, "gridfm_datakit", "grids")
const OUTPUT_ROOT = joinpath(
    REPO_ROOT,
    "scripts",
    "runtime",
    "outputs_julia",
    "full_matrix",
)
function usage(io=stdout)
    println(io, "Usage: run_matrix.jl --scope small|large --setup setup1|setup2 [--dry-run]")
end

function option_value(args, option)
    indices = findall(==(option), args)
    length(indices) <= 1 || error("$option may only be provided once")
    isempty(indices) && return nothing
    index = only(indices)
    index < length(args) || error("missing value after $option")
    startswith(args[index + 1], "--") && error("missing value after $option")
    return args[index + 1]
end

function parse_args(args)
    if "--help" in args || "-h" in args
        usage()
        return nothing
    end

    allowed = Set(["--scope", "--setup", "--dry-run"])
    index = 1
    while index <= length(args)
        arg = args[index]
        arg in allowed || error("unknown argument: $arg")
        index += arg == "--dry-run" ? 1 : 2
    end

    scope = option_value(args, "--scope")
    setup = option_value(args, "--setup")
    scope in MATRIX_SCOPES || error("--scope must be small or large")
    haskey(MATRIX_SETUPS, setup) || error("--setup must be setup1 or setup2")
    return (scope = scope, setup = setup, dry_run = "--dry-run" in args)
end

function corrected_case_file(network)
    return joinpath(GRIDS_DIR, "pglib_opf_$(network)_corrected.m")
end

function benchmark_command(config, mode, scope, setup, output_csv)
    args = String[
        BENCHMARK,
        "--setup", setup_name(setup),
        "--network", config.network,
        "--mode", mode,
        "--n-pfs", string(config.count),
        "--process-start", string(PROCESS_START),
        "--process-stop", string(PROCESS_STOP),
        "--process-step", string(PROCESS_STEP),
        "--dispatch-batch-size", string(dispatch_batch_size(scope)),
        "--output-csv", output_csv,
        "--resume",
        "--init-timeout-s", string(INIT_TIMEOUT_S),
    ]
    if mode == "pf"
        push!(args, config.pf_fast ? "--pf-fast" : "--no-pf-fast")
    end
    if setup == "setup1"
        append!(args, ["--case-file", corrected_case_file(config.network)])
    else
        append!(args, ["--data-base", gridfm_data_base()])
    end
    return Cmd(vcat(Base.julia_cmd().exec, args))
end

function validate_files(configs, setup, dry_run)
    if !dry_run
        isfile(BENCHMARK) || error("benchmark engine not found: $BENCHMARK")
    end
    if setup == "setup1"
        missing = filter(path -> !isfile(path), corrected_case_file.(getproperty.(configs, :network)))
        isempty(missing) || error("corrected case file not found: $(first(missing))")
    elseif setup == "setup2"
        base = gridfm_data_base()
        dry_run || isdir(base) || error("data base not found: $base")
    end
end

function main(args)
    options = parse_args(args)
    options === nothing && return

    configs = networks_for_scope(options.scope)
    validate_files(configs, options.setup, options.dry_run)
    output_dir = joinpath(OUTPUT_ROOT, options.scope, options.setup)
    options.dry_run || mkpath(output_dir)

    for config in configs, mode in MATRIX_MODES
        output_csv = joinpath(output_dir, "benchmark_$(config.network)_$(mode).csv")
        command = benchmark_command(config, mode, options.scope, options.setup, output_csv)
        println(command)
        flush(stdout)
        options.dry_run || run(command)
    end

    println(options.dry_run ? "Matrix validation complete." : "Matrix complete: $output_dir")
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main(ARGS)
    catch err
        usage(stderr)
        showerror(stderr, err)
        println(stderr)
        exit(1)
    end
end
