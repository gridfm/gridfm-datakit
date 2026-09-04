#!/usr/bin/env julia
"""Pure-Julia distributed benchmark for cached-base and per-solve-load setups."""

using Distributed
using LinearAlgebra
using PowerModels
using Ipopt
using Memento
using Statistics
using Printf

Memento.config!("not_set")
BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "scenario_staging.jl"))

const SCRIPT_PATH = abspath(@__FILE__)
const DEFAULT_DATA_BASE = get(
    ENV,
    "GRIDFM_DATA_BASE",
    "/dccstor/gridfm/powermodels_data/v4/finetuning",
)
const MODES = ("pf", "dcpf", "opf", "dcopf")
const SETUPS = ("cached-base", "per-solve-load")
const CSV_HEADER = (
    "p",
    "opf_elapsed_s",
    "init_elapsed_s",
    "pf_elapsed_s",
    "n_pfs",
    "total_completed",
    "min_pf_runtime_s",
    "mean_pf_runtime_s",
    "max_pf_runtime_s",
    "min_pf_solve_time_s",
    "mean_pf_solve_time_s",
    "max_pf_solve_time_s",
    "min_worker_completed",
    "mean_worker_completed",
    "max_worker_completed",
    "min_parse_time_s",
    "mean_parse_time_s",
    "max_parse_time_s",
    "successful_count",
    "failed_count",
    "first_error",
)

const WORKER_NETWORK = Ref{Any}(nothing)
const WORKER_SCENARIO_DIR = Ref("")
const WORKER_SCENARIO_COUNT = Ref(0)
const WORKER_SETUP = Ref("")
const WORKER_MODE = Ref("")
const WORKER_PF_FAST = Ref(true)
const WORKER_MAX_ITER = Ref(100) 
const WORKER_TOL = Ref(1e-6)
const WORKER_PRINT_LEVEL = Ref(0)


function parse_flag(args, key, default=nothing)
    i = findfirst(==(key), args)
    i === nothing && return default
    i == length(args) && error("missing value after $key")
    return args[i + 1]
end


has_flag(args, key) = key in args


function parse_int(args, key, default)
    value = parse_flag(args, key, nothing)
    return value === nothing ? default : parse(Int, value)
end


function parse_float(args, key, default)
    value = parse_flag(args, key, nothing)
    return value === nothing ? default : parse(Float64, value)
end


function parse_bool_flag(args, key, default)
    has_flag(args, "--no-$key") && return false
    has_flag(args, "--$key") && return true
    return default
end


function parse_config(args)
    setup = parse_flag(args, "--setup", nothing)
    setup in SETUPS || error("--setup must be cached-base or per-solve-load")
    mode = parse_flag(args, "--mode", "pf")
    mode in MODES || error("unsupported mode: $mode")

    return (
        setup = setup,
        case_file = parse_flag(args, "--case-file", nothing),
        data_base = parse_flag(args, "--data-base", DEFAULT_DATA_BASE),
        network = parse_flag(args, "--network", "case14_ieee"),
        mode = mode,
        pf_fast = parse_bool_flag(args, "pf-fast", true),
        n_pfs = parse_int(args, "--n-pfs", 1_000_000),
        p_start = parse_int(args, "--process-start", 24),
        p_stop = parse_int(args, "--process-stop", 216),
        p_step = parse_int(args, "--process-step", 16),
        dispatch_batch_size = parse_int(args, "--dispatch-batch-size", 32),
        max_iter = parse_int(args, "--max-iter", 100),
        tol = parse_float(args, "--tol", 1e-6),
        print_level = parse_int(args, "--print-level", 0),
        init_timeout_s = parse_float(args, "--init-timeout-s", 900.0),
        resume = has_flag(args, "--resume"),
        output_csv = parse_flag(
            args,
            "--output-csv",
            "benchmark_dynamic_pf_sweep_distributed.csv",
        ),
        staged_scenario_dir = parse_flag(args, "--staged-scenario-dir", nothing),
    )
end


function optimizer(max_iter, tol, print_level)
    return optimizer_with_attributes(
        Ipopt.Optimizer,
        "tol" => tol,
        "print_level" => print_level,
        "max_iter" => max_iter,
    )
end


function solve_one(network, mode, pf_fast, max_iter, tol, print_level)
    result = if mode == "pf"
        pf_fast ?
        compute_ac_pf(network) :
        solve_ac_pf(network, optimizer(max_iter, tol, print_level))
    elseif mode == "dcpf"
        compute_dc_pf(network)
    elseif mode == "opf"
        solve_ac_opf(network, optimizer(max_iter, tol, print_level))
    elseif mode == "dcopf"
        solve_dc_opf(network, optimizer(max_iter, tol, print_level))
    else
        error("unknown mode: $mode")
    end

    status = result["termination_status"]
    if mode in ("pf", "dcpf") && (mode == "dcpf" || pf_fast)
        status == false && error("$(uppercase(mode)) failed")
    else
        string(status) == "LOCALLY_SOLVED" ||
            error("$(uppercase(mode)) failed: $status")
    end

    haskey(result, "solve_time") || error("solver result did not contain solve_time")
    return Float64(result["solve_time"])
end


function set_worker_options!(setup, mode, pf_fast, max_iter, tol, print_level)
    BLAS.set_num_threads(1)
    WORKER_SETUP[] = setup
    WORKER_MODE[] = mode
    WORKER_PF_FAST[] = pf_fast
    WORKER_MAX_ITER[] = max_iter
    WORKER_TOL[] = tol
    WORKER_PRINT_LEVEL[] = print_level
    return nothing
end


function worker_init_cached!(
    case_file,
    mode,
    pf_fast,
    max_iter,
    tol,
    print_level,
)
    set_worker_options!("cached-base", mode, pf_fast, max_iter, tol, print_level)
    WORKER_NETWORK[] = PowerModels.parse_file(case_file)
    solve_one(WORKER_NETWORK[], mode, pf_fast, max_iter, tol, print_level)
    return myid()
end


function worker_init_loading!(
    scenario_directory,
    scenario_count,
    mode,
    pf_fast,
    max_iter,
    tol,
    print_level,
)
    set_worker_options!("per-solve-load", mode, pf_fast, max_iter, tol, print_level)
    WORKER_SCENARIO_DIR[] = scenario_directory
    WORKER_SCENARIO_COUNT[] = scenario_count
    network = PowerModels.parse_file(
        corrected_scenario_path(scenario_directory, 0);
        validate=false,
    )
    solve_one(network, mode, pf_fast, max_iter, tol, print_level)
    return myid()
end


function error_text(exception, backtrace)
    text = sprint(showerror, exception, backtrace)
    return replace(text, '\0' => ' ', '\n' => ' ', '\r' => ' ')
end


function worker_run_job(job_idx)
    started_ns = time_ns()
    parse_time_s = 0.0
    try
        network = if WORKER_SETUP[] == "cached-base"
            WORKER_NETWORK[]
        elseif WORKER_SETUP[] == "per-solve-load"
            count = WORKER_SCENARIO_COUNT[]
            count > 0 || error("worker scenario count is not initialized")
            scenario_index = mod(job_idx - 1, count)
            path = corrected_scenario_path(WORKER_SCENARIO_DIR[], scenario_index)
            parse_started_ns = time_ns()
            parsed = PowerModels.parse_file(path; validate=false)
            parse_time_s = (time_ns() - parse_started_ns) / 1e9
            parsed
        else
            error("worker setup not initialized")
        end

        solve_time_s = solve_one(
            network,
            WORKER_MODE[],
            WORKER_PF_FAST[],
            WORKER_MAX_ITER[],
            WORKER_TOL[],
            WORKER_PRINT_LEVEL[],
        )
        runtime_s = (time_ns() - started_ns) / 1e9
        return (
            worker_id = myid(),
            success = true,
            runtime_s = runtime_s,
            solve_time_s = solve_time_s,
            parse_time_s = parse_time_s,
            error = "",
        )
    catch exception
        backtrace = catch_backtrace()
        runtime_s = (time_ns() - started_ns) / 1e9
        return (
            worker_id = myid(),
            success = false,
            runtime_s = runtime_s,
            solve_time_s = NaN,
            parse_time_s = parse_time_s,
            error = error_text(exception, backtrace),
        )
    end
end


function worker_env()
    return Dict(
        "JULIA_NUM_THREADS" => "1",
        "OPENBLAS_NUM_THREADS" => "1",
        "OMP_NUM_THREADS" => "1",
        "MKL_NUM_THREADS" => "1",
    )
end


function wait_futures(futures, deadline_ns, phase)
    remaining_s = (deadline_ns - time_ns()) / 1e9
    remaining_s > 0 || error("worker initialization timed out during $phase")
    status = timedwait(
        () -> all(isready, futures),
        remaining_s;
        pollint=min(0.1, remaining_s),
    )
    status == :ok ||
        error("worker initialization timed out during $phase after reaching deadline")
    return fetch.(futures)
end


function add_benchmark_workers(p, deadline_ns)
    project_file = Base.active_project()
    project_dir = project_file === nothing ? "" : dirname(project_file)
    exeflags = isempty(project_dir) ?
               `--threads=1` :
               `--project=$project_dir --threads=1`

    remaining_s = (deadline_ns - time_ns()) / 1e9
    remaining_s > 0 || error("worker initialization timed out before addprocs")
    old_timeout = get(ENV, "JULIA_WORKER_TIMEOUT", nothing)
    ENV["JULIA_WORKER_TIMEOUT"] = string(max(1, ceil(Int, remaining_s)))
    ws = Int[]
    try
        ws = addprocs(p; exeflags=exeflags, env=worker_env())
    finally
        if old_timeout === nothing
            delete!(ENV, "JULIA_WORKER_TIMEOUT")
        else
            ENV["JULIA_WORKER_TIMEOUT"] = old_timeout
        end
    end

    try
        includes = [
            remotecall(
                path -> begin
                    Base.include(Main, path)
                    nothing
                end,
                wid,
                SCRIPT_PATH,
            ) for wid in ws
        ]
        wait_futures(includes, deadline_ns, "script loading")
    catch
        rmprocs(ws; waitfor=0)
        rethrow()
    end
    return ws
end


function resolve_staged_scenarios(cfg)
    if cfg.staged_scenario_dir !== nothing
        scenario_count = active_scenario_count(cfg.n_pfs)
        isdir(cfg.staged_scenario_dir) ||
            error("staged scenario directory not found: $(cfg.staged_scenario_dir)")
        validate_scenario_source(cfg.staged_scenario_dir, scenario_count)
        println(
            "using pre-staged scenarios 0:$(scenario_count - 1) from " *
            "$(cfg.staged_scenario_dir)",
        )
        flush(stdout)
        return cfg.staged_scenario_dir, scenario_count, false
    end

    data_split = scenario_data_split(cfg.mode)
    stage_dir, scenario_count = stage_scenarios_to_tmp(
        cfg.data_base,
        cfg.network,
        data_split,
        cfg.n_pfs,
    )
    println(
        "using staged scenarios 0:$(scenario_count - 1) from $stage_dir",
    )
    flush(stdout)
    return stage_dir, scenario_count, true
end


function prepare_opf_network(cfg)
    cfg.case_file === nothing && error("cached-base setup requires --case-file")
    network_file = tempname() * ".json"
    started_ns = time_ns()
    data = PowerModels.parse_file(cfg.case_file)
    result = solve_ac_opf(
        data,
        optimizer(cfg.max_iter, cfg.tol, cfg.print_level),
    )
    string(result["termination_status"]) == "LOCALLY_SOLVED" ||
        error("OPF failed: $(result["termination_status"])")
    PowerModels.update_data!(data, result["solution"])
    PowerModels.export_file(network_file, data)
    elapsed_s = (time_ns() - started_ns) / 1e9
    println(
        "OPF prepared network in $(round(elapsed_s, digits=3)) s " *
        "(solver solve_time=$(result["solve_time"]) s): $network_file",
    )
    return network_file, elapsed_s
end


function initialize_workers!(
    ws,
    cfg,
    worker_case_file,
    scenario_dir,
    scenario_count,
    deadline_ns,
)
    futures = if cfg.setup == "cached-base"
        [
            remotecall(
                worker_init_cached!,
                wid,
                worker_case_file,
                cfg.mode,
                cfg.pf_fast,
                cfg.max_iter,
                cfg.tol,
                cfg.print_level,
            ) for wid in ws
        ]
    else
        [
            remotecall(
                worker_init_loading!,
                wid,
                scenario_dir,
                scenario_count,
                cfg.mode,
                cfg.pf_fast,
                cfg.max_iter,
                cfg.tol,
                cfg.print_level,
            ) for wid in ws
        ]
    end
    wait_futures(futures, deadline_ns, "worker warmup")
    return nothing
end


function summary_stats(values)
    isempty(values) && return (NaN, NaN, NaN)
    return (minimum(values), mean(values), maximum(values))
end


function aggregate_result(cfg, p, opf_elapsed, init_elapsed, pf_elapsed, results, ws)
    length(results) == cfg.n_pfs ||
        error("expected $(cfg.n_pfs) attempted results, got $(length(results))")

    successful = filter(result -> result.success, results)
    failed = filter(result -> !result.success, results)
    runtime_stats = summary_stats(getproperty.(successful, :runtime_s))
    solve_stats = summary_stats(getproperty.(successful, :solve_time_s))
    parse_stats = cfg.setup == "cached-base" ?
                  (0.0, 0.0, 0.0) :
                  summary_stats(getproperty.(successful, :parse_time_s))

    counts = Dict(wid => 0 for wid in ws)
    for result in results
        counts[result.worker_id] = get(counts, result.worker_id, 0) + 1
    end
    completed_by_worker = [counts[wid] for wid in ws]

    return (
        p = p,
        opf_elapsed_s = opf_elapsed,
        init_elapsed_s = init_elapsed,
        pf_elapsed_s = pf_elapsed,
        n_pfs = cfg.n_pfs,
        total_completed = length(results),
        min_pf_runtime_s = runtime_stats[1],
        mean_pf_runtime_s = runtime_stats[2],
        max_pf_runtime_s = runtime_stats[3],
        min_pf_solve_time_s = solve_stats[1],
        mean_pf_solve_time_s = solve_stats[2],
        max_pf_solve_time_s = solve_stats[3],
        min_worker_completed = minimum(completed_by_worker),
        mean_worker_completed = mean(completed_by_worker),
        max_worker_completed = maximum(completed_by_worker),
        min_parse_time_s = parse_stats[1],
        mean_parse_time_s = parse_stats[2],
        max_parse_time_s = parse_stats[3],
        successful_count = length(successful),
        failed_count = length(failed),
        first_error = isempty(failed) ? "" : first(failed).error,
    )
end


format_float(value) = @sprintf("%.6f", value)


function csv_escape(value)
    text = string(value)
    if any(character -> character in (',', '"', '\n', '\r'), text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end


function csv_row(result)
    return (
        string(result.p),
        format_float(result.opf_elapsed_s),
        format_float(result.init_elapsed_s),
        format_float(result.pf_elapsed_s),
        string(result.n_pfs),
        string(result.total_completed),
        format_float(result.min_pf_runtime_s),
        format_float(result.mean_pf_runtime_s),
        format_float(result.max_pf_runtime_s),
        format_float(result.min_pf_solve_time_s),
        format_float(result.mean_pf_solve_time_s),
        format_float(result.max_pf_solve_time_s),
        string(result.min_worker_completed),
        format_float(result.mean_worker_completed),
        string(result.max_worker_completed),
        format_float(result.min_parse_time_s),
        format_float(result.mean_parse_time_s),
        format_float(result.max_parse_time_s),
        string(result.successful_count),
        string(result.failed_count),
        result.first_error,
    )
end


function split_csv_line(line)
    fields = String[]
    buffer = IOBuffer()
    quoted = false
    closed_quote = false
    index = firstindex(line)
    while index <= lastindex(line)
        character = line[index]
        if quoted
            if character == '"'
                next_index = nextind(line, index)
                if next_index <= lastindex(line) && line[next_index] == '"'
                    write(buffer, '"')
                    index = next_index
                else
                    quoted = false
                    closed_quote = true
                end
            else
                write(buffer, character)
            end
        elseif character == ','
            push!(fields, String(take!(buffer)))
            closed_quote = false
        elseif character == '"'
            position(buffer) == 0 && !closed_quote ||
                error("malformed CSV: quote inside unquoted field")
            quoted = true
        elseif closed_quote
            error("malformed CSV: characters after closing quote")
        else
            write(buffer, character)
        end
        index = nextind(line, index)
    end
    quoted && error("malformed CSV: unterminated quoted field")
    push!(fields, String(take!(buffer)))
    return fields
end


function validate_resume_field_parsing(fields, line_number)
    integer_indices = (1, 5, 6, 13, 15, 19, 20)
    float_indices = (2, 3, 4, 7, 8, 9, 10, 11, 12, 14, 16, 17, 18)
    for index in integer_indices
        try
            parse(Int, fields[index])
        catch
            error(
                "malformed resume CSV row $line_number: " *
                "$(CSV_HEADER[index]) is not an integer",
            )
        end
    end
    for index in float_indices
        try
            parse(Float64, fields[index])
        catch
            error(
                "malformed resume CSV row $line_number: " *
                "$(CSV_HEADER[index]) is not numeric",
            )
        end
    end
end


function load_resume_rows(path, cfg, requested_ps)
    rows = Dict{Int,Vector{String}}()
    !isfile(path) && return rows
    lines = readlines(path)
    isempty(lines) && error("resume CSV is empty: $path")
    header = try
        split_csv_line(lines[1])
    catch exception
        error("malformed resume CSV header: $(sprint(showerror, exception))")
    end
    header == collect(CSV_HEADER) || error(
        "incompatible resume CSV header in $path\n" *
        "expected: $(join(CSV_HEADER, ","))\n" *
        "found: $(join(header, ","))",
    )

    requested = Set(requested_ps)
    for (offset, line) in enumerate(lines[2:end])
        line_number = offset + 1
        isempty(strip(line)) && error("malformed resume CSV: blank row $line_number")
        fields = try
            split_csv_line(line)
        catch exception
            error("malformed resume CSV row $line_number: $(sprint(showerror, exception))")
        end
        length(fields) == length(CSV_HEADER) || error(
            "malformed resume CSV row $line_number: expected " *
            "$(length(CSV_HEADER)) fields, found $(length(fields))",
        )
        validate_resume_field_parsing(fields, line_number)
        p = parse(Int, fields[1])
        p in requested ||
            error("incompatible resume CSV row $line_number: p=$p is not requested")
        haskey(rows, p) &&
            error("duplicate p=$p in resume CSV (row $line_number)")
        parse(Int, fields[5]) == cfg.n_pfs || error(
            "incompatible resume CSV row $line_number: n_pfs=$(fields[5]), " *
            "requested $(cfg.n_pfs)",
        )
        parse(Int, fields[6]) == cfg.n_pfs || error(
            "incompatible resume CSV row $line_number: total_completed=$(fields[6]), " *
            "expected $(cfg.n_pfs)",
        )
        successful_count = parse(Int, fields[19])
        failed_count = parse(Int, fields[20])
        successful_count >= 0 && failed_count >= 0 ||
            error("malformed resume CSV row $line_number: negative status count")
        successful_count + failed_count == cfg.n_pfs || error(
            "incompatible resume CSV row $line_number: successful_count + " *
            "failed_count does not equal n_pfs",
        )
        rows[p] = fields
    end
    return rows
end


function write_results_csv(path, rows)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, join(CSV_HEADER, ","))
        for p in sort(collect(keys(rows)))
            println(io, join(csv_escape.(rows[p]), ","))
        end
    end
end


function benchmark_for_p(
    cfg,
    p,
    worker_case_file,
    scenario_dir,
    scenario_count,
    opf_elapsed,
)
    println()
    println(
        "=== setup=$(cfg.setup) mode=$(cfg.mode) pf_fast=$(cfg.pf_fast) " *
        "p=$p batch=$(cfg.dispatch_batch_size) ===",
    )

    init_started_ns = time_ns()
    deadline_ns = init_started_ns + round(UInt64, cfg.init_timeout_s * 1e9)
    ws = Int[]
    initialized = false
    try
        ws = add_benchmark_workers(p, deadline_ns)
        initialize_workers!(
            ws,
            cfg,
            worker_case_file,
            scenario_dir,
            scenario_count,
            deadline_ns,
        )
        initialized = true
        init_elapsed = (time_ns() - init_started_ns) / 1e9

        pool = WorkerPool(ws)
        pf_started_ns = time_ns()
        results = pmap(
            worker_run_job,
            pool,
            1:cfg.n_pfs;
            batch_size=cfg.dispatch_batch_size,
        )
        pf_elapsed = (time_ns() - pf_started_ns) / 1e9
        result = aggregate_result(
            cfg,
            p,
            opf_elapsed,
            init_elapsed,
            pf_elapsed,
            results,
            ws,
        )
        println(
            "p=$p init=$(round(init_elapsed, digits=3)) s " *
            "elapsed=$(round(pf_elapsed, digits=3)) s " *
            "success=$(result.successful_count) failed=$(result.failed_count) " *
            "runtime_mean=$(round(result.mean_pf_runtime_s, digits=6)) s " *
            "solve_mean=$(round(result.mean_pf_solve_time_s, digits=6)) s",
        )
        println(
            "worker attempted min/mean/max = " *
            "$(result.min_worker_completed) / " *
            "$(round(result.mean_worker_completed, digits=2)) / " *
            "$(result.max_worker_completed)",
        )
        return result
    finally
        if !isempty(ws)
            initialized ? rmprocs(ws) : rmprocs(ws; waitfor=0)
        end
    end
end


function validate_config(cfg)
    cfg.n_pfs > 0 || error("--n-pfs must be positive")
    cfg.p_start > 0 || error("--process-start must be positive")
    cfg.p_stop >= cfg.p_start || error("--process-stop must be >= --process-start")
    cfg.p_step > 0 || error("--process-step must be positive")
    cfg.dispatch_batch_size > 0 || error("--dispatch-batch-size must be positive")
    isfinite(cfg.init_timeout_s) && cfg.init_timeout_s > 0 ||
        error("--init-timeout-s must be positive and finite")
    if cfg.setup == "cached-base"
        cfg.case_file === nothing && error("cached-base setup requires --case-file")
        isfile(cfg.case_file) || error("case file not found: $(cfg.case_file)")
    end
    return nothing
end


function main(args)
    cfg = parse_config(args)
    validate_config(cfg)
    requested_ps = collect(cfg.p_start:cfg.p_step:cfg.p_stop)
    println("setup=$(cfg.setup) network=$(cfg.network) mode=$(cfg.mode)")
    flush(stdout)
    println(
        "n_pfs=$(cfg.n_pfs) p=$(cfg.p_start):$(cfg.p_step):$(cfg.p_stop) " *
        "dispatch_batch_size=$(cfg.dispatch_batch_size) " *
        "init_timeout_s=$(cfg.init_timeout_s)",
    )
    println("output_csv=$(cfg.output_csv) resume=$(cfg.resume)")
    flush(stdout)

    rows = cfg.resume ?
           load_resume_rows(cfg.output_csv, cfg, requested_ps) :
           Dict{Int,Vector{String}}()
    cfg.resume && isfile(cfg.output_csv) && write_results_csv(cfg.output_csv, rows)
    missing_ps = filter(p -> !haskey(rows, p), requested_ps)
    if cfg.resume
        println(
            "resume: completed=$(length(rows)) missing=$(length(missing_ps))",
        )
    end

    worker_case_file = nothing
    scenario_dir = ""
    staged_scenario_dir = ""
    owns_staged_scenario_dir = false
    scenario_count = 0
    opf_elapsed = 0.0

    try
        if !isempty(missing_ps)
            if cfg.setup == "cached-base"
                if cfg.mode in ("pf", "dcpf")
                    worker_case_file, opf_elapsed = prepare_opf_network(cfg)
                else
                    worker_case_file = cfg.case_file
                end
            else
                staged_scenario_dir, scenario_count, owns_staged_scenario_dir =
                    resolve_staged_scenarios(cfg)
                scenario_dir = staged_scenario_dir
            end
        end

        for p in missing_ps
            result = benchmark_for_p(
                cfg,
                p,
                worker_case_file,
                scenario_dir,
                scenario_count,
                opf_elapsed,
            )
            rows[p] = collect(csv_row(result))
            write_results_csv(cfg.output_csv, rows)
        end
    finally
        if cfg.setup == "cached-base" &&
           cfg.mode in ("pf", "dcpf") &&
           worker_case_file !== nothing &&
           isfile(worker_case_file)
            rm(worker_case_file; force=true)
        end
        if owns_staged_scenario_dir &&
           !isempty(staged_scenario_dir) &&
           isdir(staged_scenario_dir)
            rm(staged_scenario_dir; recursive=true, force=true)
            println("removed staged scenario directory $staged_scenario_dir")
            flush(stdout)
        end
    end

    if isempty(missing_ps) && !isfile(cfg.output_csv)
        write_results_csv(cfg.output_csv, rows)
    end
    println()
    println("finished setup=$(cfg.setup) -> $(cfg.output_csv)")
end


if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
