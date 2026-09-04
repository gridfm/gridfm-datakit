#!/usr/bin/env julia
"""Correct the fixed PF/OPF scenario matrix with one global worker pool."""

using Distributed
using LinearAlgebra
using PowerModels
using Memento
using Printf
using UUIDs

const SCRIPT_PATH = abspath(@__FILE__)
const DATA_BASE = get(
    ENV,
    "GRIDFM_DATA_BASE",
    "/dccstor/gridfm/powermodels_data/v4/finetuning",
)
const N_SCENARIOS = 10_000
const DEFAULT_WORKERS = 32
const NETWORKS = (
    "case14_ieee",
    "case30_ieee",
    "case57_ieee",
    "case118_ieee",
    "case500_goc",
    "case2000_goc",
    "case10000_goc",
)
const VALID_SPLITS = ("pf", "opf")

Memento.config!("not_set")
BLAS.set_num_threads(1)


function usage(io::IO=stdout)
    println(
        io,
        """
        Usage: julia correct_scenarios.jl [options]

          --workers N       Global Distributed worker count (default: $DEFAULT_WORKERS)
          --splits LIST     Comma-separated subset of pf,opf (default: pf,opf)
          --check-only      Report completeness without writing files
          --help            Show this help
        """,
    )
end


function parse_args(args)
    workers = DEFAULT_WORKERS
    splits = collect(VALID_SPLITS)
    check_only = false
    i = 1

    while i <= length(args)
        arg = args[i]
        if arg == "--workers"
            i == length(args) && error("missing value after --workers")
            workers = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--splits"
            i == length(args) && error("missing value after --splits")
            splits = filter(!isempty, strip.(split(args[i + 1], ",")))
            i += 2
        elseif arg == "--check-only"
            check_only = true
            i += 1
        elseif arg in ("--help", "-h")
            usage()
            return nothing
        else
            error("unknown argument: $arg")
        end
    end

    workers > 0 || error("--workers must be positive")
    isempty(splits) && error("--splits must not be empty")
    length(unique(splits)) == length(splits) ||
        error("--splits contains duplicate values")
    invalid = filter(split -> !(split in VALID_SPLITS), splits)
    isempty(invalid) ||
        error("--splits only accepts pf and opf; invalid: $(join(invalid, ","))")

    return (workers=workers, splits=splits, check_only=check_only)
end


scenario_dir(split, network) =
    joinpath(DATA_BASE, split, network, "powermodels")

raw_path(dir, index) = joinpath(dir, "scenario_$(index).json")

corrected_path(dir, index) =
    joinpath(dir, "scenario_$(index)_corrected.json")


function inspect_inputs(splits)
    records = NamedTuple[]
    jobs = NamedTuple[]
    raw_problems = String[]

    for network in NETWORKS
        for split in splits
            dir = scenario_dir(split, network)
            if !isdir(dir)
                push!(raw_problems, "directory not found: $dir")
                continue
            end

            missing_raw = Int[]
            missing_corrected = Int[]
            for index in 0:(N_SCENARIOS - 1)
                isfile(raw_path(dir, index)) || push!(missing_raw, index)
                isfile(corrected_path(dir, index)) ||
                    push!(missing_corrected, index)
            end

            if !isempty(missing_raw)
                push!(
                    raw_problems,
                    "$split/$network is missing $(length(missing_raw)) raw scenarios; " *
                    "first index=$(first(missing_raw))",
                )
                continue
            end

            existing = N_SCENARIOS - length(missing_corrected)
            push!(
                records,
                (
                    network=network,
                    split=split,
                    dir=dir,
                    existing=existing,
                    missing=length(missing_corrected),
                ),
            )
            for index in missing_corrected
                push!(
                    jobs,
                    (
                        network=network,
                        split=split,
                        index=index,
                        raw=raw_path(dir, index),
                        corrected=corrected_path(dir, index),
                    ),
                )
            end
        end
    end

    if !isempty(raw_problems)
        message = join(("raw scenario validation failed:", raw_problems...), "\n  ")
        error(message)
    end
    return records, jobs
end


function worker_environment()
    return Dict(
        "JULIA_NUM_THREADS" => "1",
        "OPENBLAS_NUM_THREADS" => "1",
        "OMP_NUM_THREADS" => "1",
        "MKL_NUM_THREADS" => "1",
        "BLIS_NUM_THREADS" => "1",
    )
end


function add_correction_workers(count)
    project_file = Base.active_project()
    flags = String["--startup-file=no", "--threads=1"]
    project_file === nothing ||
        pushfirst!(flags, "--project=$(dirname(project_file))")

    worker_ids = addprocs(
        count;
        exeflags=Cmd(flags),
        env=worker_environment(),
    )
    try
        @sync for worker_id in worker_ids
            @async remotecall_fetch(
                path -> begin
                    Base.include(Main, path)
                    nothing
                end,
                worker_id,
                SCRIPT_PATH,
            )
        end
    catch
        rmprocs(worker_ids)
        rethrow()
    end
    return worker_ids
end


function correct_one(job)
    started = time()
    final_path = job.corrected
    isfile(final_path) && return (
        network=job.network,
        split=job.split,
        index=job.index,
        status=:existing,
        elapsed=time() - started,
        error="",
    )

    temp_path = joinpath(
        dirname(final_path),
        ".$(basename(final_path)).tmp.$(getpid()).$(uuid4()).json",
    )
    try
        data = PowerModels.parse_file(job.raw)
        PowerModels.export_file(temp_path, data)
        if isfile(final_path)
            return (
                network=job.network,
                split=job.split,
                index=job.index,
                status=:existing,
                elapsed=time() - started,
                error="",
            )
        end
        mv(temp_path, final_path; force=false)
        return (
            network=job.network,
            split=job.split,
            index=job.index,
            status=:corrected,
            elapsed=time() - started,
            error="",
        )
    catch exception
        message = sprint(showerror, exception, catch_backtrace())
        return (
            network=job.network,
            split=job.split,
            index=job.index,
            status=:failed,
            elapsed=time() - started,
            error=message,
        )
    finally
        isfile(temp_path) && rm(temp_path; force=true)
    end
end


function process_jobs(jobs, worker_count)
    isempty(jobs) && return NamedTuple[], 0.0

    println(
        "Starting $(length(jobs)) missing corrections with " *
        "$worker_count global workers and batch_size=1",
    )
    worker_ids = add_correction_workers(worker_count)
    results = NamedTuple[]
    elapsed = @elapsed try
        results = pmap(
            correct_one,
            WorkerPool(worker_ids),
            jobs;
            batch_size=1,
        )
    finally
        rmprocs(worker_ids)
    end
    return results, elapsed
end


function verify_corrected(records)
    missing_by_key = Dict{Tuple{String,String},Vector{Int}}()
    for record in records
        missing = Int[]
        for index in 0:(N_SCENARIOS - 1)
            isfile(corrected_path(record.dir, index)) || push!(missing, index)
        end
        missing_by_key[(record.network, record.split)] = missing
    end
    return missing_by_key
end


function print_report(records, results, missing_by_key, wall_elapsed; check_only)
    corrected_by_key = Dict{Tuple{String,String},Int}()
    elapsed_by_key = Dict{Tuple{String,String},Float64}()
    failed = filter(result -> result.status == :failed, results)

    for result in results
        key = (result.network, result.split)
        if result.status == :corrected
            corrected_by_key[key] = get(corrected_by_key, key, 0) + 1
        end
        elapsed_by_key[key] =
            get(elapsed_by_key, key, 0.0) + result.elapsed
    end

    println()
    println(check_only ? "Completeness report:" : "Correction report:")
    for record in records
        key = (record.network, record.split)
        corrected = get(corrected_by_key, key, 0)
        final_missing = length(missing_by_key[key])
        worker_elapsed = get(elapsed_by_key, key, 0.0)
        @printf(
            "  %-4s %-15s existing=%5d missing=%5d corrected=%5d final_missing=%5d elapsed=%.3f worker-s\n",
            record.split,
            record.network,
            record.existing,
            record.missing,
            corrected,
            final_missing,
            worker_elapsed,
        )
    end

    total_existing = sum(record.existing for record in records)
    total_missing = sum(record.missing for record in records)
    total_corrected = count(result -> result.status == :corrected, results)
    total_final_missing = sum(length(missing) for missing in values(missing_by_key))
    println()
    @printf(
        "Grand summary: datasets=%d existing=%d missing=%d corrected=%d failed=%d final_missing=%d elapsed=%.3f wall-s\n",
        length(records),
        total_existing,
        total_missing,
        total_corrected,
        length(failed),
        total_final_missing,
        wall_elapsed,
    )

    if !isempty(failed)
        println(stderr, "Correction failures:")
        for result in failed
            println(
                stderr,
                "  $(result.split)/$(result.network)/scenario_$(result.index): " *
                result.error,
            )
        end
    end
    for record in records
        missing = missing_by_key[(record.network, record.split)]
        isempty(missing) && continue
        println(
            stderr,
            "  incomplete $(record.split)/$(record.network): " *
            "$(length(missing)) corrected files missing; first index=$(first(missing))",
        )
    end

    return isempty(failed) && total_final_missing == 0
end


function main(args)
    config = parse_args(args)
    config === nothing && return 0

    overall_started = time()
    println("Data base: $DATA_BASE")
    println("Networks: $(join(NETWORKS, ","))")
    println("Splits: $(join(config.splits, ","))")
    println(
        config.check_only ?
        "Mode: check-only (no files will be written)" :
        "Workers: $(config.workers) global Distributed workers",
    )

    records, jobs = inspect_inputs(config.splits)
    if config.check_only
        results = NamedTuple[]
        correction_elapsed = 0.0
    else
        results, correction_elapsed = process_jobs(jobs, config.workers)
    end

    missing_by_key = verify_corrected(records)
    complete = print_report(
        records,
        results,
        missing_by_key,
        time() - overall_started;
        check_only=config.check_only,
    )
    config.check_only ||
        @printf("Correction dispatch elapsed: %.3f wall-s\n", correction_elapsed)
    return complete ? 0 : 2
end


if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(main(ARGS))
    catch exception
        showerror(stderr, exception, catch_backtrace())
        println(stderr)
        exit(1)
    end
end
