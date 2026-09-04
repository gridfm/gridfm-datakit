"""Shared scenario staging for setup2 (per-solve-load)."""

const MAX_SCENARIOS = 10_000

scenario_data_split(mode) = mode in ("pf", "dcpf") ? "pf" : "opf"


function corrected_scenario_path(directory, scenario_index)
    return joinpath(directory, "scenario_$(scenario_index)_corrected.json")
end


function scenario_source_dir(data_base, network, data_split)
    return joinpath(data_base, data_split, network, "powermodels")
end


function active_scenario_count(n_pfs)
    return min(n_pfs, MAX_SCENARIOS)
end


function validate_scenario_source(source_dir, scenario_count)
    isdir(source_dir) || error("PowerModels directory not found: $source_dir")
    missing = String[]
    for scenario_index in 0:(scenario_count - 1)
        path = corrected_scenario_path(source_dir, scenario_index)
        isfile(path) || push!(missing, path)
    end
    if !isempty(missing)
        preview = join(missing[1:min(end, 5)], "\n")
        suffix = length(missing) > 5 ? "\n..." : ""
        error(
            "missing $(length(missing)) corrected scenario file(s):\n" *
            preview *
            suffix,
        )
    end
end


function stage_scenarios_to_tmp(data_base, network, data_split, n_pfs)
    source_dir = scenario_source_dir(data_base, network, data_split)
    scenario_count = active_scenario_count(n_pfs)
    validate_scenario_source(source_dir, scenario_count)

    tmp_root = get(ENV, "TMPDIR", tempdir())
    stage_dir = joinpath(
        tmp_root,
        "pm_scenarios_$(network)_$(data_split)_$(getpid())",
    )
    mkpath(stage_dir)
    started_ns = time_ns()
    for scenario_index in 0:(scenario_count - 1)
        if scenario_index > 0 && scenario_index % 1000 == 0
            println(
                "staging $scenario_index/$scenario_count -> $stage_dir",
            )
            flush(stdout)
        end
        source = corrected_scenario_path(source_dir, scenario_index)
        target = corrected_scenario_path(stage_dir, scenario_index)
        cp(source, target)
    end
    elapsed_s = (time_ns() - started_ns) / 1e9
    println(
        "staged $scenario_count scenarios from $source_dir to $stage_dir in " *
        "$(round(elapsed_s, digits=3)) s",
    )
    flush(stdout)
    return stage_dir, scenario_count
end
