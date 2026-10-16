"""End-to-end integration tests: generate a dataset through the CLI and check it.

Each case
1. derives a config from the shipped ``scripts/config/default.yaml`` (seeded, small),
2. runs ``gridfm_datakit generate`` and ``gridfm_datakit validate`` as subprocesses,
3. computes summary metrics from the generated parquet files, and
4. asserts them against the calibrated baseline.

``validate`` checks that every scenario is physically valid; the baseline catches
changes in *what* is generated, which ``validate`` cannot see.

Run ``pytest integrationtests --calibrate -s`` on unchanged code to (re)write the
baseline; plain ``pytest integrationtests -s`` asserts against it.
"""

import os
import shutil
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from integrationtests.baseline import (
    check_bounds,
    compute_bounds,
    print_calibration_stats,
    read_baseline,
    write_baseline,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "scripts", "config", "default.yaml")

NETWORK = "case14_ieee"
LOAD_SCENARIOS = 200
TOPOLOGY_VARIANTS = 2
SEED = 42
SN_MVA = 100.0

# Changing any of these invalidates the committed baseline: recalibrate afterwards.
CASES = {
    "pf": {"settings": {"mode": "pf"}},
    "opf": {"settings": {"mode": "opf"}},
}


def build_config(overrides: dict, data_dir: str) -> dict:
    with open(DEFAULT_CONFIG) as f:
        config = yaml.safe_load(f)
    config["network"]["name"] = NETWORK
    config["load"]["scenarios"] = LOAD_SCENARIOS
    config["topology_perturbation"]["n_topology_variants"] = TOPOLOGY_VARIANTS
    config["settings"].update(
        {
            "data_dir": data_dir,
            "num_processes": min(4, os.cpu_count() or 1),
            "large_chunk_size": 100,
            "overwrite": True,
            "enable_solver_logs": False,
            "include_dc_res": True,
            "seed": SEED,
        },
    )
    for section, values in overrides.items():
        config[section].update(values)
    return config


def compute_metrics(raw_dir: str) -> dict:
    """Scalar aggregates over all scenarios; solver runtimes are left out (not reproducible)."""
    bus = pd.read_parquet(os.path.join(raw_dir, "bus_data.parquet"))
    branch = pd.read_parquet(os.path.join(raw_dir, "branch_data.parquet"))
    gen = pd.read_parquet(os.path.join(raw_dir, "gen_data.parquet"))
    gen = gen[gen["in_service"] == 1]
    branch = branch[branch["br_status"] == 1]
    rated = branch[branch["rate_a"] > 0]
    loading = (
        pd.concat(
            [
                rated["pf"].pow(2) + rated["qf"].pow(2),
                rated["pt"].pow(2) + rated["qt"].pow(2),
            ],
            axis=1,
        )
        .max(axis=1)
        .pow(0.5)
        / rated["rate_a"]
    )
    cost = (
        gen["cp0_eur"]
        + gen["cp1_eur_per_mw"] * gen["p_mw"]
        + gen["cp2_eur_per_mw2"] * gen["p_mw"] ** 2
    )

    return {
        "n_scenarios": float(bus["scenario"].nunique()),
        "mean_total_pd_mw": float(bus.groupby("scenario")["Pd"].sum().mean()),
        "mean_total_qd_mvar": float(bus.groupby("scenario")["Qd"].sum().mean()),
        "mean_total_pg_mw": float(gen.groupby("scenario")["p_mw"].sum().mean()),
        "mean_total_qg_mvar": float(gen.groupby("scenario")["q_mvar"].sum().mean()),
        "mean_total_gen_cost": float(cost.groupby(gen["scenario"]).sum().mean()),
        "mean_active_gens": float(gen.groupby("scenario").size().mean()),
        "mean_active_branches": float(branch.groupby("scenario").size().mean()),
        "mean_vm_pu": float(bus["Vm"].mean()),
        "min_vm_pu": float(bus["Vm"].min()),
        "max_vm_pu": float(bus["Vm"].max()),
        "mean_abs_va_deg": float(bus["Va"].abs().mean()),
        "mean_abs_va_ac_minus_dc_deg": float((bus["Va"] - bus["Va_dc"]).abs().mean()),
        "mean_abs_pf_mw": float(branch["pf"].abs().mean()),
        "mean_branch_loading": float(loading.mean()),
        "max_branch_loading": float(loading.max()),
    }


def run_cli(*args: str) -> None:
    # The console script lives next to the interpreter running pytest.
    exe = shutil.which("gridfm_datakit", path=os.path.dirname(sys.executable))
    cmd = [exe or "gridfm_datakit", *args]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def generate_and_measure(case: str, work_dir: str) -> dict:
    """Generate one dataset for ``case`` from scratch, validate it, return its metrics."""
    data_dir = os.path.join(work_dir, "data")
    shutil.rmtree(data_dir, ignore_errors=True)
    config = build_config(CASES[case], data_dir)
    config_path = os.path.join(work_dir, f"{case}.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    run_cli("generate", config_path)
    raw_dir = os.path.join(data_dir, NETWORK, "raw")
    run_cli(
        "validate",
        raw_dir,
        "--n-partitions",
        "0",
        "--mode",
        config["settings"]["mode"],
        "--sn-mva",
        str(SN_MVA),
    )
    return compute_metrics(raw_dir)


@pytest.mark.parametrize("case", sorted(CASES))
def test_generate_end_to_end(case, tmp_path, calibrate_runs, ci_level, calibrate_pad):
    n_runs = max(calibrate_runs, 1)
    all_runs = []
    for run_i in range(n_runs):
        print(f"\n--- {case.upper()} generation run {run_i + 1}/{n_runs} ---")
        all_runs.append(generate_and_measure(case, str(tmp_path)))
    metric_keys = sorted(all_runs[0])

    if calibrate_runs > 0:
        bounds = compute_bounds(all_runs, metric_keys, ci_level, calibrate_pad)
        print_calibration_stats(all_runs, metric_keys, bounds, ci_level)
        write_baseline(case, bounds)
        return

    checks = read_baseline(case)
    missing = sorted(set(metric_keys) - set(checks))
    assert not missing, (
        f"Metrics {missing} have no calibrated bounds in the baseline for '{case}'. "
        f"Recalibrate on the code before your change: pytest integrationtests --calibrate -s"
    )
    check_bounds(all_runs[0], checks, f"{case} calibrated baseline")
