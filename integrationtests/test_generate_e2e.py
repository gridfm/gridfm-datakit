"""End-to-end integration tests: generate a dataset through the CLI and check it.

Each case
1. derives a config from the shipped ``scripts/config/default.yaml`` (seeded, small),
2. runs ``gridfm_datakit generate`` and ``gridfm_datakit validate`` as subprocesses,
3. computes summary metrics from the generated parquet files, and
4. asserts them against fixed physical ranges and the calibrated baseline.

Run ``pytest integrationtests --calibrate -s`` on unchanged code to (re)write the
baseline; plain ``pytest integrationtests -s`` asserts against it.
"""

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import yaml

from integrationtests.baseline import (
    check_bounds,
    compute_bounds,
    print_calibration_stats,
    read_baseline,
    write_baseline,
)
from integrationtests.metrics import NOISE_METRICS, compute_metrics, physical_ranges

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
            "seed": SEED,
        },
    )
    for section, values in overrides.items():
        config[section].update(values)
    return config


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
    return compute_metrics(raw_dir, sn_mva=SN_MVA)


@pytest.mark.parametrize("case", sorted(CASES))
def test_generate_end_to_end(case, tmp_path, calibrate_runs, ci_level, calibrate_pad):
    expected_scenarios = LOAD_SCENARIOS * TOPOLOGY_VARIANTS
    n_runs = max(calibrate_runs, 1)
    all_runs = []
    for run_i in range(n_runs):
        print(f"\n--- {case.upper()} generation run {run_i + 1}/{n_runs} ---")
        metrics = generate_and_measure(case, str(tmp_path))
        # Physical validity is checked on every run, calibration included: a
        # baseline must never be calibrated from broken data.
        check_bounds(
            metrics,
            physical_ranges(CASES[case]["settings"]["mode"], expected_scenarios),
            f"{case} physical ranges (run {run_i + 1})",
        )
        all_runs.append(metrics)

    calibrated_keys = sorted(k for k in all_runs[0] if k not in NOISE_METRICS)

    if calibrate_runs > 0:
        # Tolerate float summation-order noise (~1e-15) from parallel workers.
        nondeterministic = [
            k
            for k in calibrated_keys
            if np.ptp([run[k] for run in all_runs])
            > 1e-9 * max(abs(all_runs[0][k]), 1.0)
        ]
        if nondeterministic:
            print(
                f"\nNOTE: {len(nondeterministic)} metric(s) differ between seeded runs "
                f"(generation is not bit-reproducible here): {nondeterministic}",
            )
        bounds = compute_bounds(all_runs, calibrated_keys, ci_level, calibrate_pad)
        print_calibration_stats(all_runs, calibrated_keys, bounds, ci_level)
        write_baseline(case, bounds)
        return

    checks = read_baseline(case)
    missing = sorted(set(calibrated_keys) - set(checks))
    assert not missing, (
        f"Metrics {missing} have no calibrated bounds in the baseline for '{case}'. "
        f"Recalibrate on the code before your change: pytest integrationtests --calibrate -s"
    )
    check_bounds(all_runs[0], checks, f"{case} calibrated baseline")
