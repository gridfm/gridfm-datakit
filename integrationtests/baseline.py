"""Calibrated metric bounds: compute, store, load and check them.

The baseline JSON holds, per test key, a ``(lo, hi)`` interval for every
calibrated metric plus the environment fingerprint it was measured in.
"""

import importlib.metadata as importlib_metadata
import json
import os
import platform
import subprocess
import warnings

import numpy as np
from scipy import stats

# The committed baseline. Data generation runs on the CPU and is seeded, so these
# bounds are expected to hold on other machines too; if they do not, recalibrate on
# your machine (before changing any code) and point GRIDFM_CALIBRATION_BASELINE at it.
DEFAULT_BASELINE_PATH = os.path.join(
    os.path.dirname(__file__),
    "calibration_baseline.json",
)


def baseline_path() -> str:
    return os.environ.get("GRIDFM_CALIBRATION_BASELINE", DEFAULT_BASELINE_PATH)


def _cpu_model() -> str:
    """Best-effort CPU model string.

    Linux: /proc/cpuinfo (platform.processor() is usually empty).
    macOS: sysctl machdep.cpu.brand_string (platform.processor() is just "arm").
    """
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def _julia_package_versions() -> dict:
    """Pinned Julia solver versions from juliapkg.json (what juliacall installs)."""
    import gridfm_datakit

    path = os.path.join(os.path.dirname(gridfm_datakit.__file__), "juliapkg.json")
    try:
        with open(path) as f:
            packages = json.load(f).get("packages", {})
    except (OSError, ValueError):
        return {}
    return {name: spec.get("version") for name, spec in sorted(packages.items())}


def environment_fingerprint() -> dict:
    """Capture the hardware + software stack that can shift solver numerics.

    Calibrated bounds are measured in one environment; this fingerprint lets the
    assertion run detect (and warn about) a baseline from a different one.
    """
    fp = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_model(),
        "python_version": platform.python_version(),
        "julia_packages": _julia_package_versions(),
    }
    for pkg in ("gridfm_datakit", "numpy", "scipy", "pandas", "juliacall"):
        try:
            fp[f"{pkg}_version"] = importlib_metadata.version(pkg.replace("_", "-"))
        except importlib_metadata.PackageNotFoundError:
            fp[f"{pkg}_version"] = None
    return fp


def warn_on_fingerprint_mismatch(saved: dict, current: dict) -> None:
    """Emit a loud warning listing every field that differs (warn-and-proceed)."""
    if not saved:
        warnings.warn(
            "Calibration baseline has no environment fingerprint; cannot verify "
            "which environment it was calibrated in.",
            stacklevel=2,
        )
        return
    diffs = []
    for key in sorted(set(saved) | set(current)):
        if saved.get(key) != current.get(key):
            diffs.append(
                f"    {key}: baseline={saved.get(key)!r}  current={current.get(key)!r}",
            )
    if diffs:
        warnings.warn(
            "Calibration baseline was measured in a DIFFERENT environment; metric "
            "bounds may not hold here. If a bound fails, re-run "
            "`pytest integrationtests --calibrate -s` on the unchanged code "
            "(with GRIDFM_CALIBRATION_BASELINE pointing at a local file) before "
            "making changes. Differing fields:\n" + "\n".join(diffs),
            stacklevel=2,
        )


def compute_bounds(
    all_runs: list,
    metric_keys: list,
    confidence_interval: float,
    pad: float,
) -> dict:
    """Compute per-metric (lo, hi) bounds from calibration runs.

    lo/hi = mean +/- max(student-t margin of error, pad * |mean|)

    The ``pad`` floor keeps the interval from collapsing to zero width when the
    std across seeded, deterministic runs is zero, while metrics whose mean is
    exactly 0 stay exactly (0, 0).
    """
    n = len(all_runs)
    alpha_half = (1 + confidence_interval) / 2
    t_crit = stats.t.ppf(alpha_half, df=max(n - 1, 1))
    bounds = {}
    for key in metric_keys:
        values = [run[key] for run in all_runs if key in run]
        if not values:
            continue
        arr = np.array(values, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        me = t_crit * std / np.sqrt(len(arr))
        half_width = max(me, pad * abs(mean))
        bounds[key] = (mean - half_width, mean + half_width)
    return bounds


def write_baseline(test_key: str, bounds: dict) -> None:
    """Merge this test's calibrated bounds into the baseline JSON on disk.

    Each test's bounds are stamped with their own environment fingerprint, so
    calibrating one test never re-stamps another test's bounds.
    """
    path = baseline_path()
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    data.setdefault("bounds", {})
    data.setdefault("fingerprints", {})
    data["bounds"][test_key] = {k: list(v) for k, v in bounds.items()}
    data["fingerprints"][test_key] = environment_fingerprint()
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\nCalibrated bounds for '{test_key}' written to {path}")


def read_baseline(test_key: str) -> dict:
    """Load this test's calibrated bounds, or fail with a calibration hint."""
    path = baseline_path()
    assert os.path.exists(path), (
        f"No calibration baseline at {path}. Calibrate first, on the code **before** "
        f"your changes:\n    pytest integrationtests --calibrate -s"
    )
    with open(path) as f:
        data = json.load(f)
    bounds = data.get("bounds", {})
    assert test_key in bounds, (
        f"Baseline {path} has no entry for '{test_key}'. Calibrate it on the code "
        f"**before** your changes: pytest integrationtests --calibrate -s"
    )
    saved_fp = data.get("fingerprints", {}).get(test_key, {})
    warn_on_fingerprint_mismatch(saved_fp, environment_fingerprint())
    return {k: tuple(v) for k, v in bounds[test_key].items()}


def check_bounds(metrics: dict, checks: dict, label: str) -> None:
    """Assert every metric lies in its (lo, hi) interval, reporting all failures at once."""
    failures = []
    for metric_name, (lo, hi) in sorted(checks.items()):
        if metric_name not in metrics:
            failures.append(f"  {metric_name}: missing from computed metrics")
            continue
        value = metrics[metric_name]
        ok = lo <= value <= hi
        print(
            f"  [{'ok' if ok else 'FAIL'}] {metric_name}: {value!r} in [{lo!r}, {hi!r}]",
        )
        if not ok:
            failures.append(f"  {metric_name}: {value!r} outside [{lo!r}, {hi!r}]")
    assert not failures, (
        f"{label}: {len(failures)} metric(s) out of range:\n"
        + "\n".join(
            failures,
        )
    )


def print_calibration_stats(
    all_runs: list,
    metric_keys: list,
    bounds: dict,
    confidence_interval: float,
) -> None:
    """Print mean, std (ddof=1) and the calibrated bounds for each metric.

    The lo/hi columns are the same bounds written to the baseline, not recomputed.
    """
    n = len(all_runs)
    ci_pct = f"{confidence_interval * 100:g}"
    col_w = max(len(k) for k in metric_keys) + 2
    header = (
        f"  {'Metric':<{col_w}}  {'Mean':>14}  {'Std(ddof=1)':>12}  "
        f"{f'Bound {ci_pct}% lo':>16}  {f'Bound {ci_pct}% hi':>16}"
    )
    print(f"\n===== Calibration Results (n={n}, CI={confidence_interval}) =====")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in metric_keys:
        values = [run[key] for run in all_runs if key in run]
        if not values or key not in bounds:
            print(f"  {key:<{col_w}}  {'no data':>14}")
            continue
        arr = np.array(values, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        lo, hi = bounds[key]
        print(
            f"  {key:<{col_w}}  {mean:>14.6g}  {std:>12.4g}  {lo:>16.6g}  {hi:>16.6g}",
        )
    print("=" * len(header) + "\n")
