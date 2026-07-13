#!/usr/bin/env python3
"""Run the full dynamic-simulation pipeline on the IEEE14 example case.

Loads config.yaml (paths relative to this folder), resolves them to absolute so
the example runs from any working directory, then calls the public entry point
``generate_dynamic_data``. Full logging is enabled: the pipeline's progress
logger streams chunk-by-chunk advancement to the console, OPF/PF/Dynawo native
output is captured under out/IEEE14/raw/solver_log/, and each simulation's Dynawo
report is saved under out/reports/.

Usage:
    python scripts/dynamic_example/run.py          # from the project root
    python run.py                                  # from this folder
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from gridfm_datakit.dynamic.generate_dynamic import generate_dynamic_data

HERE = Path(__file__).resolve().parent


def _abs(path: str) -> str:
    """Resolve a folder-relative path from config.yaml to an absolute path."""
    return str((HERE / path).resolve())


def _resolve_paths(cfg: dict) -> dict:
    """Rewrite the example's relative paths to absolute, in place."""
    cfg["network"]["file"] = _abs(cfg["network"]["file"])
    for key, val in cfg["dynamic"]["input_files"].items():
        cfg["dynamic"]["input_files"][key] = _abs(val)
    sp = cfg["dynamic"]["solver_parameters"]
    for key in ("parameters_file", "network_parameters_file", "solver_parameters_file"):
        sp[key] = _abs(sp[key])
    cfg["dynamic"]["output_dir"] = _abs(cfg["dynamic"]["output_dir"])
    cfg["settings"]["data_dir"] = _abs(cfg["settings"]["data_dir"])
    return cfg


def main() -> None:
    # Stream all pipeline logs (progress, warnings, errors) to the console.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = _resolve_paths(yaml.safe_load((HERE / "config.yaml").read_text()))
    file_paths = generate_dynamic_data(cfg)

    print("\nGenerated artifacts:")
    for key in (
        "bus_data",
        "gen_data",
        "branch_data",
        "y_bus_data",
        "runtime_data",
        "dynamic_results_zarr",
        "dynamic_reports_dir",
        "metadata_json",
    ):
        if key in file_paths:
            print(f"  {key:22s} {file_paths[key]}")


if __name__ == "__main__":
    main()
