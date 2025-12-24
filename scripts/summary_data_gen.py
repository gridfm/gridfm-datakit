#!/usr/bin/env python3
import re
import pandas as pd
import yaml
from pathlib import Path

# Import your parser
from parse_ipopt_logs import parse_files, summarize_iterations


###############################################################################
# Helper functions
###############################################################################


def read_args_log(path: Path):
    lines = path.read_text().splitlines()

    # Drop the non-YAML timestamp line
    yaml_text = "\n".join(lines[2:])

    data = yaml.safe_load(yaml_text)

    return {
        "scenarios": data.get("load", {}).get("scenarios"),
        "n_topology_variants": data.get("topology_perturbation", {}).get(
            "n_topology_variants",
        ),
        "large_chunk_size": data.get("settings", {}).get("large_chunk_size"),
        "pf_fast": data.get("settings", {}).get("pf_fast"),
        "cpu": data.get("settings", {}).get("num_processes"),
        "k": data.get("topology_perturbation", {}).get("k"),
        "mode": data.get("settings", {}).get("mode"),
        "max_iter": data.get("settings", {}).get("max_iter"),
    }


def read_samples(raw_dir: Path):
    f = raw_dir / "n_scenarios.txt"
    if f.exists():
        try:
            return int(f.read_text().strip())
        except Exception:
            pass

    parquet = raw_dir / "runtime_data.parquet"
    df = pd.read_parquet(parquet)
    if "scenario" not in df.columns:
        raise ValueError(f"No scenario column in {parquet}")
    return int(df["scenario"].max())


def read_last_tqdm_time(raw_dir: Path):
    path = raw_dir / "tqdm.log"
    if not path.exists():
        return None

    lines = path.read_text().strip().splitlines()
    if not lines:
        return None

    last = lines[-1]

    # extract the first bracketed time, stopping before '<'
    m = re.search(r"\[([0-9:]+)<", last)
    if not m:
        return None

    t = m.group(1)
    parts = t.split(":")

    # Support formats:
    #   MM:SS
    #   H:MM:SS
    if len(parts) == 2:
        mm, ss = parts
        hh = 0
    elif len(parts) == 3:
        hh, mm, ss = parts
    else:
        return None

    try:
        hh = int(hh)
        mm = int(mm)
        ss = int(ss)
    except ValueError:
        return None

    return hh * 3600 + mm * 60 + ss


###############################################################################
# IPOPT maximum iteration extraction using YOUR PARSER
###############################################################################


def compute_max_iter(solver_log_dir: Path, mode: str):
    patterns = {
        "opf": "opf_*.log",
        "dcopf": "dcopf_*.log",
        "pf": "pf_*.log",
        "dcpf": "dcpf_*.log",
    }
    pattern = patterns.get(mode)
    if pattern is None:
        return None

    log_files = list((solver_log_dir).glob(pattern))
    if not log_files:
        return None

    # Call your parser exactly as designed
    runs = parse_files([str(p) for p in log_files])
    summary = summarize_iterations(runs)

    # The request: maximum iterations among convergent cases
    opt = summary["optimal"]["max"]
    # non = summary["non_optimal"]["max"]

    # Choose the maximum across all complete runs
    # candidates = [v for v in [opt, non] if v is not None]
    return opt


###############################################################################
# Per-grid extraction
###############################################################################


def process_grid(grid_root: Path):
    raw_dir = grid_root / "raw"

    args_log = raw_dir / "args.log"
    if not args_log.exists():
        raise FileNotFoundError(f"Missing args.log at {args_log}")

    args = read_args_log(args_log)

    samples = read_samples(raw_dir)
    wall_seconds = read_last_tqdm_time(raw_dir)
    max_iter_reached = compute_max_iter(raw_dir / "solver_log", "opf")

    scenarios = args["scenarios"]
    n_top = args["n_topology_variants"]
    cpu = args["cpu"]

    conv_rate = None
    if scenarios and n_top:
        conv_rate = samples / (scenarios * n_top)

    cpu_hours = None
    if cpu and wall_seconds:
        cpu_hours = cpu * (wall_seconds / 3600)

    return {
        "grid": grid_root.name,
        "type": args["mode"],
        "samples": samples,
        "scenarios": scenarios,
        "n_topology_variants": n_top,
        "conv_rate": conv_rate,
        "large_chunk_size": args["large_chunk_size"],
        "pf_fast": args["pf_fast"],
        "cpu": cpu,
        "k": args["k"],
        "time_seconds": wall_seconds,
        "cpu_hours": cpu_hours,
        "max_iter_reached": max_iter_reached,
        "max_iter": args["max_iter"],
    }


###############################################################################
# Directory traversal
###############################################################################


def main(base_dir="/dccstor/gridfm/powermodels_data/v3/finetuning"):
    base = Path(base_dir)
    rows = []

    for t in ["pf", "opf"]:
        tdir = base / t
        if not tdir.exists():
            continue
        for grid in sorted(tdir.iterdir()):
            print(f"Processing {grid}...")
            if not grid.is_dir():
                continue
            try:
                row = process_grid(grid)
                rows.append(row)
            except Exception as e:
                print(f"Error in {grid}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(base / "summary.csv", index=False)
    print(f"Wrote {base / 'summary.csv'}")


if __name__ == "__main__":
    main()
