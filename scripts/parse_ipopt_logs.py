#!/usr/bin/env python3
"""
Ultra-fast IPOPT log parser:
- Streams files (no readlines)
- No regex (string operations)
- __slots__ RunRecord
- Vectorized percentiles
"""

from pathlib import Path
import argparse
from typing import List, Optional, Dict
import numpy as np
from tqdm import tqdm
import mmap
from multiprocessing import Pool, cpu_count


class RunRecord:
    __slots__ = ("iterations", "exit_msg")

    def __init__(self):
        self.iterations: Optional[int] = None
        self.exit_msg: Optional[str] = None

    def is_complete(self) -> bool:
        return self.iterations is not None and self.exit_msg is not None

    def is_optimal(self) -> Optional[bool]:
        if self.exit_msg is None:
            return None
        return "Optimal Solution Found." in self.exit_msg


def _finalize_run(current: RunRecord, runs: List[RunRecord]):
    if current.iterations is not None or current.exit_msg is not None:
        runs.append(current)


def parse_ipopt_runs_from_file(path: str) -> List[RunRecord]:
    runs = []
    current = RunRecord()

    _finalize = _finalize_run
    RunRec = RunRecord

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            readline = mm.readline

            while True:
                bline = readline()
                if not bline:
                    break

                line = bline.decode("utf-8", "ignore")

                if "This is Ipopt version" in line:
                    _finalize(current, runs)
                    current = RunRec()
                    continue

                if "Number of Iterations" in line:
                    idx = line.rfind(":")
                    if idx != -1:
                        num = line[idx + 1 :].strip()
                        if num.isdigit():
                            current.iterations = int(num)
                    continue

                if line.lstrip().startswith("EXIT:"):
                    current.exit_msg = line.split("EXIT:", 1)[1].strip()
                    continue

    _finalize(current, runs)
    return runs


def parse_files(filepaths: List[str]) -> List[RunRecord]:
    all_runs = []

    with Pool(processes=cpu_count()) as pool:
        for runs in tqdm(
            pool.imap_unordered(parse_ipopt_runs_from_file, filepaths),
            total=len(filepaths),
            desc="Parsing log files (parallel)",
        ):
            all_runs.extend(runs)

    return all_runs


def summarize_iterations(
    runs: List[RunRecord],
) -> Dict[str, Dict[str, Optional[float]]]:
    optimal_iters = [r.iterations for r in runs if r.is_complete() and r.is_optimal()]
    non_optimal_iters = [
        r.iterations for r in runs if r.is_complete() and not r.is_optimal()
    ]

    def stats(xs: List[int]):
        if not xs:
            return {"count": 0, "min": None, "max": None, "avg": None, "p99": None}

        arr = np.array(xs, dtype=float)
        p = np.percentile(arr, [99, 99.9, 99.99, 99.999, 99.99999])

        return {
            "count": len(xs),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "avg": float(arr.mean()),
            "p99": float(p[0]),
            "999": float(p[1]),
            "9999": float(p[2]),
            "99999": float(p[3]),
            "1/100000": float(p[4]),
        }

    return {"optimal": stats(optimal_iters), "non_optimal": stats(non_optimal_iters)}


def main():
    ap = argparse.ArgumentParser(
        description="Ultra-fast IPOPT iteration statistics parser.",
    )
    ap.add_argument(
        "--dir",
        required=True,
        help="Base directory that contains solver_log/",
    )
    ap.add_argument(
        "--type",
        required=True,
        choices=["opf", "dcopf", "pf", "dcpf"],
        help="Type of logs to parse.",
    )
    ap.add_argument(
        "--show-incomplete",
        action="store_true",
        help="List runs that are missing iterations and/or EXIT lines.",
    )
    args = ap.parse_args()

    patterns = {
        "opf": "opf_*.log",
        "dcopf": "dcopf_*.log",
        "pf": "pf_*.log",
        "dcpf": "dcpf_*.log",
    }

    pattern = patterns[args.type]
    base = Path(args.dir).expanduser().resolve()
    log_files = list((base / "solver_log").glob(pattern))

    if not log_files:
        print(f"No logs found for pattern solver_log/{pattern}")
        return

    print(f"Found {len(log_files)} log files for pattern solver_log/{pattern}")

    runs = parse_files([str(p) for p in log_files])
    complete = [r for r in runs if r.is_complete()]
    incomplete = [r for r in runs if not r.is_complete()]
    summary = summarize_iterations(runs)

    print("====== IPOPT Iteration Statistics ======")
    print(f"Total runs detected: {len(runs)}")
    print(f"Complete runs:       {len(complete)}")
    print(f"Incomplete runs:     {len(incomplete)}\n")

    def fmt_stats(title, s):
        print(f"[{title}]")
        for key in [
            "count",
            "min",
            "max",
            "avg",
            "p99",
            "999",
            "9999",
            "99999",
            "1/100000",
        ]:
            val = s.get(key)
            if isinstance(val, float):
                val = round(val, 3)
            print(f"  {key:10}: {val}")
        print()

    fmt_stats("Optimal solution", summary["optimal"])
    fmt_stats("No optimal solution", summary["non_optimal"])

    if args.show_incomplete and incomplete:
        print("---- Incomplete runs ----")
        for idx, r in enumerate(incomplete, 1):
            print(f"Run {idx}: iterations={r.iterations}, exit={r.exit_msg!r}")


if __name__ == "__main__":
    main()
