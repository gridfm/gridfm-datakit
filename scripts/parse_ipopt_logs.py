#!/usr/bin/env python3
"""
Parse IPOPT logs and report iteration statistics conditioned on termination status.

Usage:
    python parse_ipopt_logs.py file1.log file2.log ...

What it does:
- A single file may contain multiple IPOPT runs back-to-back.
- For each run, it extracts:
    * Number of Iterations....: <int>
    * EXIT: <termination message>
- A run is classified as "optimal" if the EXIT line contains
  the substring "Optimal Solution Found." (case-sensitive match).
  Any other EXIT message is classified as "non_optimal".
- Reports count, min, max, average, and 99th percentile iterations for both classes.
- Also reports bookkeeping about runs with incomplete data.
"""

import argparse
import re
import statistics
from typing import List, Optional, Dict
import numpy as np

ITER_RE = re.compile(r"^\s*Number of Iterations\.{4,}:\s*(\d+)\s*$")
EXIT_RE = re.compile(r"^\s*EXIT:\s*(.+?)\s*$")
RUN_START_RE = re.compile(r"^\s*This is Ipopt version\b")  # optional delimiter


class RunRecord:
    def __init__(self):
        self.iterations: Optional[int] = None
        self.exit_msg: Optional[str] = None

    def is_complete(self) -> bool:
        return self.iterations is not None and self.exit_msg is not None

    def is_optimal(self) -> Optional[bool]:
        if self.exit_msg is None:
            return None
        return "Optimal Solution Found." in self.exit_msg


def _finalize_run(current: RunRecord, runs: List[RunRecord]) -> None:
    """Append the current run if it has any content (iterations or exit)."""
    if current.iterations is not None or current.exit_msg is not None:
        runs.append(current)


def parse_ipopt_runs_from_text(lines: List[str]) -> List[RunRecord]:
    """Parse IPOPT runs from a sequence of log lines."""
    runs: List[RunRecord] = []
    current = RunRecord()

    for line in lines:
        if RUN_START_RE.search(line):
            _finalize_run(current, runs)
            current = RunRecord()
            continue

        m_iter = ITER_RE.match(line)
        if m_iter:
            current.iterations = int(m_iter.group(1))
            continue

        m_exit = EXIT_RE.match(line)
        if m_exit:
            current.exit_msg = m_exit.group(1).strip()
            continue

    _finalize_run(current, runs)
    return runs


def parse_files(filepaths: List[str]) -> List[RunRecord]:
    all_runs: List[RunRecord] = []
    for p in filepaths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        runs = parse_ipopt_runs_from_text(lines)
        all_runs.extend(runs)
    return all_runs


def summarize_iterations(
    runs: List[RunRecord],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Return summary statistics for optimal and non-optimal classes."""
    optimal_iters = [r.iterations for r in runs if r.is_complete() and r.is_optimal()]
    non_optimal_iters = [
        r.iterations for r in runs if r.is_complete() and not r.is_optimal()
    ]

    def stats(xs: List[int]) -> Dict[str, Optional[float]]:
        if not xs:
            return {"count": 0, "min": None, "max": None, "avg": None, "p99": None}
        return {
            "count": len(xs),
            "min": min(xs),
            "max": max(xs),
            "avg": statistics.mean(xs),
            "p99": float(np.percentile(xs, 99)),
            "999": float(np.percentile(xs, 99.9)),
        }

    return {
        "optimal": stats(optimal_iters),
        "non_optimal": stats(non_optimal_iters),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Extract IPOPT iteration statistics by termination status.",
    )
    ap.add_argument("logs", nargs="+", help="One or more IPOPT log files.")
    ap.add_argument(
        "--show-incomplete",
        action="store_true",
        help="List runs that are missing iterations and/or EXIT lines.",
    )
    args = ap.parse_args()

    runs = parse_files(args.logs)
    complete = [r for r in runs if r.is_complete()]
    incomplete = [r for r in runs if not r.is_complete()]
    summary = summarize_iterations(runs)

    print("====== IPOPT Iteration Statistics ======")
    print(f"Total runs detected: {len(runs)}")
    print(f"Complete runs:       {len(complete)}")
    print(f"Incomplete runs:     {len(incomplete)}")
    print()

    def fmt_stats(title: str, s: Dict[str, Optional[float]]) -> None:
        print(f"[{title}]")
        print(f"  count : {s['count']}")
        print(f"  min   : {s['min']}")
        print(f"  max   : {s['max']}")
        print(f"  avg   : {None if s['avg'] is None else round(s['avg'], 3)}")
        print(f"  p99   : {None if s['p99'] is None else round(s['p99'], 3)}")
        print(f"  p99.9 : {None if s['999'] is None else round(s['999'], 3)}")
        print()

    fmt_stats("Optimal solution", summary["optimal"])
    fmt_stats("No optimal solution", summary["non_optimal"])

    if args.show_incomplete and incomplete:
        print("---- Incomplete runs ----")
        for idx, r in enumerate(incomplete, 1):
            print(f"Run {idx}: iterations={r.iterations}, exit={r.exit_msg!r}")


if __name__ == "__main__":
    main()
