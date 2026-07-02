#!/usr/bin/env python3
"""Run parquet -> JSON -> Julia solver roundtrip checks across finetuning datasets."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from gridfm_datakit.convert.roundtrip_check import (
    CASES,
    SCENARIOS,
    SOLVER_SPECS,
    SLOW_PF_CASES,
    RoundtripResult,
    configure_juliacall_env,
    coverage_report_markdown,
    define_julia_roundtrip_helpers,
    iter_test_cases,
    load_args_log,
    pf_fast_for_case,
    run_roundtrip,
    summary_markdown,
)

DEFAULT_OUT = (
    "/u/apu/gridfm-datakit/scripts/runtime/parquet_to_powermodels/outputs"
    "/parquet_json_roundtrip"
)


def _write_results_csv(path: Path, results: list[RoundtripResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "dataset",
                "solver",
                "scenario",
                "passed",
                "error",
                "columns_compared",
                "max_abs_diff",
                "failed_columns",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "case": r.case,
                    "dataset": r.dataset,
                    "solver": r.solver,
                    "scenario": r.scenario,
                    "passed": int(r.passed),
                    "error": r.error,
                    "columns_compared": r.columns_compared,
                    "max_abs_diff": f"{r.max_abs_diff:.3e}",
                    "failed_columns": ";".join(r.failed_columns),
                },
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUT,
        help=f"Directory for CSV and markdown reports (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=list(CASES),
        help="Cases to test (default: all 7)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        type=int,
        default=list(SCENARIOS),
        help="Scenario indices (default: 0 1)",
    )
    parser.add_argument(
        "--solvers",
        nargs="*",
        choices=list(SOLVER_SPECS),
        default=list(SOLVER_SPECS),
        help="Solvers to test (default: pf and opf)",
    )
    parser.add_argument("--max-iter", type=int, default=None, help="Override Ipopt max_iter")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-9, help="Numerical tolerance")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    planned = list(iter_test_cases(args.cases, args.scenarios, args.solvers))
    print(f"Planned checks: {len(planned)} (= cases × solvers × scenarios)")
    print(
        "Solvers: pf (AC PF + DCPF) on pf data, opf (AC OPF + DCOPF) on opf data; "
        f"AC PF fast except {sorted(SLOW_PF_CASES)} (Ipopt)",
    )

    configure_juliacall_env()
    from juliacall import Main as jl

    results: list[RoundtripResult] = []
    started = time.perf_counter()

    for i, (case, dataset, solver, scenario, raw_dir) in enumerate(planned, start=1):
        cfg = load_args_log(raw_dir)
        max_iter = args.max_iter or int(cfg["settings"]["max_iter"])
        pf_mode = "fast" if pf_fast_for_case(case) else "ipopt"

        if i == 1:
            define_julia_roundtrip_helpers(jl, max_iter, args.tol)

        label = f"[{i}/{len(planned)}] {dataset}/{case} {solver} scenario={scenario} pf={pf_mode}"
        print(f"{label} ...", flush=True)
        t0 = time.perf_counter()
        result = run_roundtrip(
            jl,
            case,
            dataset,
            solver,
            scenario,
            raw_dir,
            max_iter=max_iter,
            tol=args.tol,
            atol=args.atol,
        )
        elapsed = time.perf_counter() - t0
        status = "PASS" if result.passed else "FAIL"
        print(f"  {status} ({elapsed:.1f}s)", flush=True)
        if not result.passed and result.error:
            print(f"  error: {result.error}", flush=True)
        results.append(result)

    _write_results_csv(out_dir / "roundtrip_results.csv", results)
    (out_dir / "roundtrip_summary.md").write_text(summary_markdown(results), encoding="utf-8")
    (out_dir / "coverage_report.md").write_text(coverage_report_markdown(), encoding="utf-8")

    passed = sum(r.passed for r in results)
    total = len(results)
    wall = time.perf_counter() - started
    print(f"\nDone: {passed}/{total} passed in {wall:.1f}s")
    print(f"Results: {out_dir / 'roundtrip_results.csv'}")
    print(f"Summary: {out_dir / 'roundtrip_summary.md'}")
    print(f"Coverage: {out_dir / 'coverage_report.md'}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
