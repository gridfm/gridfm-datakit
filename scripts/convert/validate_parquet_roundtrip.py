#!/usr/bin/env python3
"""Validate parquet -> JSON -> PowerModels roundtrip on random scenarios.

Discovers case datasets under ``{data-root}/pf`` and ``{data-root}/opf`` (default:
``/dccstor/gridfm/powermodels_data/v4/finetuning``), samples N scenarios per case,
re-solves, and compares rebuilt tables to stored parquet.

On success, writes ``0_VALIDATION_OK`` at each of ``pf/`` and ``opf/`` (same marker
name already used in the tree; content lists which scenarios passed). Exits
non-zero if any check fails and removes stale ``0_VALIDATION_OK`` markers.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd

from gridfm_datakit.convert.roundtrip_check import (
    configure_juliacall_env,
    define_julia_roundtrip_helpers,
    load_args_log,
    run_roundtrip,
)

DEFAULT_DATA_ROOT = Path("/dccstor/gridfm/powermodels_data/v4/finetuning")
MODES = ("pf", "opf")
MARKER_NAME = "0_VALIDATION_OK"
TOL = 1e-6
ATOL = 1e-8


def discover_cases(mode_dir: Path) -> List[Path]:
    if not mode_dir.is_dir():
        return []
    cases = []
    for path in sorted(mode_dir.iterdir()):
        if not path.is_dir():
            continue
        if (path / "raw" / "bus_data.parquet").exists():
            cases.append(path)
    return cases


def list_scenarios(raw_dir: Path) -> List[int]:
    """Return available scenario indices (prefer n_scenarios.txt when contiguous)."""
    n_path = raw_dir / "n_scenarios.txt"
    if n_path.is_file():
        n = int(n_path.read_text().strip())
        if n > 0:
            return list(range(n))

    runtime = raw_dir / "runtime_data.parquet"
    if not runtime.exists():
        raise FileNotFoundError(f"No scenario index source under {raw_dir}")
    df = pd.read_parquet(runtime, columns=["scenario"], engine="pyarrow")
    return sorted(int(s) for s in df["scenario"].unique())


def sample_scenarios(scenarios: Sequence[int], k: int, rng: random.Random) -> List[int]:
    if not scenarios:
        raise ValueError("no scenarios available")
    if k >= len(scenarios):
        return list(scenarios)
    return sorted(rng.sample(list(scenarios), k))


def format_marker(
    mode: str,
    data_root: Path,
    rows: Sequence[Tuple[str, int, bool, float, str]],
    seed: int,
    n_sample: int,
) -> str:
    passed = [r for r in rows if r[2]]
    failed = [r for r in rows if not r[2]]
    by_case: dict[str, list[int]] = {}
    for case, sc, ok, _, _ in rows:
        if ok:
            by_case.setdefault(case, []).append(sc)

    lines = [
        "parquet_json_roundtrip_validation_ok",
        f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}",
        f"data_root: {data_root}",
        f"mode: {mode}",
        f"solver: {mode}",
        f"n_sample_per_case: {n_sample}",
        f"seed: {seed}",
        f"atol: {ATOL}",
        f"tol: {TOL}",
        f"n_checks: {len(rows)}",
        f"n_passed: {len(passed)}",
        f"n_failed: {len(failed)}",
        "validated_scenarios:",
    ]
    for case in sorted(by_case):
        scs = ",".join(str(s) for s in sorted(by_case[case]))
        lines.append(f"  {case}: {scs}")
    if failed:
        lines.append("failures:")
        for case, sc, _, maxdiff, detail in failed:
            lines.append(f"  {case}/s{sc}: max_abs_diff={maxdiff:.6e} {detail}")
    lines.append("")
    return "\n".join(lines)


def write_marker(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def remove_marker(path: Path) -> None:
    if path.is_file():
        path.unlink()


def validate_mode(
    jl,
    mode: str,
    mode_dir: Path,
    n_sample: int,
    seed: int,
    cases_filter: Iterable[str] | None,
) -> Tuple[bool, List[Tuple[str, int, bool, float, str]]]:
    cases = discover_cases(mode_dir)
    if cases_filter is not None:
        allow = set(cases_filter)
        cases = [c for c in cases if c.name in allow]

    if not cases:
        print(f"[warn] no cases under {mode_dir}", flush=True)
        return True, []

    rng = random.Random(seed)
    rows: List[Tuple[str, int, bool, float, str]] = []

    for case_dir in cases:
        raw_dir = case_dir / "raw"
        case = case_dir.name
        scenarios = sample_scenarios(list_scenarios(raw_dir), n_sample, rng)
        max_iter = int(load_args_log(str(raw_dir))["settings"]["max_iter"])
        print(
            f"[{mode}] {case}: scenarios={scenarios} max_iter={max_iter}",
            flush=True,
        )
        for sc in scenarios:
            t0 = time.time()
            result = run_roundtrip(
                jl,
                case=case,
                dataset=mode,
                solver=mode,
                scenario=sc,
                raw_dir=str(raw_dir),
                max_iter=max_iter,
                tol=TOL,
                atol=ATOL,
            )
            dt = time.time() - t0
            detail = result.error or ",".join(result.failed_columns)
            status = "PASS" if result.passed else "FAIL"
            print(
                f"  {status} s{sc} max_abs_diff={result.max_abs_diff:.6e} "
                f"({dt:.1f}s) {detail}",
                flush=True,
            )
            rows.append((case, sc, result.passed, result.max_abs_diff, detail))

    ok = all(r[2] for r in rows) if rows else True
    return ok, rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing pf/ and opf/ (default: v4/finetuning)",
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=10,
        help="Random scenarios per case (default: 10)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(MODES),
        default=list(MODES),
        help="Subset of modes to validate (default: pf opf)",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Optional case name filter (default: all cases with raw parquet)",
    )
    args = parser.parse_args(argv)

    data_root: Path = args.data_root
    if not data_root.is_dir():
        print(f"data root not found: {data_root}", file=sys.stderr)
        return 2

    configure_juliacall_env()
    from juliacall import Main as jl

    # Julia helpers bake max_iter into _ipopt default string; runtime still passes
    # per-call max_iter. Use a placeholder from the first available args.log.
    sample_max_iter = 100
    for mode in args.modes:
        for case_dir in discover_cases(data_root / mode):
            sample_max_iter = int(
                load_args_log(str(case_dir / "raw"))["settings"]["max_iter"]
            )
            break
        else:
            continue
        break
    define_julia_roundtrip_helpers(jl, sample_max_iter, TOL)
    try:
        jl.seval(
            'using Memento; Memento.setlevel!(Memento.getlogger("PowerModels"), "error")'
        )
    except Exception:
        pass

    all_ok = True
    for mode in args.modes:
        mode_dir = data_root / mode
        marker = mode_dir / MARKER_NAME
        # Per-mode seed offset so pf/opf draws differ with the same --seed.
        mode_seed = args.seed + (0 if mode == "pf" else 1_000_003)
        ok, rows = validate_mode(
            jl,
            mode,
            mode_dir,
            n_sample=args.n_scenarios,
            seed=mode_seed,
            cases_filter=args.cases,
        )
        if ok and rows:
            text = format_marker(mode, data_root, rows, mode_seed, args.n_scenarios)
            write_marker(marker, text)
            print(f"wrote {marker}", flush=True)
        else:
            remove_marker(marker)
            if not ok:
                all_ok = False
                print(f"validation failed for {mode}; removed {marker}", flush=True)
            elif not rows:
                print(f"no checks run for {mode}", flush=True)

    if not all_ok:
        print("VALIDATION FAILED", flush=True)
        return 1
    print("VALIDATION OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
