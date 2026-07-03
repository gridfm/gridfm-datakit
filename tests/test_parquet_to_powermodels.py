"""Parquet -> JSON -> Julia roundtrip tests for scenarios 3 and 4.

7 cases × 2 solvers (pf, opf) × 2 scenarios = 28 checks using
tests/fixtures/parquet_json_roundtrip/ (see scripts/extract_roundtrip_fixtures.py).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from gridfm_datakit.convert.roundtrip_check import (
    CASES,
    SOLVER_SPECS,
    configure_juliacall_env,
    define_julia_roundtrip_helpers,
    load_args_log,
    run_roundtrip,
)

FIXTURE_BASE = Path(__file__).resolve().parent / "fixtures" / "parquet_json_roundtrip"
ROUNDTRIP_SCENARIOS = (3, 4)
TOL = 1e-6
ATOL = 1e-8


def _fixture_raw_dir(case: str, dataset: str) -> Path:
    return FIXTURE_BASE / dataset / case / "raw"


def _planned_cases() -> list:
    planned = []
    for case in CASES:
        for solver, spec in SOLVER_SPECS.items():
            dataset = spec["dataset"]
            raw_dir = _fixture_raw_dir(case, dataset)
            if not raw_dir.is_dir():
                continue
            for scenario in ROUNDTRIP_SCENARIOS:
                planned.append(
                    pytest.param(
                        case,
                        dataset,
                        solver,
                        scenario,
                        str(raw_dir),
                        id=f"{dataset}/{case}/{solver}/s{scenario}",
                    ),
                )
    return planned


PLANNED = _planned_cases()


@pytest.fixture(scope="module")
def jl():
    if shutil.which("julia") is None:
        pytest.skip("julia not on PATH")
    sample_raw = _fixture_raw_dir("case14_ieee", "pf")
    if not sample_raw.is_dir():
        pytest.skip("roundtrip fixtures not found")
    configure_juliacall_env()
    from juliacall import Main as jl_main

    max_iter = int(load_args_log(str(sample_raw))["settings"]["max_iter"])
    define_julia_roundtrip_helpers(jl_main, max_iter, TOL)
    return jl_main


@pytest.mark.skipif(not PLANNED, reason="roundtrip fixtures not found")
@pytest.mark.parametrize("case,dataset,solver,scenario,raw_dir", PLANNED)
def test_parquet_json_roundtrip(jl, case, dataset, solver, scenario, raw_dir):
    max_iter = int(load_args_log(raw_dir)["settings"]["max_iter"])
    result = run_roundtrip(
        jl,
        case,
        dataset,
        solver,
        scenario,
        raw_dir,
        max_iter=max_iter,
        tol=TOL,
        atol=ATOL,
    )
    assert result.passed, result.error or ", ".join(result.failed_columns)
