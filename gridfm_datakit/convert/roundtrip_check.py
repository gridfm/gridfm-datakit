"""Parquet -> JSON -> Julia solver roundtrip checks (pytest fixtures)."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from matpowercaseframes import CaseFrames

from gridfm_datakit.convert.parquet_to_powermodels import parquet_to_json
from gridfm_datakit.network import Network
from gridfm_datakit.process.process_network import pf_post_processing
from gridfm_datakit.utils.column_names import (
    BRANCH_COLUMNS,
    BUS_COLUMNS,
    DC_BRANCH_COLUMNS,
    DC_BUS_COLUMNS,
    DC_GEN_COLUMNS,
    DC_RUNTIME_COLUMNS,
    GEN_COLUMNS,
    RUNTIME_COLUMNS,
    YBUS_COLUMNS,
)
from gridfm_datakit.utils.utils import n_scenario_per_partition

CASES = (
    "case14_ieee",
    "case30_ieee",
    "case57_ieee",
    "case118_ieee",
    "case500_goc",
    "case2000_goc",
    "case10000_goc",
)
SLOW_PF_CASES = frozenset({"case2000_goc", "case10000_goc"})
PARQUET_TABLES = ("bus", "gen", "branch", "y_bus", "runtime")
RUNTIME_TIMING_COLS = frozenset({"ac", "dc"})

TABLE_COLUMNS = {
    "bus": BUS_COLUMNS + DC_BUS_COLUMNS,
    "gen": GEN_COLUMNS + DC_GEN_COLUMNS,
    "branch": BRANCH_COLUMNS + DC_BRANCH_COLUMNS,
    "y_bus": YBUS_COLUMNS,
    "runtime": RUNTIME_COLUMNS + DC_RUNTIME_COLUMNS,
}
PF_DATA_KEYS = {
    "bus": "bus",
    "gen": "gen",
    "branch": "branch",
    "y_bus": "Y_bus",
    "runtime": "runtime",
}
SORT_COLUMNS = {
    "bus": ["bus"],
    "gen": ["idx"],
    "branch": ["idx"],
    "y_bus": ["index1", "index2"],
    "runtime": ["load_scenario_idx"],
}

SOLVER_SPECS = {
    "pf": {"dataset": "pf", "primary": "pf"},
    "opf": {"dataset": "opf", "primary": "opf"},
}


@dataclass
class RoundtripResult:
    case: str
    dataset: str
    solver: str
    scenario: int
    passed: bool
    error: str = ""
    columns_compared: int = 0
    max_abs_diff: float = 0.0
    failed_columns: List[str] = field(default_factory=list)
    column_diffs: Dict[str, float] = field(default_factory=dict)


def load_args_log(raw_dir: str) -> Dict[str, Any]:
    path = os.path.join(raw_dir, "args.log")
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    yaml_text = "".join(line for line in lines if not line.startswith("New generation"))
    return yaml.safe_load(yaml_text)


def configure_juliacall_env() -> None:
    julia_exe = shutil.which("julia")
    if julia_exe is None:
        raise RuntimeError("Could not find 'julia' on PATH")
    os.environ.setdefault("PYTHON_JULIACALL_EXE", julia_exe)
    project = os.environ.get("JULIA_PROJECT") or os.environ.get("PYTHON_JULIACALL_PROJECT")
    if not project:
        repo_julia_env = Path(__file__).resolve().parents[2] / "venv" / "julia_env"
        if repo_julia_env.is_dir():
            project = str(repo_julia_env)
    if project:
        os.environ.setdefault("JULIA_PROJECT", project)
        os.environ.setdefault("PYTHON_JULIACALL_PROJECT", project)


def define_julia_roundtrip_helpers(jl: Any, max_iter: int, tol: float) -> None:
    """Solver entrypoints aligned with benchmark_case118_dynamic_pf_sweep_juliacall.py."""
    jl.seval(
        f"""
        using PowerModels
        using Ipopt

        function _ipopt(max_iter, tol)
            optimizer_with_attributes(
                Ipopt.Optimizer,
                "tol" => {tol},
                "print_level" => 0,
                "max_iter" => max_iter,
            )
        end

        function solve_pf_fast(network)
            result = compute_ac_pf(network)
            result["termination_status"] == false && error("PF failed")
            update_data!(network, result["solution"])
            flows = calc_branch_flow_ac(network)
            result["solution"]["branch"] = flows["branch"]
            result["solution"]["pf"] = true
            return result
        end

        function solve_pf_ipopt(network, max_iter, tol)
            result = solve_ac_pf(network, _ipopt(max_iter, tol))
            string(result["termination_status"]) != "LOCALLY_SOLVED" &&
                error("PF Ipopt failed with status $(result["termination_status"])")
            update_data!(network, result["solution"])
            flows = calc_branch_flow_ac(network)
            result["solution"]["branch"] = flows["branch"]
            result["solution"]["pf"] = true
            return result
        end

        function solve_dcpf_fast(network)
            result = compute_dc_pf(network)
            result["termination_status"] == false && error("DC PF failed")
            update_data!(network, result["solution"])
            flows = calc_branch_flow_dc(network)
            result["solution"]["branch"] = flows["branch"]
            result["solution"]["pf"] = true
            return result
        end

        function solve_opf(network, max_iter, tol)
            result = solve_ac_opf(network, _ipopt(max_iter, tol))
            string(result["termination_status"]) != "LOCALLY_SOLVED" &&
                error("OPF failed with status $(result["termination_status"])")
            result["solution"]["pf"] = false
            return result
        end

        function solve_dcopf(network, max_iter, tol)
            result = solve_dc_opf(network, _ipopt(max_iter, tol))
            string(result["termination_status"]) != "LOCALLY_SOLVED" &&
                error("DC OPF failed with status $(result["termination_status"])")
            result["solution"]["pf"] = false
            return result
        end
        """,
    )


def pf_fast_for_case(case: str) -> bool:
    """Fast AC PF (``compute_ac_pf``) except large GOC cases (``solve_ac_pf``/Ipopt)."""
    return case not in SLOW_PF_CASES


def load_scenario_parquet(data_dir: str, scenario: int) -> Dict[str, pd.DataFrame]:
    partition = scenario // n_scenario_per_partition
    filters = [("scenario_partition", "=", partition)]
    tables: Dict[str, pd.DataFrame] = {}
    for name in PARQUET_TABLES:
        df = pd.read_parquet(
            os.path.join(data_dir, f"{name}_data.parquet"),
            filters=filters,
            engine="pyarrow",
        )
        tables[name] = (
            df.loc[df["scenario"] == scenario]
            .drop(columns=["scenario", "scenario_partition"])
            .sort_values(SORT_COLUMNS[name], kind="stable")
            .reset_index(drop=True)
        )
    return tables


def network_from_json(json_path: str, jl: Any) -> Network:
    with tempfile.TemporaryDirectory() as tmp:
        m_path = os.path.join(tmp, "case.m")
        jl.PowerModels.export_matpower(m_path, jl.PowerModels.parse_file(json_path))
        frames = CaseFrames(m_path)
        mpc = {
            key: frames.__getattribute__(key)
            if not isinstance(frames.__getattribute__(key), pd.DataFrame)
            else frames.__getattribute__(key).values
            for key in frames._attributes
        }
        return Network(mpc)


def compare_tables(
    original: Dict[str, pd.DataFrame],
    rebuilt: Dict[str, pd.DataFrame],
    atol: float = 1e-8,
) -> Tuple[bool, Dict[str, float], List[str]]:
    failed: List[str] = []
    diffs: Dict[str, float] = {}

    for table in PARQUET_TABLES:
        cols = [c for c in TABLE_COLUMNS[table] if c not in RUNTIME_TIMING_COLS]
        if not cols:
            continue
        for col in cols:
            key = f"{table}.{col}"
            a = original[table][col].to_numpy()
            b = rebuilt[table][col].to_numpy()
            if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
                diff = float(np.nanmax(np.abs(a.astype(float) - b.astype(float))))
            else:
                diff = 0.0 if np.array_equal(a, b) else float("inf")
            diffs[key] = diff
            if not np.allclose(a, b, rtol=0.0, atol=atol, equal_nan=True):
                failed.append(key)

        if table == "runtime":
            for col in RUNTIME_TIMING_COLS:
                if col in rebuilt[table].columns:
                    val = float(rebuilt[table][col].iloc[0])
                    if val <= 0.0:
                        failed.append(f"{table}.{col}_nonpositive")

    return len(failed) == 0, diffs, failed


def run_roundtrip(
    jl: Any,
    case: str,
    dataset: str,
    solver: str,
    scenario: int,
    raw_dir: str,
    max_iter: int,
    tol: float = 1e-6,
    atol: float = 1e-8,
) -> RoundtripResult:
    spec = SOLVER_SPECS[solver]
    result = RoundtripResult(
        case=case,
        dataset=dataset,
        solver=solver,
        scenario=scenario,
        passed=False,
    )

    try:
        original = load_scenario_parquet(raw_dir, scenario)
        load_scenario_idx = int(original["bus"]["load_scenario_idx"].iloc[0])

        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "case.json")
            parquet_to_json(raw_dir, scenario, json_path)

            network = jl.PowerModels.parse_file(json_path)
            dc_network = jl.PowerModels.parse_file(json_path)

            if spec["primary"] == "pf":
                if pf_fast_for_case(case):
                    ac_res = jl.solve_pf_fast(network)
                else:
                    ac_res = jl.solve_pf_ipopt(network, max_iter, tol)
                dc_res = jl.solve_dcpf_fast(dc_network)
            else:
                ac_res = jl.solve_opf(network, max_iter, tol)
                dc_res = jl.solve_dcopf(dc_network, max_iter, tol)

            net = network_from_json(json_path, jl)
            pf_data = pf_post_processing(
                load_scenario_idx,
                net,
                ac_res,
                dc_res,
                include_dc_res=True,
            )

            rebuilt = {
                name: pd.DataFrame(
                    pf_data[PF_DATA_KEYS[name]],
                    columns=TABLE_COLUMNS[name],
                )
                .sort_values(SORT_COLUMNS[name], kind="stable")
                .reset_index(drop=True)
                for name in PARQUET_TABLES
            }

            passed, diffs, failed = compare_tables(original, rebuilt, atol=atol)
            result.passed = passed
            result.column_diffs = diffs
            result.failed_columns = failed
            result.columns_compared = len(diffs)
            if diffs:
                result.max_abs_diff = max(diffs.values())
    except Exception as exc:
        result.error = str(exc)

    return result
