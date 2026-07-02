"""
Build PFDelta-style task splits from local parquet datasets in `data_processed/`.

This follows the same split logic as `PFDeltaDataset.shuffle_split_and_save_data()`
in `core/datasets/pfdelta_dataset.py`, but instead of listing JSON filenames it:
  - filters parquet datasets by scenario keys, then
  - writes the filtered parquet outputs under:
      data_tasks/<case>/task_<task>/<split>/<grid_type>/<regime>/

Regimes:
  - feasible: filter by `scenario` (0-based). PFDelta indices are 1-based, so we subtract 1.
  - near infeasible (nose): filter by `pfdelta_scenario`
  - approaching infeasible (around_nose): filter by (`pfdelta_scenario`, `lam`)

Example:
  /Users/apu/pfdelta/venv/bin/python scripts/build_task_splits_from_data_processed.py \
    --case case14 --tasks 1.3 4.2
"""

from __future__ import annotations
import numpy as np

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd


TASK_CONFIG: Dict[float, Dict[str, Dict[str, int]]] =  {  # values here will have the number of train samples
            1.1: {
                "feasible": {"n": 54000, "n-1": 0, "n-2": 0},
                "near infeasible": {"n": 0, "n-1": 0, "n-2": 0},
            },
            1.2: {
                "feasible": {"n": 27000, "n-1": 27000, "n-2": 0},
                "near infeasible": {"n": 0, "n-1": 0, "n-2": 0},
            },
            1.3: {
                "feasible": {"n": 18000, "n-1": 18000, "n-2": 18000},
                "near infeasible": {"n": 0, "n-1": 0, "n-2": 0},
            },
            2.1: {
                "feasible": {"n": 18000, "n-1": 18000, "n-2": 18000},
                "near infeasible": {"n": 0, "n-1": 0, "n-2": 0},
            },
            2.2: {
                "feasible": {"n": 12000, "n-1": 12000, "n-2": 12000},
                "near infeasible": {"n": 0, "n-1": 0, "n-2": 0},
            },
            2.3: {
                "feasible": {"n": 6000, "n-1": 6000, "n-2": 6000},
                "near infeasible": {"n": 0, "n-1": 0, "n-2": 0},
            },
            3.1: {
                "feasible": {"n": 18000, "n-1": 18000, "n-2": 18000},
                "near infeasible": {"n": 0, "n-1": 0, "n-2": 0},
            },
            3.2: {
                "feasible": {"n": 6000, "n-1": 6000, "n-2": 6000},
                "near infeasible": {"n": 0, "n-1": 0, "n-2": 0},
            },
            3.3: {
                "feasible": {"n": 6000, "n-1": 6000, "n-2": 6000},
                "near infeasible": {"n": 0, "n-1": 0, "n-2": 0},
            },
            4.1: {
                "near infeasible": {"n": 1800, "n-1": 1800, "n-2": 1800},
                "feasible": {"n": 16200, "n-1": 16200, "n-2": 16200},
            },
            4.2: {
                "approaching infeasible": {"n": 7200, "n-1": 7200, "n-2": 7200},
                "near infeasible": {"n": 1800, "n-1": 1800, "n-2": 1800},
                "feasible": {"n": 9000, "n-1": 9000, "n-2": 9000},
            },
            4.3: {"near infeasible": {"n": 3600, "n-1": 3600, "n-2": 3600},
                  "approaching infeasible": {"n": 14400, "n-1": 14400, "n-2": 14400},
                  "feasible": {"n": 0, "n-1": 0, "n-2": 0}
            },
            "analysis": {
                "feasible": {"n": 56000, "n-1": 29000, "n-2": 20000},
                "near infeasible": {"n": 2000, "n-1": 2000, "n-2": 2000},
                "approaching infeasible": {"n": 7200, "n-1": 7200, "n-2": 7200},
            },
        }

TEST_CONFIG: Dict[str, Optional[Dict[str, int]]] = {
    "feasible": {"n": 2000, "n-1": 2000, "n-2": 2000},
    "approaching infeasible": None,  # no test split in PFDelta
    "near infeasible": {"n": 200, "n-1": 200, "n-2": 200},
}

GRID_TYPES = ("n", "n-1", "n-2")
ALL_CASE_NAMES = ("case57", "case118", "case500") 

# Matches PFDeltaDataset.task_split_config behavior for tasks 3.x
TASK_SPLIT_CONFIG: Dict[float, Dict[str, Sequence[str]]] = {
    3.1: {
        # train is selected via CLI --case (like PFDeltaDataset.case_name)
        "train": (),
        "valid": (),
        "test": ALL_CASE_NAMES,
    },
    3.2: {
        "train": ("case14", "case30", "case57"),
        "valid": ("case14", "case30", "case57"),
        "test": ("case118", "case500", "case2000"),
    },
    3.3: {
        "train": ("case118", "case500", "case2000"),
        "valid": ("case118", "case500", "case2000"),
        "test": ("case14", "case30", "case57"),
    },
}



def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_shuffle_map(shuffle_json_path: str) -> Dict[int, int]:
    with open(shuffle_json_path, "r") as f:
        shuffle_dict = json.load(f)
    # Keep insertion order from JSON (matches PFDeltaDataset).
    return {int(k): int(v) for k, v in shuffle_dict.items()}


def _feasible_pfdelta_split_scenarios_0based(
    *,
    shuffle_map: Dict[int, int],
    train_size: int,
    test_size: int,
) -> Dict[str, List[int]]:
    # Mirrors PFDeltaDataset.shuffle_split_and_save_data() feasible branch.
    keys = list(shuffle_map.keys())  # insertion order
    scenarios_shuffled_pfdelta = [shuffle_map[i] for i in keys] # the way they did it....
    if not np.all([int(i) == int(v) for i, v in zip(scenarios_shuffled_pfdelta, shuffle_map.values())]):
        raise ValueError("scenarios_shuffled_pfdelta != shuffle_map.values()")

    entire_size = len(scenarios_shuffled_pfdelta)
    if (train_size + test_size) > entire_size:
        raise ValueError(
            f"train_size({train_size})+test_size({test_size}) > entire_size({entire_size})"
        )

    train_pf = scenarios_shuffled_pfdelta[: int(0.9 * train_size)]
    val_pf = scenarios_shuffled_pfdelta[int(0.9 * train_size) : int(train_size)]
    test_pf = scenarios_shuffled_pfdelta[-int(test_size) :] if test_size else []

    return {
        "train": train_pf,
        "valid": val_pf,
        "test": test_pf,
    }


def _list_parquet_tables(raw_dir: str) -> List[str]:
    # Return full paths to parquet tables (files or directories ending with .parquet).
    entries = []
    for name in sorted(os.listdir(raw_dir)):
        if not name.endswith(".parquet"):
            continue
        entries.append(os.path.join(raw_dir, name))
    return entries


def _table_names_from_dirs(raw_dirs: Sequence[str]) -> List[str]:
    names: Set[str] = set()
    for d in raw_dirs:
        if not d or not os.path.exists(d):
            continue
        for p in _list_parquet_tables(d):
            names.add(os.path.basename(p))
    return sorted(names)


def _read_and_filter_table(paths: Sequence[str], scenario_set: Set[int]) -> pd.DataFrame:
    # `paths` may contain 1 (usual) or 2 (feasible train+test) parquet sources.
    dfs = [pd.read_parquet(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    return df[df["scenario"].isin(scenario_set)]


def _write_split_tables(
    *,
    table_names: Sequence[str],
    raw_dirs: Sequence[str],
    out_dir: str,
    scenarios: Sequence[int],
    log_prefix: str,
) -> None:
    if not scenarios:
        return

    _ensure_dir(out_dir)

    # PFDelta-consistent ordering
    scenario_order = [int(s) for s in scenarios]
    scenario_set = set(scenario_order)
    rank = {s: i for i, s in enumerate(scenario_order)}

    for tname in table_names:
        src_paths: List[str] = []
        for rd in raw_dirs:
            p = os.path.join(rd, tname)
            if os.path.exists(p):
                src_paths.append(p)
        if not src_paths:
            continue

        df = _read_and_filter_table(src_paths, scenario_set)

        if len(df) and "scenario" in df.columns:
            df = df.copy()
            df["_pfdelta_rank"] = df["scenario"].astype(int).map(rank)

            if df["_pfdelta_rank"].isna().any():
                missing = df.loc[df["_pfdelta_rank"].isna(), "scenario"].head(5).tolist()
                raise RuntimeError(
                    f"Internal error: unexpected scenarios in {tname}: {missing}"
                )

            # Stable sort = exact PFDelta order
            df = (
                df.sort_values("_pfdelta_rank", kind="mergesort")
                  .drop(columns=["_pfdelta_rank"])
            )

        out_file = os.path.join(out_dir, tname)
        rows = _write_filtered_parquet_pandas(df=df, out_file=out_file)
        print(f"{log_prefix}: wrote {rows} rows -> {out_file}")



def _write_filtered_parquet_pandas(*, df: pd.DataFrame, out_file: str) -> int:
    _ensure_dir(os.path.dirname(out_file))
    df.to_parquet(out_file, index=False)
    return int(len(df))


def _source_raw_dir(data_processed: str, case: str, grid_type: str, suffix: str) -> str:
    # Example: case14_ieee_n_feasible_train/raw
    case_type = "goc" if case == "case500" or case == "case2000" else "ieee"
    return os.path.join(data_processed, f"{case}_{case_type}_{grid_type}_{suffix}", "raw")

def _missing_paths(*paths: str) -> List[str]:
    return [p for p in paths if not os.path.exists(p)]


def _scenarios_for_pfdelta_scenarios(key_df: pd.DataFrame, pfdelta_vals: Sequence[int]) -> List[int]:
    """
    Return `scenario` ids in the SAME order as `pfdelta_vals`.

    This is important because PFDelta's split logic for nose/around_nose is based on
    filename ordering; to match it, downstream scenario lists must preserve that order.
    """
    if not pfdelta_vals:
        return []
    df = key_df[["pfdelta_scenario", "scenario"]].dropna().copy()
    df["pfdelta_scenario"] = df["pfdelta_scenario"].astype(int)
    df["scenario"] = df["scenario"].astype(int)
    mapping = df.drop_duplicates(subset=["pfdelta_scenario"]).set_index("pfdelta_scenario")["scenario"].to_dict()
    return [mapping[int(pid)] for pid in pfdelta_vals]


def _compute_splits_feasible(
    *,
    shuffle_files: str,
    grid_type: str,
    train_size: int,
    test_size: int,
) -> Dict[str, List[int]]:
    shuffle_path = os.path.join(shuffle_files, grid_type, "raw_shuffle.json")
    shuffle_map = _load_shuffle_map(shuffle_path)
    return _feasible_pfdelta_split_scenarios_0based(
        shuffle_map=shuffle_map, train_size=train_size, test_size=test_size
    )


def _compute_splits_nose(
    *,
    train_raw: str,
    test_raw: str,
    train_size: int,
    test_size: int,
) -> Dict[str, List[int]]:
    key_train = pd.read_parquet(
        os.path.join(train_raw, "bus_data.parquet"), columns=["scenario", "pfdelta_scenario"]
    )
    key_test = pd.read_parquet(
        os.path.join(test_raw, "bus_data.parquet"), columns=["scenario", "pfdelta_scenario"]
    )

    # Match PFDeltaDataset exactly: `sorted(os.listdir(...))` over filenames like
    # `sample_<ID>_nose.json` (lexicographic on the full filename string).
    train_ids = key_train["pfdelta_scenario"].dropna().astype(int).unique().tolist()
    test_ids = key_test["pfdelta_scenario"].dropna().astype(int).unique().tolist()

    train_keys = sorted([f"sample_{int(i)}_nose" for i in train_ids])
    test_keys = sorted([f"sample_{int(i)}_nose" for i in test_ids])

    # Extract back the pfdelta_scenario ids in that order, then take the first N.
    train_pfdelta = [int(k.split("_")[1]) for k in train_keys[:train_size]] if train_size else []
    test_pfdelta = [int(k.split("_")[1]) for k in test_keys[:test_size]] if test_size else []

    split_idx = int(0.9 * len(train_pfdelta))
    pf_splits = {
        "train": train_pfdelta[:split_idx],
        "valid": train_pfdelta[split_idx:],
        "test": test_pfdelta,
    }

    return {
        "train": _scenarios_for_pfdelta_scenarios(key_train, pf_splits["train"]),
        "valid": _scenarios_for_pfdelta_scenarios(key_train, pf_splits["valid"]),
        "test": _scenarios_for_pfdelta_scenarios(key_test, pf_splits["test"]),
    }


def _compute_splits_around_nose(
    *,
    train_raw: str,
    train_size: int,
) -> Dict[str, List[int]]:
    key_df = pd.read_parquet(
        os.path.join(train_raw, "bus_data.parquet"),
        columns=["scenario", "pfdelta_scenario", "lam"],
    ).dropna(subset=["pfdelta_scenario", "lam", "scenario"])

    def _lam_to_lamstr(lam: float) -> str:
        return f"{lam:.5f}".replace(".", "p")

    # One row per PFDelta sample, including scenario
    pairs_df = (
        key_df[["scenario", "pfdelta_scenario", "lam"]]
        .drop_duplicates()
        .assign(
            scenario=lambda d: d["scenario"].astype(int),
            pfdelta_scenario=lambda d: d["pfdelta_scenario"].astype(int),
            lam_float=lambda d: d["lam"].astype(float),
        )
    )
    pairs_df["lam_str"] = pairs_df["lam_float"].map(_lam_to_lamstr)
    pairs_df["fname_key"] = (
        "sample_"
        + pairs_df["pfdelta_scenario"].astype(str)
        + "_lam_"
        + pairs_df["lam_str"]
    )

    # Filename-accurate ordering, truncate to train_size
    pairs_df = pairs_df.sort_values("fname_key", kind="mergesort").head(train_size)

    split_idx = int(0.9 * len(pairs_df))
    train_df = pairs_df.iloc[:split_idx]
    valid_df = pairs_df.iloc[split_idx:]

    return {
        "train": train_df["scenario"].tolist(),
        "valid": valid_df["scenario"].tolist(),
    }

def build_task_for_case(
    *,
    task: float,
    case: str,
    data_processed: str,
    shuffle_files: str,
    out_root: str,
    allowed_splits: Optional[Set[str]] = None,
) -> None:
    if task not in TASK_CONFIG:
        raise ValueError(f"Unsupported task: {task}")

    task_cfg = TASK_CONFIG[task]

    for feasibility, per_grid_train in task_cfg.items():
        test_cfg = TEST_CONFIG.get(feasibility)

        for grid_type in GRID_TYPES:
            train_size = int(per_grid_train.get(grid_type, 0))
            test_size = int(test_cfg.get(grid_type, 0)) if test_cfg else 0

            if train_size == 0 and test_size == 0:
                continue

            if feasibility == "feasible":
                split_scenarios = _compute_splits_feasible(
                    shuffle_files=shuffle_files,
                    grid_type=grid_type,
                    train_size=train_size,
                    test_size=test_size,
                )

                train_raw = _source_raw_dir(data_processed, case, grid_type, "feasible_train")
                test_raw = _source_raw_dir(data_processed, case, grid_type, "feasible_test")
                missing = _missing_paths(train_raw, test_raw)
                if missing:
                    print(f"[task {task}] {case} {grid_type} feasible: missing {missing}; skipping")
                    continue
                raw_dirs = (train_raw, test_raw)
                table_names = _table_names_from_dirs(raw_dirs)
                for split in ("train", "valid", "test"):
                    if allowed_splits is not None and split not in allowed_splits:
                        continue
                    _write_split_tables(
                        table_names=table_names,
                        raw_dirs=raw_dirs,
                        out_dir=os.path.join(out_root, split, case, grid_type,"feasible", "raw"),
                        scenarios=split_scenarios[split],
                        log_prefix=f"[task {task}] {case} {grid_type} feasible {split}",
                    )

            elif feasibility == "near infeasible":
                train_raw = _source_raw_dir(data_processed, case, grid_type, "nose_train")
                test_raw = _source_raw_dir(data_processed, case, grid_type, "nose_test")
                missing = _missing_paths(train_raw, test_raw)
                if missing:
                    print(f"[task {task}] {case} {grid_type} nose: missing {missing}; skipping")
                    continue
                split_scenarios = _compute_splits_nose(
                    train_raw=train_raw,
                    test_raw=test_raw,
                    train_size=train_size,
                    test_size=test_size,
                )
                all_table_names = _table_names_from_dirs((train_raw, test_raw))

                for split in ("train", "valid", "test"):
                    if allowed_splits is not None and split not in allowed_splits:
                        continue
                    scenarios = split_scenarios[split]
                    if not scenarios:
                        continue
                    src_raw = train_raw if split in ("train", "valid") else test_raw
                    _write_split_tables(
                        table_names=all_table_names,
                        raw_dirs=(src_raw,),
                        out_dir=os.path.join(out_root, split, case, grid_type,"nose", "raw"),
                        scenarios=scenarios,
                        log_prefix=f"[task {task}] {case} {grid_type} nose {split}",
                    )

            elif feasibility == "approaching infeasible":
                train_raw = _source_raw_dir(data_processed, case, grid_type, "around_nose_train")
                if not os.path.exists(train_raw):
                    print(f"[task {task}] {case} {grid_type} around_nose: missing {train_raw}; skipping")
                    continue
                split_scenarios = _compute_splits_around_nose(train_raw=train_raw, train_size=train_size)
                table_names = _table_names_from_dirs((train_raw,))

                for split in ("train", "valid"):  # no test
                    if allowed_splits is not None and split not in allowed_splits:
                        continue
                    scenarios = split_scenarios[split]
                    if not scenarios:
                        continue
                    _write_split_tables(
                        table_names=table_names,
                        raw_dirs=(train_raw,),
                        out_dir=os.path.join(out_root, split, case, grid_type,"around_nose", "raw"),
                        scenarios=scenarios,
                        log_prefix=f"[task {task}] {case} {grid_type} around_nose {split}",
                    )

            else:
                raise ValueError(f"Unknown feasibility: {feasibility}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="case118", help="case name (e.g., case118)")
    p.add_argument(
        "--tasks",
        nargs="+",
        type=float,
        default=[3.1, 3.2, 3.3],
        help="task ids to build (e.g., 1.3 4.2). Default: 4.3",
    )
    p.add_argument(
        "--data-processed",
        default="/dccstor/gridfm/pfdelta_converted_split",
        help="path to data_processed",
    )
    p.add_argument(
        "--shuffle-files",
        default="/dccstor/gridfm/pfdelta_converted/shuffle_files",
        help="path to shuffle_files (n/raw_shuffle.json etc.)",
    )
    p.add_argument(
        "--out-root",
        default="/dccstor/gridfm/pfdelta_converted_split_tasks",
        help="output root",
    )
    args = p.parse_args()

    for t in args.tasks:
        out_root = os.path.join(args.out_root,f"task_{t}", args.case) if t not in (3.2, 3.3) else os.path.join(args.out_root,f"task_{t}")

        # For tasks 3.x, mimic PFDelta's per-split case selection as if all case parquet folders existed.
        if t in (3.1, 3.2, 3.3):
            # Determine which cases to write for each split.
            split_cases = TASK_SPLIT_CONFIG[t]


            for case in ALL_CASE_NAMES:
                allowed: Set[str] = set()
                if case in split_cases["valid"]:
                    allowed.add("valid")
                if case in split_cases["test"]:
                    allowed.add("test")
                if t == 3.1:
                    if case == args.case:
                        allowed.add("train")
                        allowed.add("valid")
                else:
                    if case in split_cases["train"]:
                        allowed.add("train")

                if not allowed:
                    continue

                build_task_for_case(
                    task=t,
                    case=case,
                    data_processed=args.data_processed,
                    shuffle_files=args.shuffle_files,
                    out_root=out_root,
                    allowed_splits=allowed,
                )
        else:
            build_task_for_case(
                task=t,
                case=args.case,
                data_processed=args.data_processed,
                shuffle_files=args.shuffle_files,
                out_root=out_root,
                allowed_splits=None,
            )


if __name__ == "__main__":
    main()

