"""Convert one gridfm-datakit parquet scenario to mixed-units PowerModels JSON."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import pandas as pd

from gridfm_datakit.utils.utils import n_scenario_per_partition


def _load_frames(data_dir: str, scenario: int) -> Dict[str, pd.DataFrame]:
    partition = scenario // n_scenario_per_partition
    filters = [("scenario_partition", "=", partition)]
    frames = {}
    for name in ("bus", "gen", "branch"):
        path = os.path.join(data_dir, f"{name}_data.parquet")
        df = pd.read_parquet(path, filters=filters, engine="pyarrow")
        df = df.loc[df["scenario"] == scenario]
        if df.empty:
            raise ValueError(f"No rows for scenario {scenario} in {name}_data.parquet")
        key = "bus" if name == "bus" else "idx"
        frames[name] = df.sort_values(key, kind="stable").reset_index(drop=True)
    return frames


def _bus_type(row: pd.Series) -> int:
    if row["REF"] == 1:
        return 3
    if row["PV"] == 1:
        return 2
    return 1


def _gencost(row: pd.Series) -> tuple[int, list[float]]:
    if row["cp2_eur_per_mw2"] != 0:
        return 3, [row["cp2_eur_per_mw2"], row["cp1_eur_per_mw"], row["cp0_eur"]]
    return 2, [row["cp1_eur_per_mw"], row["cp0_eur"]]


def frames_to_json(
    bus_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    output_path: str,
    base_mva: float = 100.0,
) -> str:
    """Write pre-loaded bus/gen/branch frames as mixed-units PowerModels JSON."""
    vm = {int(r.bus): float(r.Vm) for r in bus_df.itertuples()}

    data: Dict[str, Any] = {
        "per_unit": False,
        "baseMVA": base_mva,
        "name": "case_from_parquet",
        "source_type": "gridfm-datakit",
        "source_version": "1",
        "bus": {},
        "load": {},
        "shunt": {},
        "gen": {},
        "branch": {},
        "storage": {},
        "dcline": {},
        "switch": {},
    }

    load_i = shunt_i = 1
    for row in bus_df.itertuples():
        bus_id = int(row.bus) + 1
        data["bus"][str(bus_id)] = {
            "index": bus_id,
            "bus_i": bus_id,
            "bus_type": _bus_type(bus_df.loc[row.Index]),
            "vm": float(row.Vm),
            "va": float(row.Va),
            "vmin": float(row.min_vm_pu),
            "vmax": float(row.max_vm_pu),
            "base_kv": float(row.vn_kv),
            "area": 1,
            "zone": 1,
        }
        if row.Pd > 0 or row.Qd > 0:
            data["load"][str(load_i)] = {
                "index": load_i,
                "load_bus": bus_id,
                "status": 1,
                "pd": float(row.Pd),
                "qd": float(row.Qd),
            }
            load_i += 1
        if row.GS!=0 or row.BS!=0:
            data["shunt"][str(shunt_i)] = {
                "index": shunt_i,
                "shunt_bus": bus_id,
                "status": 1,
                "gs": float(row.GS) * base_mva,
                "bs": float(row.BS) * base_mva,
            }
            shunt_i += 1

    for i, row in enumerate(gen_df.itertuples(), start=1):
        n_cost, cost = _gencost(gen_df.loc[row.Index])
        data["gen"][str(i)] = {
            "index": i,
            "gen_bus": int(row.bus) + 1,
            "gen_status": int(row.in_service),
            "pg": float(row.p_mw),
            "qg": float(row.q_mvar),
            "pmin": float(row.min_p_mw),
            "pmax": float(row.max_p_mw),
            "qmin": float(row.min_q_mvar),
            "qmax": float(row.max_q_mvar),
            "vg": vm[int(row.bus)],
            "mbase": base_mva,
            "startup": 0.0,
            "shutdown": 0.0,
            "model": 2,
            "ncost": n_cost,
            "cost": cost,
        }

    for i, row in enumerate(branch_df.itertuples(), start=1):
        tap = float(row.tap)
        transformer = ((tap != 1.0) and (tap!=0))
        data["branch"][str(i)] = {
            "index": i,
            "f_bus": int(row.from_bus) + 1,
            "t_bus": int(row.to_bus) + 1,
            "br_r": float(row.r),
            "br_x": float(row.x),
            "g_fr": 0.0,
            "b_fr": float(row.b) / 2.0,
            "g_to": 0.0,
            "b_to": float(row.b) / 2.0,
            "rate_a": float(row.rate_a),
            "rate_b": 0.0,
            "rate_c": 0.0,
            "tap": tap if transformer else 1.0,
            "shift": float(row.shift),
            "angmin": float(row.ang_min),
            "angmax": float(row.ang_max),
            "transformer": transformer,
            "br_status": int(row.br_status),
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return output_path


def parquet_to_json(
    data_dir: str,
    scenario: int,
    output_path: str,
    base_mva: float = 100.0,
) -> str:
    """Write one parquet scenario as mixed-units PowerModels JSON."""
    bus_df, gen_df, branch_df = _load_frames(data_dir, scenario).values()
    return frames_to_json(bus_df, gen_df, branch_df, output_path, base_mva=base_mva)
