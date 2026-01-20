#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch converter for Matpower-style solved JSON snapshots under a data_dir.

Expected input format:
{
  "solved_net": { ... },   # network + solution merged
}

Behavior:
- Scans a directory for `sample_*.json` files (non-recursive).
- Parses each file as a single scenario and extracts per-bus quantities.
- Processes files in fixed-size chunks with multiprocessing.
- After each chunk, appends the chunk's rows to a single parquet file.

All powers are scaled by baseMVA (MW / MVar),
and voltage angles are stored in degrees.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from multiprocessing import Pool, cpu_count
from gridfm_datakit.utils.utils import n_scenario_per_partition
from gridfm_datakit.utils.power_balance import compute_branch_admittances

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------- Utilities ----------

def wrap_angle_deg(angle_deg: float) -> float:
    angle_deg = (angle_deg + 180) % 360 - 180
    return angle_deg

def require(cond: bool, msg: str):
    if not cond:
        raise ValueError(msg)


def check_index(name: str, idx: int, n: int):
    require(0 <= idx < n, f"Index out of range for {name}: {idx} not in [0, {n - 1}].")


def is_close_zero(z: complex) -> bool:
    return (z.real == 0.0) and (z.imag == 0.0)


def parse_scenario_from_path(p: Path):
    """
    Extract scenario index and optional lambda from filenames.

    Supported patterns:
      - sample_<scen>_lam_<lambda>.json   (e.g. sample_2666_lam_0p14900.json)
      - sample_<scen>_nose.json
      - sample_<scen>.json
    """
    name = p.name

    m = re.search(
        r"^sample_(\d+)(?:_lam_([0-9]+p[0-9]+)|_nose)?\.json$",
        name,
    )
    require(m is not None, f"Filename does not match expected pattern: {name}")
    scen_idx = int(m.group(1))
    require(scen_idx >= 0, f"Scenario index must be non-negative in file {name}")

    lam = None
    if m.group(2) is not None:
        lam = float(m.group(2).replace("p", "."))

    return scen_idx, lam


# ---------- Core per-file conversion ----------

def convert_one_pfdelta(
    data: dict,
    scenario: int,
    scenario_orig: int,
    lam: float | None,
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:

    require("solved_net" in data, "Missing 'solved_net' section.")
    net = data["solved_net"]
    baseMVA = float(net["baseMVA"])

    buses: Dict[str, dict] = net["bus"]
    n_buses = len(buses)

    for bus_i in range(1, n_buses + 1):
        b = buses.get(str(bus_i))
        require(b is not None, f"Missing bus {bus_i}")
        require(int(b["bus_i"]) == bus_i, f"Bus index mismatch at bus {bus_i}")

    Pd = np.zeros(n_buses)
    Qd = np.zeros(n_buses)
    Pg = np.zeros(n_buses)
    Qg = np.zeros(n_buses)
    Vm = np.zeros(n_buses)
    Va_deg = np.zeros(n_buses)
    GS = np.zeros(n_buses)
    BS = np.zeros(n_buses)

    btype = np.zeros(n_buses, dtype=int)
    vn_kv = np.zeros(n_buses)
    min_vm_pu = np.zeros(n_buses)
    max_vm_pu = np.zeros(n_buses)

    for bus_i in range(1, n_buses + 1):
        idx = bus_i - 1
        binfo = buses[str(bus_i)]
        btype[idx] = int(binfo["bus_type"])
        require(
            btype[idx] in (1, 2, 3, 4),
            f"Invalid bus type {btype[idx]} at bus {bus_i}",
        )
        vn_kv[idx] = float(binfo["base_kv"])
        min_vm_pu[idx] = float(binfo["vmin"])
        max_vm_pu[idx] = float(binfo["vmax"])
        Vm[idx] = float(binfo["vm"])
        Va_deg[idx] = wrap_angle_deg(np.rad2deg(float(binfo["va"])))

    for ld in net.get("load", {}).values():
        if int(ld["status"]) == 0:
            continue
        b = int(ld["load_bus"]) - 1
        Pd[b] += float(ld["pd"]) * baseMVA
        Qd[b] += float(ld["qd"]) * baseMVA

    for sh in net.get("shunt", {}).values():
        if int(sh["status"]) == 0:
            continue
        b = int(sh["shunt_bus"]) - 1
        GS[b] += float(sh["gs"])
        BS[b] += float(sh["bs"])

    net_gens: Dict[str, dict] = net["gen"]
    for g in net_gens.values():
        if int(g["gen_status"]) == 0:
            continue
        b = int(g["gen_bus"]) - 1
        Pg[b] += float(g["pg"]) * baseMVA
        Qg[b] += float(g["qg"]) * baseMVA

    Y: Dict[Tuple[int, int], complex] = defaultdict(complex)
    branch_rows: List[dict] = []
    branches: Dict[str, dict] = net["branch"]

    # Verify branches are consecutive from 1 to n and branch_id == branch index
    for br_idx in range(1, len(branches) + 1):
        br = branches.get(str(br_idx))
        require(br is not None, f"Missing branch {br_idx}")
        require(int(br["index"]) == br_idx, f"Branch index mismatch at branch {br_idx}")

    for br_idx in range(1, len(branches) + 1):
        br = branches[str(br_idx)]
        f = int(br["f_bus"]) - 1
        t = int(br["t_bus"]) - 1

        br_r = float(br["br_r"])
        br_x = float(br["br_x"])
        b = 2.0 * float(br["b_fr"])
        tap = float(br["tap"]) or 1.0
        shift = float(br["shift"])
        status = int(br["br_status"])
        
        require(
            float(br["g_fr"]) == 0.0,
            f"g_fr is not equal to 0 at branch {br_idx}: g_fr: {br['g_fr']}",
        )
        require(
            float(br["g_to"]) == 0.0,
            f"g_to is not equal to 0 at branch {br_idx}: g_to: {br['g_to']}",
        )
        require(
            float(br["b_fr"]) == float(br["b_to"]),
            f"b_fr and b_to are not equal at branch {br_idx}: b_fr: {br['b_fr']}, b_to: {br['b_to']}",
        )

        if status == 0:
            Yff = Yft = Ytf = Ytt = complex(0.0, 0.0)
            require(float(br["pf"]) == 0.0, f"Non-zero pf on out-of-service branch {br_idx}")
            require(float(br["qf"]) == 0.0, f"Non-zero qf on out-of-service branch {br_idx}")
            require(float(br["pt"]) == 0.0, f"Non-zero pt on out-of-service branch {br_idx}")
            require(float(br["qt"]) == 0.0, f"Non-zero qt on out-of-service branch {br_idx}")
        else:
            Yff, Yft, Ytf, Ytt = compute_branch_admittances(
                r=br_r, x=br_x, b=b, tap_mag=tap, shift=shift
            )

        branch_rows.append(
            {
                "scenario": scenario,
                "pfdelta_scenario": scenario_orig,
                "lam": lam,
                "idx": br_idx - 1,
                "from_bus": f,
                "to_bus": t,
                "pf": float(br["pf"]) * baseMVA,
                "qf": float(br["qf"]) * baseMVA,
                "pt": float(br["pt"]) * baseMVA,
                "qt": float(br["qt"]) * baseMVA,
                "Yff_r": Yff.real,
                "Yff_i": Yff.imag,
                "Yft_r": Yft.real,
                "Yft_i": Yft.imag,
                "Ytf_r": Ytf.real,
                "Ytf_i": Ytf.imag,
                "Ytt_r": Ytt.real,
                "Ytt_i": Ytt.imag,
                "tap": tap,
                "shift": np.rad2deg(float(shift)),
                "ang_min": np.rad2deg(float(br["angmin"])), 
                "ang_max": np.rad2deg(float(br["angmax"])),
                "rate_a": float(br["rate_a"]) * baseMVA,
                "br_status": status,
                "r": br_r,
                "x": br_x,
                "b": b,
            }
        )

        Y[(f, f)] += Yff
        Y[(t, t)] += Ytt
        Y[(f, t)] += Yft
        Y[(t, f)] += Ytf

    for b in range(n_buses):
        y_sh = complex(GS[b], BS[b])
        if not is_close_zero(y_sh):
            Y[(b, b)] += y_sh

    bus_rows = []
    for b in range(n_buses):
        bus_rows.append(
            {
                "scenario": scenario,
                "pfdelta_scenario": scenario_orig,
                "lam": lam,
                "bus": b,
                "Pd": Pd[b],
                "Qd": Qd[b],
                "Pg": Pg[b],
                "Qg": Qg[b],
                "Vm": Vm[b],
                "Va": Va_deg[b],
                "PQ": 1 if btype[b] == 1 else 0,
                "PV": 1 if btype[b] == 2 else 0,
                "REF": 1 if btype[b] == 3 else 0,
                "vn_kv": vn_kv[b],
                "min_vm_pu": min_vm_pu[b],
                "max_vm_pu": max_vm_pu[b],
                "GS": GS[b],
                "BS": BS[b],
            }
        )

    slack_buses = np.where(btype == 3)[0]
    require(len(slack_buses) == 1, "There should be exactly one slack bus.")
    slack_bus = int(slack_buses[0])
    gen_rows = []
    out_idx = 0
    for gen_id, g in sorted(net_gens.items(), key=lambda x: int(x[0])):
        bus = int(g["gen_bus"]) - 1
        status = int(g["gen_status"])
        
        
        cost = g["cost"]
        c0 = 0.0
        c1 = 0.0
        c2 = 0.0
        if isinstance(cost, (list, tuple)):
            if len(cost) == 3:
                c2, c1, c0 = [float(x) for x in cost]
            elif len(cost) == 2:
                c1, c0 = [float(x) for x in cost]
            elif len(cost) == 1:
                c0 = float(cost[0])

        gen_rows.append(
            {
                "scenario": scenario,
                "pfdelta_scenario": scenario_orig,
                "lam": lam,
                "idx": out_idx,
                "bus": bus,
                "p_mw": (float(g["pg"]) * baseMVA) if status == 1 else 0.0,
                "q_mvar": (float(g["qg"]) * baseMVA) if status == 1 else 0.0,
                "min_p_mw": float(g["pmin"]) * baseMVA,
                "max_p_mw": float(g["pmax"]) * baseMVA,
                "min_q_mvar": float(g["qmin"]) * baseMVA,
                "max_q_mvar": float(g["qmax"]) * baseMVA,
                "cp0_eur": c0,
                "cp1_eur_per_mw": c1 / baseMVA,
                "cp2_eur_per_mw2": c2 / (baseMVA**2),
                "in_service": status,
                "is_slack_gen": 1 if bus == slack_bus else 0,
            }
        )
        if status == 0:
            require(
                np.isclose(float(g["pg"]), 0.0) and np.isclose(float(g["qg"]), 0.0),
                f"Expected out-of-service generator {gen_id} to have pg=qg=0 in solved_net.",
            )
        out_idx += 1

    y_rows = [
        {
            "scenario": scenario,
            "pfdelta_scenario": scenario_orig,
            "lam": lam,
            "index1": i,
            "index2": j,
            "G": y.real,
            "B": y.imag,
        }
        for (i, j), y in Y.items()
        if not is_close_zero(y)
    ]

    return branch_rows, bus_rows, gen_rows, y_rows


def process_one(args):
    jf, scenario, scenario_orig, lam = args
    with open(jf, "r") as f:
        data = json.load(f)
    return convert_one_pfdelta(data, scenario, scenario_orig, lam)


# ---------- I/O helpers ----------

def append_df(out_path: Path, df: pd.DataFrame):
    if df.empty:
        return

    if "lam" in df.columns and df["lam"].isna().all():
        df = df.drop(columns=["lam"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df["scenario_partition"] = (df["scenario"] // n_scenario_per_partition).astype("int64")
    df.to_parquet(
        out_path,
        partition_cols=["scenario_partition"],
        engine="pyarrow",
        index=False,
    )


# ---------- Main driver ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=2000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    branch_parquet = out_dir / "branch_data.parquet"
    bus_parquet = out_dir / "bus_data.parquet"
    gen_parquet = out_dir / "gen_data.parquet"
    ybus_parquet = out_dir / "y_bus_data.parquet"

    for p in (branch_parquet, bus_parquet, gen_parquet, ybus_parquet):
        p.unlink(missing_ok=True)

    files = sorted(data_dir.glob("sample_*.json"))
    require(files, "No sample_*.json files found")

    parsed = [(f, *parse_scenario_from_path(f)) for f in files]
    parsed.sort(key=lambda x: x[1])

    parsed = [(f, i, orig, lam) for i, (f, orig, lam) in enumerate(parsed)]

    chunk_size = max(1, args.chunk_size)
    for i in range(0, len(parsed), chunk_size):
        chunk = parsed[i : i + chunk_size]

        with Pool(cpu_count()) as pool:
            results = list(
                tqdm(
                    pool.imap(process_one, chunk),
                    total=len(chunk),
                    desc=f"Chunk {i // chunk_size + 1}",
                )
            )

        all_branch, all_bus, all_gen, all_ybus = [], [], [], []
        for b, u, g, y in results:
            all_branch.extend(b)
            all_bus.extend(u)
            all_gen.extend(g)
            all_ybus.extend(y)

        append_df(branch_parquet, pd.DataFrame(all_branch))
        append_df(bus_parquet, pd.DataFrame(all_bus))
        append_df(gen_parquet, pd.DataFrame(all_gen))
        append_df(ybus_parquet, pd.DataFrame(all_ybus))


if __name__ == "__main__":
    main()
