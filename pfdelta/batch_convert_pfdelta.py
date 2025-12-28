#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch converter for pfdelta PowerModels/PGLib JSON solutions under a data_dir.

Behavior (mirrors opf_data/batch_convert.py style for PF/OPF data):
- Scans a directory for `sample_*.json` files (non-recursive).
- Parses each file as a single scenario and extracts per-bus quantities.
- Processes files in fixed-size chunks with multiprocessing.
- After each chunk, appends the chunk's rows to a single parquet file.


All powers are scaled by baseMVA (so they are in MW / MVar if baseMVA is in MVA),
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


def require(cond: bool, msg: str):
    if not cond:
        raise ValueError(msg)


def to_complex(r: float, x: float) -> complex:
    return complex(r, x)


def inv_complex(z: complex) -> complex:
    require(abs(z) > 0.0, "Nonphysical impedance: zero magnitude (r == 0 and x == 0).")
    return 1.0 / z


def check_index(name: str, idx: int, n: int):
    require(0 <= idx < n, f"Index out of range for {name}: {idx} not in [0, {n - 1}].")


def is_close_zero(z: complex) -> bool:
    return (z.real == 0.0) and (z.imag == 0.0)


def parse_scenario_from_path(p: Path) -> int:
    """
    Extract a zero-based scenario index from filenames of the form `sample_#.json`.
    Example:
        sample_1.json   ->  scenario 0
        sample_1000.json -> scenario 999
    """
    m = re.search(r"sample_(\d+)\.json$", p.name)
    require(m is not None, f"Filename does not match 'sample_#.json': {p.name}")
    # Use zero-based scenario indices to match other datasets in this repo.
    scen_idx = int(m.group(1)) - 1
    require(scen_idx >= 0, f"Scenario index must be non-negative in file {p.name}")
    return scen_idx


# ---------- Core per-file conversion ----------


def convert_one_pfdelta(
    data: dict,
    scenario: int,
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    """
    Convert a single pfdelta JSON (network + solution) into per-bus rows.

    Expected top-level structure (PowerModels / PGLib style):
      {
        "network": {
            "baseMVA": ...,
            "per_unit": true/false,
            "bus": { "<bus_id>": {...} },
            "gen": { "<gen_id>": {...} },
            "load": { "<load_id>": {...} },
            "branch": { "<branch_id>": {...} },
            ...
        },
        "solution": {
            "solution": {
                "baseMVA": ...,
                "bus": { "<bus_id>": {"vm": ..., "va": ...}, ... },
                "gen": { "<gen_id>": {"pg": ..., "qg": ...}, ... },
                "branch": {...},
                ...
            },
            ...
        }
      }
    """
    require(
        "network" in data and "solution" in data,
        "Missing 'network' or 'solution' sections.",
    )

    net = data["network"]
    sol_root = data["solution"]["solution"]

    # Base MVA used for scaling to MW / MVar
    baseMVA = net["baseMVA"]

    # ---- Nested function for branch processing ----
    def process_branch(
        bid: str,
        br: dict,
        sol_branches: Dict[str, dict],
        out_idx: int,
    ) -> Tuple[dict, complex, complex, complex, complex, int, int]:
        """Process a single branch (line or transformer) and return branch row + Y-matrix contributions."""
        # Bus indices
        f_bus_i = int(br["f_bus"])
        t_bus_i = int(br["t_bus"])
        require(
            1 <= f_bus_i <= n_buses,
            f"Branch {bid} from_bus {f_bus_i} out of range.",
        )
        require(1 <= t_bus_i <= n_buses, f"Branch {bid} to_bus {t_bus_i} out of range.")
        f = f_bus_i - 1  # Convert 1-based to 0-based
        t = t_bus_i - 1

        # Common branch parameters
        angmin = float(br["angmin"])
        angmax = float(br["angmax"])
        br_r = float(br["br_r"])
        br_x = float(br["br_x"])
        rate_a = float(br["rate_a"])
        br_status = float(br["br_status"])
        g_fr = float(br["g_fr"])
        b_fr = float(br["b_fr"])
        g_to = float(br["g_to"])
        b_to = float(br["b_to"])

        require(g_fr == 0, f"g_fr is not equal to 0 at branch {bid}: g_fr: {g_fr}")
        require(g_to == 0, f"g_to is not equal to 0 at branch {bid}: g_to: {g_to}")
        require(
            b_fr == b_to,
            f"b_fr and b_to are not equal at branch {bid}: b_fr: {b_fr}, b_to: {b_to}",
        )
        b = b_fr * 2

        # Tap and shift parameters
        tap_mag = float(br["tap"])
        if tap_mag == 0:
            tap_mag = 1.0  # we assume tap = 1 for AC lines
        shift = float(br["shift"])

        # Deactivated branches: zero admittances and power flows to match datakit
        if br_status == 0:
            Yff = Ytt = Yft = Ytf = complex(0.0, 0.0)
            pt = qt = pf = qf = 0.0
        else:
            require(br_x != 0.0, f"Nonphysical branch reactance at {bid}: x == 0.")

            require(tap_mag > 0.0, f"Nonphysical transformer tap at {bid}: tap <= 0.")

            # Calculate admittances using unified function
            Yff, Yft, Ytf, Ytt = compute_branch_admittances(
                r=br_r,
                x=br_x,
                b=b,  # Total shunt susceptance (function splits it)
                tap_mag=tap_mag,
                shift=shift,
            )

            # Solution power flows
            sb = sol_branches[bid]
            require(sb is not None, f"Solution is missing branch entry for id={bid}.")
            pt = float(sb["pt"])
            qt = float(sb["qt"])
            pf = float(sb["pf"])
            qf = float(sb["qf"])

        branch_row = {
            "scenario": scenario,
            "idx": out_idx,
            "from_bus": f,
            "to_bus": t,
            "pf": pf * baseMVA,
            "qf": qf * baseMVA,
            "pt": pt * baseMVA,
            "qt": qt * baseMVA,
            "Yff_r": Yff.real,
            "Yff_i": Yff.imag,
            "Yft_r": Yft.real,
            "Yft_i": Yft.imag,
            "Ytf_r": Ytf.real,
            "Ytf_i": Ytf.imag,
            "Ytt_r": Ytt.real,
            "Ytt_i": Ytt.imag,
            "tap": tap_mag,
            "shift": np.rad2deg(shift),
            "ang_min": np.rad2deg(angmin),
            "ang_max": np.rad2deg(angmax),
            "rate_a": rate_a * baseMVA,
            "br_status": br_status,
            "r": br_r,
            "x": br_x,
            "b": b,
        }

        return branch_row, Yff, Ytt, Yft, Ytf, f, t

    # ---- Buses & indexing ----
    buses: Dict[str, dict] = net["bus"]

    # Verify buses are consecutive from 1 to n and bus_id == bus_i
    n_buses = len(buses)
    for bus_i in range(1, n_buses + 1):
        bus_key = str(bus_i)
        require(
            bus_key in buses,
            f"Missing bus with key '{bus_key}' (expected consecutive buses 1 to {n_buses}).",
        )
        bus_info = buses[bus_key]
        actual_bus_i = int(bus_info["bus_i"])
        require(
            actual_bus_i == bus_i,
            f"Bus key '{bus_key}' has bus_i={actual_bus_i}, expected {bus_i}.",
        )

    # ---- Initialize per-bus aggregates ----
    Pd = np.zeros(n_buses, dtype=float)
    Qd = np.zeros(n_buses, dtype=float)
    Pg = np.zeros(n_buses, dtype=float)
    Qg = np.zeros(n_buses, dtype=float)
    Vm = np.zeros(n_buses, dtype=float)
    Va_deg = np.zeros(n_buses, dtype=float)
    GS = np.zeros(n_buses, dtype=float)
    BS = np.zeros(n_buses, dtype=float)

    # Bus attributes
    btype = np.zeros(n_buses, dtype=int)
    vn_kv = np.zeros(n_buses, dtype=float)
    min_vm_pu = np.zeros(n_buses, dtype=float)
    max_vm_pu = np.zeros(n_buses, dtype=float)

    for bus_i in range(1, n_buses + 1):
        idx = bus_i - 1  # 0-based index for arrays
        binfo = buses[str(bus_i)]
        bt = int(binfo["bus_type"])
        require(bt in (1, 2, 3, 4), f"Unknown bus_type {bt}.")
        btype[idx] = bt
        vn_kv[idx] = float(binfo["base_kv"])
        min_vm_pu[idx] = float(binfo["vmin"])
        max_vm_pu[idx] = float(binfo["vmax"])

    # ---- Loads: aggregate pd / qd to buses ----
    loads: Dict[str, dict] = net["load"]
    for ld in loads.values():
        status = int(ld["status"])
        if status == 0:
            continue
        bus_i = int(ld["load_bus"])
        require(1 <= bus_i <= n_buses, f"Load references invalid bus {bus_i}.")
        b = bus_i - 1  # Convert 1-based to 0-based
        Pd[b] += float(ld["pd"]) * baseMVA
        Qd[b] += float(ld["qd"]) * baseMVA

    # ---- Shunts: aggregate gs / bs to buses ----
    shunts: Dict[str, dict] = net["shunt"]
    for sh in shunts.values():
        status = int(sh["status"])
        if status == 0:
            continue
        bus_i = int(sh["shunt_bus"])
        require(1 <= bus_i <= n_buses, f"Shunt references invalid bus {bus_i}.")
        b = bus_i - 1  # Convert 1-based to 0-based
        GS[b] += float(sh["gs"])
        BS[b] += float(sh["bs"])

    # ---- Generators: aggregate pg / qg (from solution) to buses ----
    net_gens: Dict[str, dict] = net["gen"]
    sol_gens: Dict[str, dict] = sol_root["gen"]

    for gen_id, g in net_gens.items():
        status = int(g["gen_status"])
        if status == 0:
            # Offline generator: contribute zero to Pg/Qg.
            continue

        bus_i = int(g["gen_bus"])
        require(
            1 <= bus_i <= n_buses,
            f"Generator {gen_id} references invalid bus {bus_i}.",
        )
        b = bus_i - 1  # Convert 1-based to 0-based

        sg = sol_gens[gen_id]
        assert sg is not None, f"Solution is missing generator entry for id={gen_id}."

        Pg[b] += float(sg["pg"]) * baseMVA
        Qg[b] += float(sg["qg"]) * baseMVA

    # ---- Bus voltages (vm, va in radians) from solution ----
    sol_buses: Dict[str, dict] = sol_root["bus"]
    require(
        len(sol_buses) == n_buses,
        "Mismatch between network and solution bus counts.",
    )

    for bus_i in range(1, n_buses + 1):
        idx = bus_i - 1  # 0-based index for arrays
        sb = sol_buses[str(bus_i)]
        require(sb is not None, f"Solution is missing bus entry for bus_i={bus_i}.")

        vm = float(sb["vm"])
        va = float(sb["va"])  # radians
        Vm[idx] = vm
        Va_deg[idx] = np.rad2deg(va)

    # ---- Build Ybus from branches and shunts ----
    Y: Dict[Tuple[int, int], complex] = defaultdict(complex)

    def add_to_Y(i: int, j: int, y: complex):
        Y[(i, j)] += y

    branches: Dict[str, dict] = net["branch"]
    sol_branches: Dict[str, dict] = sol_root["branch"]

    # Verify branches are consecutive from 1 to n and branch_id == branch index
    n_branches = len(branches)
    for br_idx in range(1, n_branches + 1):
        br_key = str(br_idx)
        require(
            br_key in branches,
            f"Missing branch with key '{br_key}' (expected consecutive branches 1 to {n_branches}).",
        )
        br_info = branches[br_key]
        actual_br_idx = int(br_info["index"])
        require(
            actual_br_idx == br_idx,
            f"Branch key '{br_key}' has index={actual_br_idx}, expected {br_idx}.",
        )

    # Process all branches (lines and transformers)
    branch_rows: List[dict] = []
    for br_idx in range(1, n_branches + 1):
        out_idx = br_idx - 1  # 0-based index for output
        br = branches[str(br_idx)]

        branch_row, Yff, Ytt, Yft, Ytf, f, t = process_branch(
            str(br_idx),
            br,
            sol_branches,
            out_idx,
        )
        branch_rows.append(branch_row)
        add_to_Y(f, f, Yff)
        add_to_Y(t, t, Ytt)
        add_to_Y(f, t, Yft)
        add_to_Y(t, f, Ytf)

    # Shunt admittances on diagonal
    for b in range(n_buses):
        y_sh = complex(GS[b], BS[b])
        if not is_close_zero(y_sh):
            add_to_Y(b, b, y_sh)

    for i, j in Y.keys():
        check_index("Ybus index1", i, n_buses)
        check_index("Ybus index2", j, n_buses)

    # ---- Bus rows ----
    bus_rows: List[dict] = []
    for b in range(n_buses):
        PQ = 1 if btype[b] == 1 else 0
        PV = 1 if btype[b] == 2 else 0
        REF = 1 if btype[b] == 3 else 0
        bus_rows.append(
            {
                "scenario": scenario,
                "bus": b,
                "Pd": Pd[b],
                "Qd": Qd[b],
                "Pg": Pg[b],
                "Qg": Qg[b],
                "Vm": Vm[b],
                "Va": Va_deg[b],
                "PQ": PQ,
                "PV": PV,
                "REF": REF,
                "vn_kv": vn_kv[b],
                "min_vm_pu": min_vm_pu[b],
                "max_vm_pu": max_vm_pu[b],
                "GS": GS[b],
                "BS": BS[b],
            },
        )

    # ---- Generator rows ----
    slack_buses = np.where(btype == 3)[0]
    require(len(slack_buses) == 1, "There should be exactly one slack bus.")
    slack_bus = int(slack_buses[0])

    # Verify generators are consecutive from 1 to n and gen_id == gen index
    n_gens = len(net_gens)
    for gen_idx in range(1, n_gens + 1):
        gen_key = str(gen_idx)
        require(
            gen_key in net_gens,
            f"Missing generator with key '{gen_key}' (expected consecutive generators 1 to {n_gens}).",
        )
        gen_info = net_gens[gen_key]
        actual_gen_idx = int(gen_info["index"])
        require(
            actual_gen_idx == gen_idx,
            f"Generator key '{gen_key}' has index={actual_gen_idx}, expected {gen_idx}.",
        )

    gen_rows: List[dict] = []
    total_cost = 0.0
    out_idx = 0  # Counter for active generators only

    # Process all generators
    for gen_idx in range(1, n_gens + 1):
        gen_key = str(gen_idx)
        g = net_gens[gen_key]
        status = int(g["gen_status"])

        bus_i = int(g["gen_bus"])
        require(
            1 <= bus_i <= n_buses,
            f"Generator {gen_key} references invalid bus {bus_i}.",
        )
        bus_of_g = bus_i - 1  # Convert 1-based to 0-based

        if status == 1:
            sg = sol_gens[gen_key]
            p_mw = float(sg["pg"]) * baseMVA
            q_mvar = float(sg["qg"]) * baseMVA

        else:
            p_mw = 0.0
            q_mvar = 0.0

        pmin = float(g["pmin"]) * baseMVA
        pmax = float(g["pmax"]) * baseMVA
        qmin = float(g["qmin"]) * baseMVA
        qmax = float(g["qmax"]) * baseMVA

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

        gen_row = {
            "scenario": scenario,
            "idx": out_idx,
            "bus": bus_of_g,
            "p_mw": p_mw,
            "q_mvar": q_mvar,
            "min_p_mw": pmin,
            "max_p_mw": pmax,
            "min_q_mvar": qmin,
            "max_q_mvar": qmax,
            "cp0_eur": c0,
            "cp1_eur_per_mw": c1 / baseMVA,
            "cp2_eur_per_mw2": c2 / (baseMVA**2),
            "in_service": status,
            "is_slack_gen": 1 if bus_of_g == slack_bus else 0,
        }
        gen_rows.append(gen_row)

        total_cost += status * (
            gen_row["cp0_eur"]
            + gen_row["cp1_eur_per_mw"] * p_mw
            + gen_row["cp2_eur_per_mw2"] * (p_mw) ** 2
        )
        out_idx += 1  # Increment only for active generators

    sol_obj = float(data["solution"]["objective"])
    require(
        np.isclose(total_cost, sol_obj, atol=1e-3, rtol=1e-5),
        f"Generator cost sum {total_cost:.8f} != solution objective {sol_obj:.8f}, scenario {scenario}",
    )

    # ---- Ybus rows ----
    y_rows: List[dict] = []
    for (i, j), y in Y.items():
        if not is_close_zero(y):
            y_rows.append(
                {
                    "scenario": scenario,
                    "index1": i,
                    "index2": j,
                    "G": y.real,
                    "B": y.imag,
                },
            )

    return branch_rows, bus_rows, gen_rows, y_rows


def process_one(args):
    jf, scen_idx = args
    with open(jf, "r") as f:
        data = json.load(f)
    return convert_one_pfdelta(data, scenario=scen_idx)


# ---------- I/O helpers ----------


def append_df(out_path: Path, df: pd.DataFrame):
    """
    Append a DataFrame to a parquet file, creating it if needed.
    Uses fastparquet for compatibility with existing scripts.
    """
    if df.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df["scenario_partition"] = (df["scenario"] // n_scenario_per_partition).astype(
        "int64",
    )
    df.to_parquet(
        out_path,
        partition_cols=["scenario_partition"],
        engine="pyarrow",
        index=False,
    )


# ---------- Main driver (chunked + append) ----------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch-convert pfdelta JSON files (sample_*.json) to aggregated "
            "branch/bus/gen/y_bus parquets (chunked, append-per-chunk)."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default="/Users/apu/to_del/gridfm-datakit/pfdelta/data/case118_ieee_n_minus_one/raw",
        help="Directory containing sample_*.json files (non-recursive).",
    )
    parser.add_argument(
        "--out-dir",
        default="/Users/apu/to_del/gridfm-datakit/pfdelta/data/case118_ieee_n_minus_one/converted",
        help="Directory for aggregated parquet output.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Files per processing chunk (default: 2000).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    require(
        data_dir.exists() and data_dir.is_dir(),
        f"data_dir does not exist or is not a directory: {data_dir}",
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    branch_parquet = out_dir / "branch_data.parquet"
    bus_parquet = out_dir / "bus_data.parquet"
    gen_parquet = out_dir / "gen_data.parquet"
    ybus_parquet = out_dir / "y_bus_data.parquet"

    # Clean existing outputs to avoid duplication.
    for p in (branch_parquet, bus_parquet, gen_parquet, ybus_parquet):
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    # ---- Collect files and scenarios ----
    files = sorted(data_dir.glob("sample_*.json"))
    require(len(files) > 0, f"No files matching 'sample_*.json' found under {data_dir}")

    parsed: List[tuple[Path, int]] = []
    for jf in files:
        scen_idx = parse_scenario_from_path(jf)
        parsed.append((jf, scen_idx))

    # Sort by scenario index to ensure consistent global ordering.
    parsed.sort(key=lambda t: t[1])

    # ---- Process in chunks and append after each chunk ----
    chunk_size = max(1, args.chunk_size)
    num_chunks = (len(parsed) + chunk_size - 1) // chunk_size

    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size
        end = min((chunk_id + 1) * chunk_size, len(parsed))
        chunk_items = parsed[start:end]

        pool_args = [(jf, scen_idx) for jf, scen_idx in chunk_items]

        with Pool(cpu_count()) as pool:
            print(
                f"Processing chunk {chunk_id + 1}/{num_chunks} with {cpu_count()} processes",
            )
            results = list(
                tqdm(
                    pool.imap(process_one, pool_args),
                    total=len(pool_args),
                    desc=f"Chunk {chunk_id + 1}/{num_chunks}",
                    leave=True,
                ),
            )

        print("Aggregating chunk results...")
        all_branch: List[dict] = []
        all_bus: List[dict] = []
        all_gen: List[dict] = []
        all_ybus: List[dict] = []

        for r in tqdm(results, desc="Aggregating", leave=False):
            b_rows, u_rows, g_rows, y_rows = r
            all_branch.extend(b_rows)
            all_bus.extend(u_rows)
            all_gen.extend(g_rows)
            all_ybus.extend(y_rows)

        def fast_df(rows: List[dict]) -> pd.DataFrame:
            if not rows:
                return pd.DataFrame()
            keys = rows[0].keys()
            return pd.DataFrame.from_records(rows, columns=keys)

        branch_df = fast_df(all_branch)
        bus_df = fast_df(all_bus)
        gen_df = fast_df(all_gen)
        ybus_df = fast_df(all_ybus)

        print("Appending chunk dataframes to final parquets...")
        append_df(branch_parquet, branch_df)
        append_df(bus_parquet, bus_df)
        append_df(gen_parquet, gen_df)
        append_df(ybus_parquet, ybus_df)


if __name__ == "__main__":
    main()
