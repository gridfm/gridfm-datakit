#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch converter for grid/solution JSONs under a data_dir, processed in chunks.

Key behavior per your request:
- Processes input files in fixed-size chunks (default: 100).
- After each chunk, sorts rows within the chunk and APPENDS them to the final parquets.
- No end-of-run concatenation step.
- By default, pre-existing final parquets are removed at startup to avoid duplication.

Final outputs (continuously appended per chunk, globally sorted due to monotone scenario ordering + per-chunk sort):
  - branch_data.parquet      (sorted by: scenario, idx)
  - bus_data.parquet         (sorted by: scenario, bus)
  - gen_data.parquet         (sorted by: scenario, bus, idx)
  - y_bus_data.parquet       (sorted by: scenario, index1, index2)
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from gridfm_datakit.utils.utils import n_scenario_per_partition
from gridfm_datakit.utils.power_balance import compute_branch_admittances


# ---------- Utilities ----------


def require(cond: bool, msg: str):
    if not cond:
        raise ValueError(msg)


def to_complex(r: float, x: float) -> complex:
    return complex(r, x)


def inv_complex(z: complex) -> complex:
    require(abs(z) > 0.0, "Nonphysical impedance: zero magnitude (r == 0 and x == 0).")
    return 1.0 / z


def ensure_len(name: str, a, b):
    require(
        len(a) == len(b),
        f"Inconsistent list lengths for {name}: {len(a)} vs {len(b)}.",
    )


def check_index(name: str, idx: int, n: int):
    require(0 <= idx < n, f"Index out of range for {name}: {idx} not in [0, {n - 1}].")


def is_close_zero(z: complex) -> bool:
    return (z.real == 0.0) and (z.imag == 0.0)


def parse_scenario_from_path(p: Path) -> Tuple[str, int]:
    m = re.search(r"(example)_(\d+)\.json$", p.name)
    require(m is not None, f"Filename does not match 'example_#.json': {p.name}")
    scen_label = f"{m.group(1)}_{m.group(2)}"
    scen_idx = int(m.group(2))
    return scen_label, scen_idx


# ---------- Core per-file conversion ----------


def convert_one(
    data: dict,
    scenario: int,
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    require(
        "grid" in data and "solution" in data,
        "Missing 'grid' or 'solution' sections.",
    )

    grid = data["grid"]
    sol = data["solution"]
    baseMVA = data["grid"]["context"][0][0][0]
    objective = data["metadata"]["objective"]

    # ---- Nodes / Buses ----
    buses = grid["nodes"]["bus"]
    nb = len(buses)
    require(nb > 0, "No buses found.")

    bus_sol = sol["nodes"]["bus"]
    require(len(bus_sol) == nb, "Bus solution length mismatch.")

    # ---- Loads ----
    loads = grid["nodes"].get("load", [])
    nl = len(loads)
    load_link = grid["edges"]["load_link"]
    ensure_len(
        "load_link senders/receivers",
        load_link["senders"],
        load_link["receivers"],
    )
    require(
        len(load_link["senders"]) == nl,
        f"load_link size {len(load_link['senders'])} must equal number of loads {nl}.",
    )

    # ---- Generators ----
    gens = grid["nodes"].get("generator", [])
    ng = len(gens)

    gen_sol = sol["nodes"].get("generator", [])
    require(len(gen_sol) == ng, "Generator solution length mismatch.")

    gen_link = grid["edges"]["generator_link"]
    ensure_len(
        "generator_link senders/receivers",
        gen_link["senders"],
        gen_link["receivers"],
    )
    require(
        len(gen_link["senders"]) == ng,
        f"generator_link size {len(gen_link['senders'])} must equal number of generators {ng}.",
    )

    # ---- Shunts ----
    shunts = grid["nodes"].get("shunt", [])
    nsh = len(shunts)
    sh_link = grid["edges"].get("shunt_link", {"senders": [], "receivers": []})
    ensure_len("shunt_link senders/receivers", sh_link["senders"], sh_link["receivers"])
    require(
        len(sh_link["senders"]) == nsh,
        f"shunt_link size {len(sh_link['senders'])} must equal number of shunts {nsh}.",
    )

    # ---- AC Lines ----
    ac = grid["edges"]["ac_line"]
    ac_send, ac_recv, ac_feat = ac["senders"], ac["receivers"], ac["features"]
    ensure_len("ac_line senders/receivers", ac_send, ac_recv)
    ensure_len("ac_line senders/features", ac_send, ac_feat)
    nac = len(ac_send)

    sol_ac = sol["edges"]["ac_line"]
    ensure_len("solution.ac_line senders", sol_ac["senders"], ac_send)
    ensure_len("solution.ac_line receivers", sol_ac["receivers"], ac_recv)
    ensure_len("solution.ac_line features", sol_ac["features"], ac_feat)

    # ---- Transformers ----
    tf = grid["edges"]["transformer"]
    tf_send, tf_recv, tf_feat = tf["senders"], tf["receivers"], tf["features"]
    ensure_len("transformer senders/receivers", tf_send, tf_recv)
    ensure_len("transformer senders/features", tf_send, tf_feat)
    ntf = len(tf_send)

    sol_tf = sol["edges"]["transformer"]
    ensure_len("solution.transformer senders", sol_tf["senders"], tf_send)
    ensure_len("solution.transformer receivers", sol_tf["receivers"], tf_recv)
    ensure_len("solution.transformer features", sol_tf["features"], tf_feat)

    # ---- Index checks & feature arity ----
    for i in range(nac):
        f, t = ac_send[i], ac_recv[i]
        check_index("ac_line sender", f, nb)
        check_index("ac_line receiver", t, nb)
        require(len(ac_feat[i]) >= 9, "ac_line feature must have at least 9 elements.")
        require(
            len(sol_ac["features"][i]) >= 4,
            "solution.ac_line feature must have 4 elements.",
        )

    for i in range(ntf):
        f, t = tf_send[i], tf_recv[i]
        check_index("transformer sender", f, nb)
        check_index("transformer receiver", t, nb)
        require(
            len(tf_feat[i]) >= 11,
            "transformer feature must have at least 11 elements.",
        )
        require(
            len(sol_tf["features"][i]) >= 4,
            "solution.transformer feature must have 4 elements.",
        )

    for i in range(nl):
        check_index("load_link receivers(bus)", load_link["receivers"][i], nb)

    for g in range(ng):
        check_index("generator_link receivers(bus)", gen_link["receivers"][g], nb)

    for s in range(nsh):
        check_index("shunt_link receivers(bus)", sh_link["receivers"][s], nb)

    # ---- Aggregate bus injections and shunts ----
    Pd = [0.0] * nb
    Qd = [0.0] * nb
    for i in range(nl):
        pd, qd = loads[i]
        b = load_link["receivers"][i]
        Pd[b] += pd
        Qd[b] += qd

    Pg = [0.0] * nb
    Qg = [0.0] * nb
    for g in range(ng):
        pg, qg = gen_sol[g]
        b = gen_link["receivers"][g]
        Pg[b] += pg
        Qg[b] += qg

    GS = [0.0] * nb
    BS = [0.0] * nb
    for s in range(nsh):
        bs, gs = shunts[s]
        b = sh_link["receivers"][s]
        GS[b] += gs
        BS[b] += bs

    # ---- Bus voltage setpoints/types/bounds ----
    Vm = [sol_vm for (_, sol_vm) in bus_sol]
    Va = [sol_va for (sol_va, _) in bus_sol]

    vn_kv = [x[0] for x in buses]
    btype = [int(x[1]) for x in buses]
    min_vm = [x[2] for x in buses]
    max_vm = [x[3] for x in buses]

    for bt in btype:
        require(bt in (1, 2, 3, 4), f"Unknown bus_type {bt}.")

    # ---- Build branch rows and Ybus accumulator ----
    Y: Dict[Tuple[int, int], complex] = defaultdict(complex)
    branch_rows: List[dict] = []

    def add_to_Y(i: int, j: int, y: complex):
        Y[(i, j)] += y

    # AC lines
    for k in range(nac):
        f, t = ac_send[k], ac_recv[k]
        angmin, angmax, b_fr, b_to, br_r, br_x, rate_a, rate_b, rate_c = ac_feat[k][:9]
        require(br_x != 0.0, f"Nonphysical line reactance at line {k}: x == 0.")
        assert b_fr == b_to, (
            f"b_fr and b_to are not equal at line {k}: b_fr: {b_fr}, b_to: {b_to}"
        )

        # Calculate admittances (AC line: tap=1.0, shift=0.0)
        try:
            Yff, Yft, Ytf, Ytt = compute_branch_admittances(
                r=br_r,
                x=br_x,
                b=b_fr * 2,  # Total shunt susceptance
                tap_mag=1.0,  # AC line has tap = 1.0
                shift=0.0,  # No phase shift
            )
        except ValueError as e:
            raise ValueError(f"Admittance calculation failed at AC line {k}: {e}")

        pt, qt, pf, qf = sol_ac["features"][k]
        tap = 1.0
        br_status = 1.0

        branch_rows.append(
            {
                "scenario": scenario,
                "idx": k,
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
                "tap": tap,
                "shift": 0.0,
                "ang_min": np.rad2deg(angmin),
                "ang_max": np.rad2deg(angmax),
                "rate_a": rate_a * baseMVA,
                "br_status": br_status,
                "r": br_r,
                "x": br_x,
                "b": b_fr * 2,
            },
        )

        add_to_Y(f, f, Yff)
        add_to_Y(t, t, Ytt)
        add_to_Y(f, t, Yft)
        add_to_Y(t, f, Ytf)

    n_lines = len(branch_rows)

    # Transformers
    for k in range(ntf):
        f, t = tf_send[k], tf_recv[k]
        (
            angmin,
            angmax,
            br_r,
            br_x,
            rate_a,
            rate_b,
            rate_c,
            tap_mag,
            shift,
            b_fr,
            b_to,
        ) = tf_feat[k][:11]
        require(br_x != 0.0, f"Nonphysical transformer reactance at tf {k}: x == 0.")
        assert b_fr == b_to, (
            f"b_fr and b_to are not equal at transformer {k}: b_fr: {b_fr}, b_to: {b_to}"
        )
        b = b_fr * 2

        require(
            tap_mag > 0.0,
            f"Nonphysical transformer tap magnitude at transformer {k}: tap == 0.",
        )

        # Calculate admittances
        Yff, Yft, Ytf, Ytt = compute_branch_admittances(
            r=br_r,
            x=br_x,
            b=b,  # Total shunt susceptance
            tap_mag=tap_mag,
            shift=shift,
        )

        pt, qt, pf, qf = sol_tf["features"][k]
        br_status = 1.0

        branch_rows.append(
            {
                "scenario": scenario,
                "idx": n_lines + k,
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
            },
        )

        add_to_Y(f, f, Yff)
        add_to_Y(t, t, Ytt)
        add_to_Y(f, t, Yft)
        add_to_Y(t, f, Ytf)

    # Shunt admittances
    for b in range(nb):
        y_sh = complex(GS[b], BS[b])
        if not is_close_zero(y_sh):
            add_to_Y(b, b, y_sh)

    for i, j in Y.keys():
        check_index("Ybus index1", i, nb)
        check_index("Ybus index2", j, nb)

    # ---- Bus rows ----
    bus_rows: List[dict] = []
    for b in range(nb):
        PQ = 1 if btype[b] == 1 else 0
        PV = 1 if btype[b] == 2 else 0
        REF = 1 if btype[b] == 3 else 0
        assert btype[b] != 4, f"Bus {b} is inactive"
        bus_rows.append(
            {
                "scenario": scenario,
                "bus": b,
                "Pd": Pd[b] * baseMVA,
                "Qd": Qd[b] * baseMVA,
                "Pg": Pg[b] * baseMVA,
                "Qg": Qg[b] * baseMVA,
                "Vm": Vm[b],
                "Va": np.rad2deg(Va[b]),
                "PQ": PQ,
                "PV": PV,
                "REF": REF,
                "vn_kv": vn_kv[b],
                "min_vm_pu": min_vm[b],
                "max_vm_pu": max_vm[b],
                "GS": GS[b],
                "BS": BS[b],
            },
        )

    slack_buses = np.where(np.array(btype) == 3)[0]
    # assert only one slack bus
    assert len(slack_buses) == 1, "There should be exactly one slack bus"
    slack_bus = slack_buses[0]

    # ---- Generator rows and cost check ----
    cost = 0.0
    gen_rows: List[dict] = []
    for g in range(ng):
        (mbase, pg0, pmin, pmax, qg0, qmin, qmax, vg, c2, c1, c0) = gens[g][:11]
        bus_of_g = gen_link["receivers"][g]
        p_mw, q_mvar = gen_sol[g]
        cost += c0 + c1 * p_mw + c2 * (p_mw) ** 2
        gen_rows.append(
            {
                "scenario": scenario,
                "idx": g,
                "bus": bus_of_g,
                "p_mw": p_mw * baseMVA,
                "q_mvar": q_mvar * baseMVA,
                "min_p_mw": pmin * baseMVA,
                "max_p_mw": pmax * baseMVA,
                "min_q_mvar": qmin * baseMVA,
                "max_q_mvar": qmax * baseMVA,
                "cp0_eur": c0,
                "cp1_eur_per_mw": c1 / baseMVA,
                "cp2_eur_per_mw2": c2 / baseMVA**2,
                "in_service": 1,
                "is_slack_gen": 1 if bus_of_g == slack_bus else 0,
            },
        )

    assert (cost - objective) < 1e-6, (
        f"Generator cost function does not match objective. Calculated: {cost}, objective: {objective}"
    )

    # ---- Ybus rows & validation ----
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
    return convert_one(data, scenario=scen_idx)


# ---------- I/O helpers ----------


def append_df(out_path: Path, df: pd.DataFrame):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Add partition column for scenario-based partitioning (n_scenario_per_partition scenarios per partition)
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
        description="Batch-convert JSON grid/solution files to aggregated parquets with Ybus validation (chunked, append-per-chunk).",
    )
    parser.add_argument(
        "data_dir",
        help="Directory containing example*.json files (searched recursively).",
    )
    parser.add_argument("out_dir", help="Directory for aggregated parquet outputs.")

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Files per processing chunk (default: 100).",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    require(
        data_dir.exists() and data_dir.is_dir(),
        f"data_dir does not exist or is not a directory: {data_dir}",
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean outputs unless resuming
    branch_parquet = out_dir / "branch_data.parquet"
    bus_parquet = out_dir / "bus_data.parquet"
    gen_parquet = out_dir / "gen_data.parquet"
    ybus_parquet = out_dir / "y_bus_data.parquet"
    for p in (branch_parquet, bus_parquet, gen_parquet, ybus_parquet):
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    # ---- Collect files, parse indices, sort by scenario index ----
    files = list(data_dir.rglob("example*.json"))
    require(len(files) > 0, f"No files matching 'example*.json' found under {data_dir}")

    parsed: List[Tuple[Path, int]] = []
    for jf in files:
        _, scen_idx = parse_scenario_from_path(jf)
        parsed.append((jf, scen_idx))

    parsed.sort(key=lambda t: t[1])  # ensures monotone scenario order across chunks

    # ---- Process in chunks and APPEND after each chunk ----
    chunk_size = max(1, args.chunk_size)
    num_chunks = (len(parsed) + chunk_size - 1) // chunk_size

    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size
        end = min((chunk_id + 1) * chunk_size, len(parsed))
        chunk_items = parsed[start:end]

        # Prepare args for multiprocessing
        pool_args = [(jf, scen_idx) for jf, scen_idx in chunk_items]

        # ---- Multiprocessing pool ----
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

        # results is a list of tuples: (b_rows, u_rows, g_rows, y_rows)
        print("Aggregating chunk results...")
        all_branch = []
        all_bus = []
        all_gen = []
        all_ybus = []

        for r in tqdm(results, desc="Aggregating", leave=False):
            b_rows, u_rows, g_rows, y_rows = r
            all_branch.extend(b_rows)
            all_bus.extend(u_rows)
            all_gen.extend(g_rows)
            all_ybus.extend(y_rows)

        print("Sorting chunk dataframes...")

        def fast_df(rows: List[dict]) -> pd.DataFrame:
            if not rows:
                return pd.DataFrame()
            keys = rows[0].keys()
            return pd.DataFrame.from_records(rows, columns=keys)

        branch_df = fast_df(all_branch)
        bus_df = fast_df(all_bus)
        gen_df = fast_df(all_gen)
        ybus_df = fast_df(all_ybus)

        # Append to final parquets
        print("Appending chunk dataframes to final parquets...")
        append_df(branch_parquet, branch_df)
        append_df(bus_parquet, bus_df)
        append_df(gen_parquet, gen_df)
        append_df(ybus_parquet, ybus_df)


if __name__ == "__main__":
    main()
