"""Score DC power flow solutions in the AC model, for every .pt scenario.

WHAT THIS DOES
--------------
For each data_index_<i>.pt in a processed dataset directory:

  1. rebuild a MATPOWER case from the stored tensors,
  2. solve DC power flow with run_dcpf(..., fast=True),
  3. compute the per-bus active power mismatch |P_mis_ac| in the *AC* model,
     evaluated at the DC operating point (Vm = 1.0, Va = Va_dc),
  4. average it over the buses of the scenario -- the same reduction as
     stats.py: group_by_scenario["P_mis_ac"].mean().reindex(scenarios).

Everything is derived from the .pt files.  No .m case file and no parquet is read,
so the numbers describe exactly the data your model was trained on.

THE TWO SLACK CONVENTIONS
-------------------------
A DC power flow does not enforce the slack bus balance -- the slack is the free
variable that absorbs whatever the rest of the system does not supply.  So its
generation has to be reconstructed afterwards, and there are two ways to do it.
Both are reported, as two columns:

  mean_p_residual_lossless
      Pg_slack closes the balance against PowerModels' *lossless* DC flows
      (p = -b_series * dtheta, so pf + pt = 0 on every branch).  This is what
      datakit does today, via apply_slack_single_gen.  The residual, however, is
      scored with the full complex Y matrix, which keeps series resistance -- so
      the whole model difference lands on the slack bus as a spurious residual.

  mean_p_residual_full_y
      Pg_slack closes the balance against the very flows the residual is scored
      with.  The slack residual is then 0 by construction.

Everything else -- angles, loads, shunts, non-slack setpoints -- is identical
between the two columns, so their difference isolates the slack convention alone.
Prefer full_y when comparing a neural solver against DC: it removes an artifact
that is pure bookkeeping, and it costs microseconds.


USAGE
-----
    python dc_benchmark/compute_dc_residuals.py \
        --processed-dir dc_benchmark/data/case118_ieee/processed

    # quick smoke test on the first 20 scenarios
    python dc_benchmark/compute_dc_residuals.py --limit 20

Outputs, written to --output-dir (defaults to --processed-dir):

    dc_residuals.csv        pt_index, mean_p_residual_lossless, mean_p_residual_full_y
    dc_residuals_stats.txt  summary statistics over both columns
"""

from __future__ import annotations

import argparse
import os
import re

import juliacall  # noqa: F401  -- must be imported before torch, or the process segfaults
import numpy as np
import pandas as pd
import torch

from gridfm_datakit.network import Network
from gridfm_datakit.process.process_network import init_julia
from gridfm_datakit.process.solvers import run_dcpf
from gridfm_datakit.utils import idx_brch as ib
from gridfm_datakit.utils import idx_bus as bs
from gridfm_datakit.utils import idx_gen as ig
from gridfm_datakit.utils.idx_cost import POLYNOMIAL
from gridfm_datakit.utils.power_balance import (
    compute_branch_powers_vectorized,
    compute_bus_balance,
)

# Column layout of the stored tensors (see gridfm_graphkit.datasets.globals).
PD_H, QD_H, QG_H, VM_H, VA_H, PQ_H, PV_H, REF_H = range(8)
MIN_VM_H, MAX_VM_H, MIN_QG_H, MAX_QG_H, GS_H, BS_H, VN_KV_H = range(8, 15)
PG_H, MIN_PG_H, MAX_PG_H, C0_H, C1_H, C2_H, G_ON_H = range(7)
P_E, Q_E, YFF_R, YFF_I, YFT_R, YFT_I, TAP_E, ANGMIN_E, ANGMAX_E, RATE_A_E, B_ON_E = (
    range(11)
)

SN_MVA = 100.0  # gridfm datasets are generated at baseMVA = 100


def load_pt(path: str) -> dict:
    """Load one .pt scenario file into plain numpy arrays."""
    d = torch.load(path, weights_only=True)
    edge = d[("bus", "connects", "bus")]
    # Each branch is stored twice: forward rows first, then reverse rows.
    n_edges = edge["edge_attr"].shape[0] // 2
    return {
        "bus": d["bus"]["x"].numpy().astype(float),
        "gen": d["gen"]["x"].numpy().astype(float),
        "gen_bus": d[("gen", "connected_to", "bus")]["edge_index"][1].numpy(),
        "edge_index": edge["edge_index"].numpy(),
        "edge_attr": edge["edge_attr"].numpy().astype(float),
        "n_edges": n_edges,
    }


def branch_rxb(fwd: np.ndarray, rev: np.ndarray) -> tuple:
    """Recover (r, x, b, tap, shift_deg) from the stored branch admittances.

    Inverts branch_vectors (gridfm_datakit/network.py:766): with tap = |tap| * exp(j*shift),
        Yft = -y / conj(tap),  Ytf = -y / tap,  Ytt = y + j*b/2
    so Yft/Ytf = exp(2j*shift) gives the shift, and y and b follow from there.

    Round-trip verified against branch_vectors on 60 case118 scenarios: Yft, Ytf and
    Ytt agree to 1.5e-14 p.u.  Yff agrees to 2.3e-6, entirely because the stored
    tensors are float32 -- TAP_E holds 0.96 as 0.95999998, and Yff divides by tap^2,
    so that ~2e-8 relative error is doubled and scaled by |Ysf| ~ 50 p.u. on
    transformer branches.  Only the 9 branches with tap != 1 are affected.  Harmless
    at MW-scale residuals, but it is float32, not an algebra error.

    LIMITATION -- symmetric series impedance only.  branch_vectors supports
    asymmetric impedance (BR_R_ASYM/BR_X_ASYM, cols 21/22), where the from- and
    to-side series admittances differ (Yst != Ysf).  This inversion recovers a single
    y from Ytf and so cannot represent that; on such a branch it would silently
    return wrong r/x.  Asymmetry shows up as |Yft| != |Ytf|, which assert_symmetric
    below rejects rather than letting it pass quietly.
    """
    Yft = fwd[:, YFT_R] + 1j * fwd[:, YFT_I]
    Ytf = rev[:, YFT_R] + 1j * rev[:, YFT_I]
    Ytt = rev[:, YFF_R] + 1j * rev[:, YFF_I]
    tap = fwd[:, TAP_E]

    shift = np.angle(Yft / Ytf) / 2.0
    y = -Ytf * (tap * np.exp(1j * shift))
    z = 1.0 / y
    b = 2.0 * (Ytt - y).imag
    return z.real, z.imag, b, tap, np.degrees(shift)


def assert_symmetric(fwd: np.ndarray, rev: np.ndarray) -> None:
    """Fail loudly if any in-service branch has asymmetric series impedance.

    branch_rxb assumes Yst == Ysf.  Asymmetry makes |Yft| != |Ytf|; the tolerance is
    loose enough to absorb float32 storage error (~1e-6 relative) and tight enough to
    catch real asymmetry.  Measured exactly 0.0 on case118, so this never fires there.
    """
    on = fwd[:, B_ON_E] == 1
    if not on.any():
        return
    a = np.abs(fwd[on, YFT_R] + 1j * fwd[on, YFT_I])
    b = np.abs(rev[on, YFT_R] + 1j * rev[on, YFT_I])
    rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-30)
    if rel.max() > 1e-4:
        raise ValueError(
            f"asymmetric series impedance on {int((rel > 1e-4).sum())} branch(es) "
            f"(max relative |Yft|-|Ytf| mismatch {rel.max():.3e}). branch_rxb cannot "
            "invert these; the recovered r/x would be wrong. Extend branch_rxb to "
            "handle BR_R_ASYM/BR_X_ASYM before using this script on such a dataset.",
        )


def build_network(scen: dict) -> Network:
    """Build a Network (MATPOWER case) from one scenario's tensors."""
    bus_x, gen_x = scen["bus"], scen["gen"]
    n_bus, n_gen = len(bus_x), len(gen_x)
    fwd = scen["edge_attr"][: scen["n_edges"]]
    rev = scen["edge_attr"][scen["n_edges"] :]
    from_bus = scen["edge_index"][0, : scen["n_edges"]]
    to_bus = scen["edge_index"][1, : scen["n_edges"]]

    bus = np.zeros((n_bus, bs.bus_cols))
    bus[:, bs.BUS_I] = np.arange(1, n_bus + 1)  # MATPOWER bus ids are 1-based
    bus[:, bs.BUS_TYPE] = np.select(
        [bus_x[:, REF_H] == 1, bus_x[:, PV_H] == 1],
        [bs.REF, bs.PV],
        default=bs.PQ,
    )
    bus[:, bs.PD] = bus_x[:, PD_H]
    bus[:, bs.QD] = bus_x[:, QD_H]
    bus[:, bs.GS] = bus_x[:, GS_H]
    bus[:, bs.BS] = bus_x[:, BS_H]
    bus[:, bs.BUS_AREA] = 1
    bus[:, bs.VM] = bus_x[:, VM_H]
    bus[:, bs.VA] = bus_x[:, VA_H]
    bus[:, bs.BASE_KV] = bus_x[:, VN_KV_H]
    bus[:, bs.ZONE] = 1
    bus[:, bs.VMAX] = bus_x[:, MAX_VM_H]
    bus[:, bs.VMIN] = bus_x[:, MIN_VM_H]

    gen = np.zeros((n_gen, ig.gen_cols))
    gen[:, ig.GEN_BUS] = scen["gen_bus"] + 1
    gen[:, ig.PG] = gen_x[:, PG_H]
    # Qg and the Q limits are stored per bus, so spread each bus value over its gens.
    gen[:, ig.QG] = bus_x[scen["gen_bus"], QG_H]
    gen[:, ig.QMAX] = bus_x[scen["gen_bus"], MAX_QG_H]
    gen[:, ig.QMIN] = bus_x[scen["gen_bus"], MIN_QG_H]
    gen[:, ig.VG] = bus_x[scen["gen_bus"], VM_H]
    gen[:, ig.MBASE] = SN_MVA
    gen[:, ig.GEN_STATUS] = gen_x[:, G_ON_H]
    gen[:, ig.PMAX] = gen_x[:, MAX_PG_H]
    gen[:, ig.PMIN] = gen_x[:, MIN_PG_H]

    assert_symmetric(fwd, rev)
    r, x, b, tap, shift = branch_rxb(fwd, rev)
    branch = np.zeros((scen["n_edges"], ib.branch_cols))
    branch[:, ib.F_BUS] = from_bus + 1
    branch[:, ib.T_BUS] = to_bus + 1
    branch[:, ib.BR_R] = r
    branch[:, ib.BR_X] = x
    branch[:, ib.BR_B] = b
    branch[:, ib.RATE_A] = fwd[:, RATE_A_E]
    branch[:, ib.TAP] = tap
    branch[:, ib.SHIFT] = shift
    branch[:, ib.BR_STATUS] = fwd[:, B_ON_E]
    branch[:, ib.ANGMIN] = fwd[:, ANGMIN_E]
    branch[:, ib.ANGMAX] = fwd[:, ANGMAX_E]

    gencost = np.zeros((n_gen, 7))
    gencost[:, 0] = POLYNOMIAL
    gencost[:, 3] = 3  # NCOST: three coefficients
    gencost[:, 4] = gen_x[:, C2_H]
    gencost[:, 5] = gen_x[:, C1_H]
    gencost[:, 6] = gen_x[:, C0_H]

    return Network(
        {
            "baseMVA": SN_MVA,
            "bus": bus,
            "gen": gen,
            "branch": branch,
            "gencost": gencost,
        },
    )


def dc_angles(result: dict, n_bus: int) -> np.ndarray:
    """Read DC bus angles out of a run_dcpf result, in degrees."""
    va = np.zeros(n_bus)
    for bus_id, vals in result["solution"]["bus"].items():
        va[int(bus_id) - 1] = float(vals["va"])
    return np.degrees(va)


def dc_flows(result: dict, n_edges: int) -> tuple[np.ndarray, np.ndarray]:
    """Read the lossless DC branch flows out of a run_dcpf result, in MW.

    These are PowerModels' own p = -b_series * dtheta flows, so pf + pt is exactly
    0.0 on every in-service branch.  Same access pattern as
    gridfm_datakit/process/process_network.py:553-566.

    build_network writes the full branch matrix to the .m file (Network.to_mpc does
    not filter by status), so the solver keys branches by their 1-based row in that
    matrix and every branch gets a key.  Out-of-service branches come back as NaN,
    not 0, so they are zeroed here -- otherwise a single one would propagate through
    the slack summation and turn the whole scenario's mean into NaN.
    """
    pf = np.array(
        [float(result["solution"]["branch"][str(i + 1)]["pf"]) for i in range(n_edges)],
    )
    pt = np.array(
        [float(result["solution"]["branch"][str(i + 1)]["pt"]) for i in range(n_edges)],
    )
    out = ~np.isfinite(pf) | ~np.isfinite(pt)
    pf[out] = 0.0
    pt[out] = 0.0
    return pf * SN_MVA, pt * SN_MVA


def slack_generation(scen: dict, pf: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """Per-bus active generation in MW, with the slack closed against pf/pt.

    Non-slack generators keep their setpoints: a power flow does not redispatch,
    so the values stored in the .pt are exactly what DC-PF was handed and held
    fixed.  Only the slack floats, and its balance equation is the one DC-PF
    leaves unenforced, so we close it here:

        Pg_slack = Pd + P_shunt + sum(branch flows out of the slack bus)

    Which flows you pass in is exactly the choice between the two conventions
    described in the module docstring.
    """
    bus_x = scen["bus"]
    from_bus = scen["edge_index"][0, : scen["n_edges"]]
    to_bus = scen["edge_index"][1, : scen["n_edges"]]

    p_out = np.zeros(len(bus_x))
    np.add.at(p_out, from_bus, pf)
    np.add.at(p_out, to_bus, pt)

    pg = np.bincount(
        scen["gen_bus"],
        weights=scen["gen"][:, PG_H] * scen["gen"][:, G_ON_H],
        minlength=len(bus_x),
    )

    # GS is in MW at |V| = 1 p.u., and the DC operating point has |V| = 1.
    slack = int(np.argmax(bus_x[:, REF_H]))
    pg[slack] = (bus_x[:, PD_H] + bus_x[:, GS_H] + p_out)[slack]
    return pg


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        default=os.path.join(here, "data", "case118_ieee", "processed"),
        help="Directory containing data_index_<i>.pt files.",
    )
    parser.add_argument("--output-dir", default=None, help="Defaults to --processed-dir.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N scenarios (0 = all).",
    )
    args = parser.parse_args()

    out_dir = args.output_dir or args.processed_dir
    os.makedirs(out_dir, exist_ok=True)

    indices = sorted(
        int(m.group(1))
        for m in (
            re.match(r"^data_index_(\d+)\.pt$", f) for f in os.listdir(args.processed_dir)
        )
        if m
    )
    if args.limit:
        indices = indices[: args.limit]
    print(f"Found {len(indices)} scenario files in {args.processed_dir}")

    jl = init_julia(max_iter=1000)

    rows, failed = [], []
    n_off_total = 0  # out-of-service branches seen, reported in the stats file

    for n, idx in enumerate(indices, start=1):
        scen = load_pt(os.path.join(args.processed_dir, f"data_index_{idx}.pt"))
        n_bus = len(scen["bus"])

        try:
            result = run_dcpf(build_network(scen), jl, fast=True)
            va_dc = dc_angles(result, n_bus)
            pf_dc, pt_dc = dc_flows(result, scen["n_edges"])
        except Exception as exc:
            print(f"  scenario {idx}: DC-PF failed ({exc})")
            failed.append(idx)
            rows.append({"pt_index": idx, "lossless": np.nan, "full_y": np.nan})
            continue

        # One scenario at a time: the utils are vectorised over a "scenario"
        # column, and a single scenario keeps the bookkeeping obvious.
        fwd = scen["edge_attr"][: scen["n_edges"]].copy()
        rev = scen["edge_attr"][scen["n_edges"] :].copy()

        # Zero the admittances of out-of-service branches.
        # compute_branch_powers_vectorized works purely from the Y columns and never
        # looks at branch status, so a disabled branch would otherwise still be given
        # a full-Y flow (measured: 7.5 MW on case118) even though the DC solver
        # correctly ignored it.  Zeroing Y here makes its flow identically 0, which
        # matches deleting the branch outright.
        off = fwd[:, B_ON_E] != 1
        n_off_total += int(off.sum())
        if off.any():
            for col in (YFF_R, YFF_I, YFT_R, YFT_I):
                fwd[off, col] = 0.0
                rev[off, col] = 0.0

        branch_df = pd.DataFrame(
            {
                "scenario": idx,
                "from_bus": scen["edge_index"][0, : scen["n_edges"]],
                "to_bus": scen["edge_index"][1, : scen["n_edges"]],
                "Yff_r": fwd[:, YFF_R],
                "Yff_i": fwd[:, YFF_I],
                "Yft_r": fwd[:, YFT_R],
                "Yft_i": fwd[:, YFT_I],
                "Ytf_r": rev[:, YFT_R],
                "Ytf_i": rev[:, YFT_I],
                "Ytt_r": rev[:, YFF_R],
                "Ytt_i": rev[:, YFF_I],
            },
        )
        bus_df = pd.DataFrame(
            {
                "scenario": idx,
                "bus": np.arange(n_bus),
                "Pd": scen["bus"][:, PD_H],
                "Qd": scen["bus"][:, QD_H],
                "Qg": scen["bus"][:, QG_H],
                "Vm": 1.0,  # the DC operating point is flat-voltage
                "Va": va_dc,
                "GS": scen["bus"][:, GS_H],
                "BS": scen["bus"][:, BS_H],
            },
        )

        # Flows the residual is scored with: full complex Y at Vm = 1, Va = Va_dc.
        # These keep series resistance, so pf + pt != 0.
        pf, qf, pt, qt = compute_branch_powers_vectorized(
            branch_df,
            bus_df,
            dc=False,
            sn_mva=SN_MVA,
        )
        flows = pd.DataFrame(
            {"pf": pf, "qf": qf, "pt": pt, "qt": qt},
            index=branch_df.index,
        )

        means = {}
        for label, (a, b) in {
            # pf_dc/pt_dc come straight from the solver: the lossless flows datakit
            # currently closes the slack against.
            "lossless": (pf_dc, pt_dc),
            "full_y": (pf, pt),
        }.items():
            df = bus_df.copy()
            df["Pg"] = slack_generation(scen, a, b)
            balance = compute_bus_balance(df, branch_df, flows, dc=False, sn_mva=SN_MVA)
            means[label] = balance["P_mis_ac"].mean()

        rows.append({"pt_index": idx, **means})

        if n % 50 == 0 or n == len(indices):
            print(f"  solved {n}/{len(indices)}")

    df_out = (
        pd.DataFrame(rows)
        .rename(
            columns={
                "lossless": "mean_p_residual_lossless",
                "full_y": "mean_p_residual_full_y",
            },
        )
        .sort_values("pt_index")
    )

    csv_path = os.path.join(out_dir, "dc_residuals.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\nWrote {len(df_out)} rows to {csv_path}")

    lines = [
        "DC-PF mean active power residual statistics",
        "=" * 44,
        "",
        f"Data          : {args.processed_dir}",
        "Solver        : run_dcpf(fast=True)",
        "Residual      : |P_mis_ac| on the DC solution (Vm=1.0, Va=Va_dc), MW",
        "Per scenario  : mean over all buses",
        "",
        f"Scenarios     : {len(df_out)}",
        f"Failed        : {len(failed)}",
        f"Branches off  : {n_off_total} (summed over scenarios; their flows are "
        "forced to 0)",
        "",
    ]

    for col, title in [
        ("mean_p_residual_lossless", "lossless slack (what datakit does today)"),
        ("mean_p_residual_full_y", "full-Y slack (slack residual is 0)"),
    ]:
        vals = df_out[col].to_numpy(dtype=float)
        ok = vals[np.isfinite(vals)]
        lines += [title, "-" * 44]
        if len(ok):
            lines += [
                f"mean of means : {ok.mean():.6e}",
                f"std           : {ok.std():.6e}",
                f"min           : {ok.min():.6e}",
                f"median        : {np.median(ok):.6e}",
                f"95th pct      : {np.percentile(ok, 95):.6e}",
                f"max           : {ok.max():.6e}",
                "",
            ]
        else:
            lines += ["No finite residuals -- every scenario failed.", ""]

    a = df_out["mean_p_residual_lossless"].to_numpy(dtype=float)
    b = df_out["mean_p_residual_full_y"].to_numpy(dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.any():
        lines += [
            "Effect of the slack convention",
            "-" * 44,
            f"mean reduction : {100 * (1 - b[m].mean() / a[m].mean()):.2f}%",
            f"median per-scenario reduction : {100 * np.median(1 - b[m] / a[m]):.2f}%",
            "",
        ]
    if failed:
        lines += [f"Failed indices: {failed}"]

    txt_path = os.path.join(out_dir, "dc_residuals_stats.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote statistics to {txt_path}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
