"""Compare the two ways of setting slack Pg when scoring a DC solution in the AC model.

For each network we solve DC power flow once, then evaluate the same metric as
stats.py -- mean |P_mis_ac| over the buses of the scenario -- twice, changing only
how the slack bus generation is chosen:

  "lossless"  Pg_slack closes the balance against PowerModels' lossless DC flows
              (p = -b_series * dtheta).  This is what datakit does today via
              apply_slack_single_gen.  Because the residual is later scored with
              the full complex Y matrix, which keeps series resistance, the whole
              AC-vs-DC model difference lands on the slack bus.

  "full-Y"    Pg_slack closes the balance against the very flows the residual is
              scored with, so the slack residual is 0 by construction.

Everything else -- angles, loads, shunts, non-slack setpoints -- is identical
between the two, so the difference isolates the slack convention.
"""

from __future__ import annotations

import argparse

import juliacall  # noqa: F401  -- must be imported before torch/numpy-heavy stacks
import numpy as np
import pandas as pd

from gridfm_datakit.network import load_net_from_pglib
from gridfm_datakit.process.process_network import init_julia
from gridfm_datakit.process.solvers import run_dcpf, run_opf
from gridfm_datakit.utils import idx_brch as ib
from gridfm_datakit.utils import idx_bus as bs
from gridfm_datakit.utils import idx_gen as ig
from gridfm_datakit.utils.power_balance import (
    compute_branch_powers_vectorized,
    compute_bus_balance,
)

CASES = ["case118_ieee", "case2000_goc", "case10000_goc"]


def dispatch_with_opf(net, jl) -> None:
    """Replace the .m PG column with a solved AC-OPF dispatch, in place.

    The PG column of a PGLib .m file is an OPF *starting point*, not a dispatch:
    on these three cases total generation is off by -23%, -12% and +63% of load.
    Left alone the slack absorbs that fictitious imbalance, which swamps the
    AC-vs-DC model difference we want to measure.

    Simply scaling PG to match load is not enough either -- it ignores generator
    limits and pushes power across the network, which on case10000_goc produced a
    661-degree angle spread and 45.9 GW branch flows on a 73.7 GW system.  Solving
    OPF gives a dispatch that respects limits, so the DC solution is a sane
    operating point and the slack carries only the model difference.

    Note: the setpoints must be written to ``net.gens``, not ``net.mpc["gen"]``.
    Network.__init__ copies the mpc matrices into .buses/.gens/.branches and
    to_mpc serialises from those, so mutating mpc["gen"] is silently discarded.
    """
    result = run_opf(net, jl)
    base = net.baseMVA
    for key, vals in result["solution"]["gen"].items():
        # PowerModels keys generators by their 1-based row in the gen matrix.
        net.gens[int(key) - 1, ig.PG] = float(vals["pg"]) * base


def build_frames(net, va_deg: np.ndarray):
    """Build the bus/branch frames compute_bus_balance expects for one scenario."""
    # net.buses/gens/branches already carry 0-based bus indices (Network remaps
    # them on construction), so no id -> position lookup is needed here.
    base = net.baseMVA
    bus = np.asarray(net.buses, dtype=float)
    gen = np.asarray(net.gens, dtype=float)
    branch = np.asarray(net.branches, dtype=float)

    n_bus = len(bus)

    on = branch[:, ib.BR_STATUS] == 1
    br = branch[on]
    f = br[:, ib.F_BUS].astype(int)
    t = br[:, ib.T_BUS].astype(int)

    # Branch admittances, same construction as the dataset writer.
    r, x, b_c = br[:, ib.BR_R], br[:, ib.BR_X], br[:, ib.BR_B]
    tap = np.where(br[:, ib.TAP] == 0, 1.0, br[:, ib.TAP])
    shift = np.radians(br[:, ib.SHIFT])
    y = 1.0 / (r + 1j * x)
    tapc = tap * np.exp(1j * shift)
    Yff = (y + 1j * b_c / 2.0) / (tap**2)
    Yft = -y / np.conj(tapc)
    Ytf = -y / tapc
    Ytt = y + 1j * b_c / 2.0

    branch_df = pd.DataFrame(
        {
            "scenario": 0,
            "from_bus": f,
            "to_bus": t,
            "Yff_r": Yff.real,
            "Yff_i": Yff.imag,
            "Yft_r": Yft.real,
            "Yft_i": Yft.imag,
            "Ytf_r": Ytf.real,
            "Ytf_i": Ytf.imag,
            "Ytt_r": Ytt.real,
            "Ytt_i": Ytt.imag,
        },
    )

    gen_bus = gen[:, ig.GEN_BUS].astype(int)
    pg_bus = np.bincount(
        gen_bus,
        weights=gen[:, ig.PG] * (gen[:, ig.GEN_STATUS] == 1),
        minlength=n_bus,
    )
    qg_bus = np.bincount(
        gen_bus,
        weights=gen[:, ig.QG] * (gen[:, ig.GEN_STATUS] == 1),
        minlength=n_bus,
    )

    bus_df = pd.DataFrame(
        {
            "scenario": 0,
            "bus": np.arange(n_bus),
            "Pd": bus[:, bs.PD],
            "Qd": bus[:, bs.QD],
            "Pg": pg_bus,
            "Qg": qg_bus,
            "Vm": 1.0,
            "Va": va_deg,
            "GS": bus[:, bs.GS],
            "BS": bus[:, bs.BS],
        },
    )

    slack = int(np.argmax(bus[:, bs.BUS_TYPE] == bs.REF))
    return bus_df, branch_df, slack, base, (f, t), (r, x, tap, shift)


def lossless_dc_flows(va_deg, f, t, r, x, tap, shift, base):
    """PowerModels' lossless DC branch flows, in MW.

    compute_dc_pf uses p = -b_series * (theta_f - theta_t) with the series
    susceptance b_series = -x / (r^2 + x^2).  Verified against the pf values
    Julia reports, to 9e-16 p.u. on case118 -- note there is no tap division and
    no phase-shift term here, so do not "improve" this formula without
    re-checking it against the solver.
    """
    b_ser = -x / (r**2 + x**2)
    th = np.radians(va_deg)
    p = -b_ser * (th[f] - th[t])
    return p * base


def close_slack(bus_df, slack, f, t, pf, pt):
    """Set Pg at the slack so its active balance closes against the given flows."""
    n = len(bus_df)
    p_out = np.zeros(n)
    np.add.at(p_out, f, pf)
    np.add.at(p_out, t, pt)
    pg = bus_df["Pg"].to_numpy(dtype=float).copy()
    pg[slack] = (
        bus_df["Pd"].to_numpy()[slack] + bus_df["GS"].to_numpy()[slack] + p_out[slack]
    )
    return pg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", default=CASES)
    args = parser.parse_args()

    jl = init_julia(max_iter=1000)
    rows = []

    for name in args.cases:
        net = load_net_from_pglib(name)
        dispatch_with_opf(net, jl)
        n_bus = len(net.buses)

        result = run_dcpf(net, jl, fast=True)
        # to_mpc writes 1-based original bus ids, so map the solver's keys back.
        va = np.zeros(n_bus)
        for bus_id, vals in result["solution"]["bus"].items():
            va[net.bus_index_mapping[int(bus_id)]] = float(vals["va"])
        va_deg = np.degrees(va)

        bus_df, branch_df, slack, base, (f, t), (r, x, tap, shift) = build_frames(
            net,
            va_deg,
        )

        # Flows the residual is scored with (full complex Y, Vm = 1, Va = Va_dc).
        pf, qf, pt, qt = compute_branch_powers_vectorized(
            branch_df,
            bus_df,
            dc=False,
            sn_mva=base,
        )
        flows = pd.DataFrame(
            {"pf": pf, "qf": qf, "pt": pt, "qt": qt},
            index=branch_df.index,
        )

        # Flows datakit uses to close the slack today.
        p_lossless = lossless_dc_flows(va_deg, f, t, r, x, tap, shift, base)

        out = {"case": name, "n_bus": n_bus, "slack_bus": slack}
        for label, (a, b) in {
            "lossless": (p_lossless, -p_lossless),
            "full-Y": (pf, pt),
        }.items():
            df = bus_df.copy()
            df["Pg"] = close_slack(bus_df, slack, f, t, a, b)
            bal = compute_bus_balance(df, branch_df, flows, dc=False, sn_mva=base)
            mis = bal["P_mis_ac"].to_numpy(dtype=float)
            out[f"mean_{label}"] = mis.mean()
            out[f"slack_{label}"] = mis[slack]
        out["reduction_pct"] = 100 * (1 - out["mean_full-Y"] / out["mean_lossless"])
        rows.append(out)
        print(
            f"{name:15s} N={n_bus:6d}  mean lossless={out['mean_lossless']:9.4f}  "
            f"full-Y={out['mean_full-Y']:9.4f}  ({out['reduction_pct']:5.1f}% lower)  "
            f"slack: {out['slack_lossless']:9.3f} -> {out['slack_full-Y']:.2e}",
        )

    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
