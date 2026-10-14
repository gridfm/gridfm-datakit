"""Summary metrics computed from a generated ``raw/`` data directory.

Every metric is a scalar aggregate over all scenarios, so a regression anywhere in
the pipeline (load scenarios, perturbations, solvers, saving) moves at least one of
them. Solver runtimes are deliberately excluded: they are not reproducible.
"""

import os

import numpy as np
import pandas as pd

from gridfm_datakit.utils.stats import compute_stats_from_data
from gridfm_datakit.utils.utils import get_num_scenarios


def _read(raw_dir: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(raw_dir, f"{name}.parquet"))


def _per_scenario_mean(df: pd.DataFrame, column: str) -> float:
    return float(df.groupby("scenario")[column].sum().mean())


def compute_metrics(raw_dir: str, sn_mva: float = 100.0) -> dict:
    """Return ``{metric_name: float}`` for the data in ``raw_dir``."""
    bus = _read(raw_dir, "bus_data")
    branch = _read(raw_dir, "branch_data")
    gen = _read(raw_dir, "gen_data")
    active_gen = gen[gen["in_service"] == 1]
    active_branch = branch[branch["br_status"] == 1]

    gen_cost = (
        active_gen["cp0_eur"]
        + active_gen["cp1_eur_per_mw"] * active_gen["p_mw"]
        + active_gen["cp2_eur_per_mw2"] * active_gen["p_mw"] ** 2
    )
    total_pd = bus.groupby("scenario")["Pd"].sum()
    total_pg = active_gen.groupby("scenario")["p_mw"].sum()

    metrics = {
        # Size of the dataset
        "n_scenarios": float(get_num_scenarios(raw_dir)),
        "n_bus_rows": float(len(bus)),
        "n_branch_rows": float(len(branch)),
        "n_gen_rows": float(len(gen)),
        "n_load_scenarios": float(bus["load_scenario_idx"].nunique()),
        # Loads and dispatch
        "mean_total_pd_mw": float(total_pd.mean()),
        "mean_total_qd_mvar": _per_scenario_mean(bus, "Qd"),
        "mean_total_pg_mw": float(total_pg.mean()),
        "mean_total_qg_mvar": _per_scenario_mean(active_gen, "q_mvar"),
        "mean_total_losses_mw": float((total_pg - total_pd).mean()),
        "mean_total_gen_cost": float(
            gen_cost.groupby(active_gen["scenario"]).sum().mean(),
        ),
        # Voltages
        "mean_vm_pu": float(bus["Vm"].mean()),
        "min_vm_pu": float(bus["Vm"].min()),
        "max_vm_pu": float(bus["Vm"].max()),
        "mean_abs_va_deg": float(bus["Va"].abs().mean()),
        # Topology
        "mean_active_gens": float(active_gen.groupby("scenario").size().mean()),
        "mean_active_branches": float(active_branch.groupby("scenario").size().mean()),
        # Branch flows
        "mean_abs_pf_mw": float(active_branch["pf"].abs().mean()),
        "mean_abs_qf_mvar": float(active_branch["qf"].abs().mean()),
    }

    if "Va_dc" in bus.columns:
        metrics["mean_abs_va_ac_minus_dc_deg"] = float(
            (bus["Va"] - bus["Va_dc"]).abs().mean(),
        )
        metrics["mean_total_pg_dc_mw"] = _per_scenario_mean(
            active_gen,
            "p_mw_dc",
        )

    s = compute_stats_from_data(raw_dir, sn_mva=sn_mva)
    metrics.update(
        {
            "mean_max_branch_loading": float(np.mean(s["max_loading"])),
            "max_branch_loading": float(np.max(s["max_loading"])),
            "mean_branch_loading": float(np.mean(s["branch_loadings"])),
            "mean_n_overloads": float(np.mean(s["n_overloads"])),
            # AC balance errors are numerical noise (~1e-8 MW or less), so they are
            # checked against fixed physical limits rather than calibrated bounds.
            "max_p_balance_error_ac_mw": float(np.max(s["p_balance_error_ac_max"])),
            "max_q_balance_error_ac_mvar": float(np.max(s["q_balance_error_ac_max"])),
        },
    )
    if "p_balance_error_dc_mean" in s:
        metrics["mean_p_balance_error_dc_mw"] = float(
            np.mean(s["p_balance_error_dc_mean"]),
        )
    return metrics


# Metrics that are pure numerical noise and must not be calibrated: a relative pad
# on a 1e-9 mean is meaningless. They are asserted against PHYSICAL_RANGES instead.
NOISE_METRICS = {"max_p_balance_error_ac_mw", "max_q_balance_error_ac_mvar"}


def physical_ranges(mode: str, expected_scenarios: int) -> dict:
    """Fixed validity ranges that must hold for any correct data, baseline or not.

    These are deliberately loose: they catch broken physics, not drift.
    """
    inf = float("inf")
    ranges = {
        "max_p_balance_error_ac_mw": (0.0, 1e-3),
        "max_q_balance_error_ac_mvar": (0.0, 1e-3),
        "min_vm_pu": (0.5, inf),
        "max_vm_pu": (-inf, 1.5),
        "mean_total_pd_mw": (0.0, inf),
        "mean_total_losses_mw": (0.0, inf),
        "mean_active_gens": (1.0, inf),
        "mean_active_branches": (1.0, inf),
    }
    if mode == "pf":
        # The PF solver should converge for (nearly) every scenario.
        ranges["n_scenarios"] = (0.95 * expected_scenarios, expected_scenarios)
    else:
        # OPF drops infeasible scenarios; most of them should still be feasible, and
        # every kept one must respect branch limits (stats count loading > 1.01).
        ranges["n_scenarios"] = (0.5 * expected_scenarios, expected_scenarios)
        ranges["mean_n_overloads"] = (0.0, 0.0)
        ranges["max_branch_loading"] = (0.0, 1.01)
        ranges["mean_total_gen_cost"] = (0.0, inf)
    return ranges
