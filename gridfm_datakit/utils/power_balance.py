"""Power balance computation utilities for power flow analysis."""

import numpy as np
import pandas as pd
from typing import Tuple, Union


def compute_branch_admittances(
    r: Union[float, np.ndarray],
    x: Union[float, np.ndarray],
    b: Union[float, np.ndarray],
    tap_mag: Union[float, np.ndarray],
    shift: Union[float, np.ndarray],
) -> Tuple[
    Union[complex, np.ndarray],
    Union[complex, np.ndarray],
    Union[complex, np.ndarray],
    Union[complex, np.ndarray],
]:
    """
    Compute branch admittances (Yff, Yft, Ytf, Ytt) from branch parameters.

    Implements the admittance matrix equations:
        Yff = (y_series + y_sh_f) / t2
        Yft = -y_series / tap.conjugate()
        Ytf = -y_series / tap
        Ytt = y_series + y_sh_t

    where:
        - y_series = 1/(r + jx) (series admittance)
        - y_sh_f, y_sh_t = shunt admittances (b/2 each side)
        - tap = tap_mag * exp(j*shift) (complex tap with phase shift)
        - t2 = |tap|^2 = tap_mag^2

    Args:
        r: Series resistance (ohms) - float or numpy array
        x: Series reactance (ohms) - float or numpy array
        b: Total shunt susceptance (split equally between both ends) - float or numpy array
        tap_mag: Tap magnitude (default: 1.0 for AC lines) - float or numpy array
        shift: Phase shift in radians (default: 0.0) - float or numpy array

    Returns:
        Tuple of (Yff, Yft, Ytf, Ytt) complex admittances
        - If inputs are scalars: returns scalars (complex)
        - If inputs are arrays: returns arrays (numpy.ndarray of complex)
    """
    # Convert to numpy arrays for vectorized operations
    r = np.asarray(r)
    x = np.asarray(x)
    b = np.asarray(b)
    tap_mag = np.asarray(tap_mag)
    shift = np.asarray(shift)

    # Calculate series admittance: y_series = 1/(r + jx)
    z_series = r + 1j * x
    y_series = 1.0 / z_series

    # Calculate tap with phase shift: tap = tap_mag * exp(j*shift)
    tap = tap_mag * (np.cos(shift) + 1j * np.sin(shift))
    t2 = tap_mag**2  # |tap|^2 = tap_mag^2

    # Calculate shunt admittances (split equally between both ends)
    y_sh = 1j * (b / 2.0)

    # Calculate admittance matrix elements
    Yff = (y_series + y_sh) / t2
    Yft = -y_series / np.conj(tap)
    Ytf = -y_series / tap
    Ytt = y_series + y_sh

    return Yff, Yft, Ytf, Ytt


def compute_branch_powers_vectorized(
    branch_df: pd.DataFrame,
    bus_df: pd.DataFrame,
    dc: bool,
    sn_mva: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute branch power flows for all branches in a vectorized fashion.

    Args:
        branch_df: DataFrame with branch data including Yff, Yft, Ytf, Ytt admittances
        bus_df: DataFrame with bus data including Vm and Va (or Va_dc for DC mode)
        dc: If True, use DC power flow (Va_dc, Vm=1.0), else use AC (Va, Vm)
        sn_mva: System base power in MVA used to scale complex power results

    Returns:
        Tuple of (pf, qf, pt, qt) power flow arrays in MW/MVAR
    """
    scenarios = branch_df["scenario"].to_numpy(dtype=int)
    from_bus = branch_df["from_bus"].to_numpy(dtype=int)
    to_bus = branch_df["to_bus"].to_numpy(dtype=int)

    idx_from = pd.MultiIndex.from_arrays(
        [scenarios, from_bus],
        names=["scenario", "bus"],
    )
    idx_to = pd.MultiIndex.from_arrays([scenarios, to_bus], names=["scenario", "bus"])

    bus_df_indexed = bus_df.set_index(["scenario", "bus"]).copy()
    Va = np.radians(bus_df_indexed["Va_dc" if dc else "Va"])
    Vm = 1.0 if dc else bus_df_indexed["Vm"]
    bus_df_indexed["V"] = Vm * (np.cos(Va) + 1j * np.sin(Va))
    Vf = bus_df_indexed["V"].loc[idx_from].to_numpy(dtype=np.complex128)
    Vt = bus_df_indexed["V"].loc[idx_to].to_numpy(dtype=np.complex128)

    Yff = branch_df["Yff_r"].to_numpy(dtype=np.float64) + 1j * branch_df[
        "Yff_i"
    ].to_numpy(dtype=np.float64)
    Yft = branch_df["Yft_r"].to_numpy(dtype=np.float64) + 1j * branch_df[
        "Yft_i"
    ].to_numpy(dtype=np.float64)
    Ytf = branch_df["Ytf_r"].to_numpy(dtype=np.float64) + 1j * branch_df[
        "Ytf_i"
    ].to_numpy(dtype=np.float64)
    Ytt = branch_df["Ytt_r"].to_numpy(dtype=np.float64) + 1j * branch_df[
        "Ytt_i"
    ].to_numpy(dtype=np.float64)

    If = Yff * Vf + Yft * Vt
    It = Ytt * Vt + Ytf * Vf

    Sf = Vf * np.conj(If) * sn_mva
    St = Vt * np.conj(It) * sn_mva

    pf = np.real(Sf)
    qf = np.imag(Sf)
    pt = np.real(St)
    qt = np.imag(St)

    return pf, qf, pt, qt


def compute_bus_balance(
    bus_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    flows: pd.DataFrame,
    dc: bool,
    sn_mva: float,
) -> pd.DataFrame:
    """
    Compute power balance at each bus for AC or DC mode.

    Balance equation: P_inj - P_out - P_sh = P_mis (and Q for AC)
    where:
    - P_inj: net injection (generation - demand)
    - P_out: total outgoing branch flows (from + to sides)
    - P_sh: shunt power consumption

    Args:
        bus_df: DataFrame with bus data (scenario, bus, Pg, Qg, Pd, Qd, GS, BS, Vm)
        branch_df: DataFrame with branch data (scenario, from_bus, to_bus)
        flows: DataFrame with flow data. For AC: columns (pf, qf, pt, qt). For DC: columns (pf_dc, pt_dc)
        dc: If True, compute DC balance, else AC balance
        sn_mva: System base power in MVA used to scale power terms

    Returns:
        DataFrame with columns [scenario, bus, P_mis_ac, Q_mis_ac] for AC or [scenario, bus, P_mis_dc] for DC
    """
    # ===== Step 1: Aggregate branch flows per bus =====
    # For each bus, sum flows where it is the "from" bus (pf, qf) and where it is the "to" bus (pt, qt)
    # For AC: include both active (pf, pt) and reactive (qf, qt) flows
    # For DC: only active flows (pf, pt)

    # Prepare bus mapping: from_bus -> bus and to_bus -> bus

    if not dc:
        from_bus = pd.concat(
            [branch_df[["scenario", "from_bus"]], flows[["pf", "qf"]]],
            axis=1,
        ).rename(columns={"from_bus": "bus"})
        to_bus = pd.concat(
            [branch_df[["scenario", "to_bus"]], flows[["pt", "qt"]]],
            axis=1,
        ).rename(columns={"to_bus": "bus"})
    else:
        from_bus = pd.concat(
            [branch_df[["scenario", "from_bus"]], flows[["pf_dc"]]],
            axis=1,
        ).rename(columns={"from_bus": "bus", "pf_dc": "pf"})
        to_bus = pd.concat(
            [branch_df[["scenario", "to_bus"]], flows[["pt_dc"]]],
            axis=1,
        ).rename(columns={"to_bus": "bus", "pt_dc": "pt"})

    # set int dtype for scenario and bus
    from_bus["scenario"] = from_bus["scenario"].astype(int)
    from_bus["bus"] = from_bus["bus"].astype(int)
    to_bus["scenario"] = to_bus["scenario"].astype(int)
    to_bus["bus"] = to_bus["bus"].astype(int)

    # Sum flows by (scenario, bus)
    out_from = from_bus.groupby(["scenario", "bus"], as_index=False).sum()
    out_to = to_bus.groupby(["scenario", "bus"], as_index=False).sum()

    # Combine from and to flows, then compute total power flow
    # Note: pf is power at "from" bus, pt is power at "to" bus (both positive when flowing from->to)
    out = out_from.merge(out_to, on=["scenario", "bus"], how="outer").fillna(0.0)
    out["P_out"] = out["pf"] + out["pt"]

    if not dc:
        out["Q_out"] = out["qf"] + out["qt"]
        out = out[["scenario", "bus", "P_out", "Q_out"]]
    else:
        out = out[["scenario", "bus", "P_out"]]

    # ===== Step 2: Compute shunt power consumption =====
    # Shunt power = G*|V|^2 (active) and -B*|V|^2 (reactive)
    # For DC: |V| = 1.0, for AC: |V| = Vm from solution
    absV2 = 1.0 if dc else bus_df["Vm"].to_numpy(dtype=np.float64) ** 2
    P_sh = bus_df["GS"].to_numpy(dtype=np.float64) * absV2 * sn_mva
    Q_sh = (
        -(bus_df["BS"].to_numpy(dtype=np.float64) * absV2 * sn_mva) if not dc else None
    )

    # ===== Step 3: Compute net injections =====
    # Net injection = generation - demand
    if dc:
        inj = bus_df[["scenario", "bus", "Pg_dc", "Pd"]].copy()
        inj["scenario"] = inj["scenario"].astype(int)
        inj["bus"] = inj["bus"].astype(int)
        inj["P_inj"] = inj["Pg_dc"].astype(np.float64) - inj["Pd"].astype(np.float64)
        inj["P_sh"] = P_sh
        inj = inj[["scenario", "bus", "P_inj", "P_sh"]]
    else:
        inj = bus_df[["scenario", "bus", "Pg", "Qg", "Pd", "Qd"]].copy()
        inj["scenario"] = inj["scenario"].astype(int)
        inj["bus"] = inj["bus"].astype(int)
        inj["P_inj"] = inj["Pg"].astype(np.float64) - inj["Pd"].astype(np.float64)
        inj["Q_inj"] = inj["Qg"].astype(np.float64) - inj["Qd"].astype(np.float64)
        inj["P_sh"] = P_sh
        inj["Q_sh"] = Q_sh
        inj = inj[["scenario", "bus", "P_inj", "Q_inj", "P_sh", "Q_sh"]]

    # ===== Step 4: Compute power balance mismatch =====
    # Mismatch = injection - outgoing flows - shunt consumption
    bal = inj.merge(out, on=["scenario", "bus"], how="left").fillna(0.0)
    if not dc:
        bal["Q_mis_ac"] = np.abs(bal["Q_inj"] - bal["Q_out"] - bal["Q_sh"])
        bal["P_mis_ac"] = np.abs(bal["P_inj"] - bal["P_out"] - bal["P_sh"])
        return bal[["scenario", "bus", "P_mis_ac", "Q_mis_ac"]]
    else:
        bal["P_mis_dc"] = np.abs(bal["P_inj"] - bal["P_out"] - bal["P_sh"])
        # assign nan to scenarios that had nan (it is enough to check Va_dc since data validation ensures that all DC columns are NaN for the same scenarios)
        scenarios_with_nan = set(bus_df[bus_df["Va_dc"].isna()]["scenario"].unique())
        bal.loc[bal["scenario"].isin(scenarios_with_nan), "P_mis_dc"] = np.nan
        return bal[["scenario", "bus", "P_mis_dc"]]
