import pandapower as pp
import pandas as pd
import numpy as np
from pandapower.auxiliary import pandapowerNet
from typing import Any, Tuple


def run_opf(net: pandapowerNet, **kwargs: Any) -> bool:
    """Runs Optimal Power Flow (OPF) and adds additional information to network elements.

    This function runs the OPF calculation and adds bus index and type information to the
    results dataframes. It also performs various validation checks on the results.

    Args:
        net: A pandapower network object containing the power system model.
        **kwargs: Additional keyword arguments to pass to pp.runopp().

    Returns:
        bool: True if the OPF converged successfully, False otherwise.

    Raises:
        AssertionError: If any of the validation checks fail, including:
            - Mismatch in active/reactive power
            - Power bounds violations
            - Bus power mismatches
            - Power balance violations
    """
    pp.runopp(net, numba=True, **kwargs)

    # add bus index and type to dataframe of opf results
    net.res_gen["bus"] = net.gen.bus
    net.res_gen["type"] = net.gen.type
    net.res_load["bus"] = net.load.bus
    net.res_load["type"] = net.load.type
    net.res_sgen["bus"] = net.sgen.bus
    net.res_sgen["type"] = net.sgen.type
    net.res_shunt["bus"] = net.shunt.bus
    net.res_ext_grid["bus"] = net.ext_grid.bus
    net.res_bus["type"] = net.bus["type"]

    # The load of course stays the same as before
    assert (net.load.p_mw == net.res_load.p_mw).all(), "Active power load has changed"
    assert (net.load.q_mvar == net.res_load.q_mvar).all(), (
        "Reactive power load has changed"
    )

    # Checking bounds on active and reactive power
    # TODO: see with matteo how we want to handle this
    if len(net.gen) != 0:
        in_service_gen = net.gen[net.gen.in_service]
        in_service_res_gen = net.res_gen[net.gen.in_service]

        if "max_p_mw" in net.gen.columns and "min_p_mw" in net.gen.columns:
            valid_gen = in_service_gen.dropna(
                subset=["max_p_mw", "min_p_mw"],
                how="any",
            )
            valid_res_gen = in_service_res_gen.loc[valid_gen.index]

            if not valid_gen.empty:
                assert (valid_gen.max_p_mw - valid_res_gen.p_mw > -1e-4).all(), (
                    f"Active power exceeds upper bound: "
                    f"{valid_res_gen.p_mw[valid_gen.max_p_mw - valid_res_gen.p_mw <= -1e-4]} exceeds "
                    f"{valid_gen.max_p_mw[valid_gen.max_p_mw - valid_res_gen.p_mw <= -1e-4]}"
                )
                assert (valid_res_gen.p_mw - valid_gen.min_p_mw > -1e-4).all(), (
                    f"Active power falls below lower bound: "
                    f"{valid_res_gen.p_mw[valid_res_gen.p_mw - valid_gen.min_p_mw <= -1e-4]} below "
                    f"{valid_gen.min_p_mw[valid_res_gen.p_mw - valid_gen.min_p_mw <= -1e-4]}"
                )

        if "max_q_mvar" in net.gen.columns and "min_q_mvar" in net.gen.columns:
            valid_q_gen = in_service_gen.dropna(
                subset=["max_q_mvar", "min_q_mvar"],
                how="any",
            )
            valid_q_res_gen = in_service_res_gen.loc[valid_q_gen.index]

            if not valid_q_gen.empty:
                assert (
                    valid_q_gen.max_q_mvar - valid_q_res_gen.q_mvar > -1e-4
                ).all(), (
                    f"Reactive power exceeds upper bound: "
                    f"{valid_q_res_gen.q_mvar[valid_q_gen.max_q_mvar - valid_q_res_gen.q_mvar <= -1e-4]} exceeds "
                    f"{valid_q_gen.max_q_mvar[valid_q_gen.max_q_mvar - valid_q_res_gen.q_mvar <= -1e-4]}"
                )
                assert (
                    valid_q_res_gen.q_mvar - valid_q_gen.min_q_mvar > -1e-4
                ).all(), (
                    f"Reactive power falls below lower bound: "
                    f"{valid_q_res_gen.q_mvar[valid_q_res_gen.q_mvar - valid_q_gen.min_q_mvar <= -1e-4]} below "
                    f"{valid_q_gen.min_q_mvar[valid_q_res_gen.q_mvar - valid_q_gen.min_q_mvar <= -1e-4]}"
                )

    if len(net.sgen) != 0:
        in_service_sgen = net.sgen[net.sgen.in_service]
        in_service_res_sgen = net.res_sgen[net.sgen.in_service]

        if "max_p_mw" in net.sgen.columns and "min_p_mw" in net.sgen.columns:
            valid_sgen = in_service_sgen.dropna(
                subset=["max_p_mw", "min_p_mw"],
                how="any",
            )
            valid_res_sgen = in_service_res_sgen.loc[valid_sgen.index]

            if not valid_sgen.empty:
                assert (valid_sgen.max_p_mw - valid_res_sgen.p_mw > -1e-4).all(), (
                    f"Active power exceeds upper bound for static generators: "
                    f"{valid_res_sgen.p_mw[valid_sgen.max_p_mw - valid_res_sgen.p_mw <= -1e-4]} exceeds "
                    f"{valid_sgen.max_p_mw[valid_sgen.max_p_mw - valid_res_sgen.p_mw <= -1e-4]}"
                )
                assert (valid_res_sgen.p_mw - valid_sgen.min_p_mw > -1e-4).all(), (
                    f"Active power falls below lower bound for static generators: "
                    f"{valid_res_sgen.p_mw[valid_res_sgen.p_mw - valid_sgen.min_p_mw <= -1e-4]} below "
                    f"{valid_sgen.min_p_mw[valid_res_sgen.p_mw - valid_sgen.min_p_mw <= -1e-4]}"
                )

        if "max_q_mvar" in net.sgen.columns and "min_q_mvar" in net.sgen.columns:
            valid_q_sgen = in_service_sgen.dropna(
                subset=["max_q_mvar", "min_q_mvar"],
                how="any",
            )
            valid_q_res_sgen = in_service_res_sgen.loc[valid_q_sgen.index]

            if not valid_q_sgen.empty:
                assert (
                    valid_q_sgen.max_q_mvar - valid_q_res_sgen.q_mvar > -1e-4
                ).all(), (
                    f"Reactive power exceeds upper bound for static generators: "
                    f"{valid_q_res_sgen.q_mvar[valid_q_sgen.max_q_mvar - valid_q_res_sgen.q_mvar <= -1e-4]} exceeds "
                    f"{valid_q_sgen.max_q_mvar[valid_q_sgen.max_q_mvar - valid_q_res_sgen.q_mvar <= -1e-4]}"
                )
                assert (
                    valid_q_res_sgen.q_mvar - valid_q_sgen.min_q_mvar > -1e-4
                ).all(), (
                    f"Reactive power falls below lower bound for static generators: "
                    f"{valid_q_res_sgen.q_mvar[valid_q_res_sgen.q_mvar - valid_q_sgen.min_q_mvar <= -1e-4]} below "
                    f"{valid_q_sgen.min_q_mvar[valid_q_res_sgen.q_mvar - valid_q_sgen.min_q_mvar <= -1e-4]}"
                )

    total_p_diff, total_q_diff = calculate_power_imbalance(net)
    assert np.abs(total_q_diff) < 1e-1, (
        f"Total reactive power imbalance in OPF: {total_q_diff}"
    )
    assert np.abs(total_p_diff) < 1e-1, (
        f"Total active power imbalance in OPF: {total_p_diff}"
    )

    return net.OPF_converged


def run_pf(net: pandapowerNet, **kwargs: Any) -> bool:
    """Runs Power Flow (PF) calculation and adds additional information to network elements.

    This function runs the power flow calculation and adds bus index and type information
    to the results dataframes. It also performs validation checks on the results.

    Args:
        net: A pandapower network object containing the power system model.
        **kwargs: Additional keyword arguments to pass to pp.runpp().

    Returns:
        bool: True if the power flow converged successfully, False otherwise.

    Raises:
        AssertionError: If any of the validation checks fail, including:
            - Bus power mismatches
            - Power balance violations
    """
    pp.runpp(net, **kwargs)

    # add bus number to df of opf results
    net.res_gen["bus"] = net.gen.bus
    net.res_gen["type"] = net.gen.type

    net.res_load["bus"] = net.load.bus
    net.res_load["type"] = net.load.type

    net.res_sgen["bus"] = net.sgen.bus
    net.res_sgen["type"] = net.sgen.type

    net.res_shunt["bus"] = net.shunt.bus
    net.res_ext_grid["bus"] = net.ext_grid.bus
    net.res_bus["type"] = net.bus["type"]

    total_p_diff, total_q_diff = calculate_power_imbalance(net)

    assert np.abs(total_q_diff) < 1e-2, (
        f"Total reactive power imbalance in PF: {total_q_diff}"
    )
    assert np.abs(total_p_diff) < 1e-2, (
        f"Total active power imbalance in PF: {total_p_diff}"
    )

    return net.converged


def run_dcpf(net: pandapowerNet, **kwargs: Any) -> bool:
    """Runs DC Power Flow (DCPF) calculation and adds additional information to network elements.

    This function runs the DC power flow calculation and adds bus index and type information
    to the results dataframes. It also performs validation checks on the results.

    Args:
        net: A pandapower network object containing the power system model.
        **kwargs: Additional keyword arguments to pass to pp.rundcpp().

    Returns:
        bool: True if the DC power flow converged successfully, False otherwise.

    Raises:
        AssertionError: If any of the validation checks fail, including:
            - Bus power mismatches
    """
    pp.rundcpp(net, **kwargs)

    # add bus number to df of opf results
    net.res_gen["bus"] = net.gen.bus
    net.res_gen["type"] = net.gen.type

    net.res_load["bus"] = net.load.bus
    net.res_load["type"] = net.load.type

    net.res_sgen["bus"] = net.sgen.bus
    net.res_sgen["type"] = net.sgen.type

    net.res_shunt["bus"] = net.shunt.bus
    net.res_ext_grid["bus"] = net.ext_grid.bus
    net.res_bus["type"] = net.bus["type"]

    total_p_diff, total_q_diff = calculate_power_imbalance(net)

    print(
        "Total reactive power imbalance in DCPF: ",
        total_q_diff,
        " (It is normal that this is not 0 as we are using a DC model)",
    )
    print(
        "Total active power imbalance in DCPF: ",
        total_p_diff,
        " (Should be close to 0)",
    )

    return net.converged


def calculate_power_imbalance(net: pandapowerNet) -> Tuple[float, float]:
    """Calculates the power imbalance in the network normalized by the number of buses.

    Args:
        net: A pandapower network object containing the power system model.

    Returns:
        Tuple containing the total active and reactive power imbalance normalized by the number of buses.
    """
    # check net power at each bus
    num_buses = len(net.bus)
    all_gens = (
        pd.concat([net.res_gen, net.res_sgen, net.res_ext_grid])[
            ["p_mw", "q_mvar", "bus"]
        ]
        .groupby("bus")
        .sum()
    )
    all_loads = (
        pd.concat([net.res_load, net.res_shunt])[["p_mw", "q_mvar", "bus"]]
        .groupby("bus")
        .sum()
    )  # all load
    net_load = (
        pd.concat([all_loads, -all_gens])
        .groupby("bus")
        .sum()
        .reindex_like(net.res_bus[["p_mw", "q_mvar"]])
        .fillna(0)
    )  # net load = load - generation

    assert np.allclose(net.res_bus[["p_mw", "q_mvar"]], net_load), (
        f"Mismatch between net load stored in res_bus and the net load computed by summing up the load and generation after solving DCPF: {net.res_bus[['p_mw', 'q_mvar']] - net_load}"
    )

    # check power balance taking into account tansformer and line losses
    total_q_diff = (
        net_load.q_mvar.sum() + net.res_line.ql_mvar.sum() + net.res_trafo.ql_mvar.sum()
    )
    total_p_diff = (
        net_load.p_mw.sum() + net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum()
    )

    return total_p_diff / num_buses, total_q_diff / num_buses
