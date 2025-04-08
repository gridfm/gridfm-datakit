import numpy as np
import pandas as pd
from GridDataGen.utils.config import *
from pandapower.auxiliary import pandapowerNet
from typing import Tuple
from pandapower import makeYbus_pypower
import copy
from GridDataGen.utils.solvers import *


def network_preprocessing(net: pandapowerNet):
    """
    Adds names to bus dataframe and bus types to load, bus, gen, sgen dataframes.
    """

    # Clean-Up things in Data-Frame // give numbered item names
    for i, row in net.bus.iterrows():
        net.bus.at[i, "name"] = "Bus " + str(i)
    for i, row in net.load.iterrows():
        net.load.at[i, "name"] = "Load " + str(i)
    for i, row in net.sgen.iterrows():
        net.sgen.at[i, "name"] = "Sgen " + str(i)
    for i, row in net.gen.iterrows():
        net.gen.at[i, "name"] = "Gen " + str(i)
    for i, row in net.shunt.iterrows():
        net.shunt.at[i, "name"] = "Shunt " + str(i)
    for i, row in net.ext_grid.iterrows():
        net.ext_grid.at[i, "name"] = "Ext_Grid " + str(i)
    for i, row in net.line.iterrows():
        net.line.at[i, "name"] = "Line " + str(i)
    for i, row in net.trafo.iterrows():
        net.trafo.at[i, "name"] = "Trafo " + str(i)

    num_buses = len(net.bus)
    bus_types = np.zeros(num_buses, dtype=int)

    indices_slack = np.unique(np.array(net.ext_grid["bus"]))

    indices_PV = np.union1d(
        np.unique(np.array(net.sgen["bus"])), np.unique(np.array(net.gen["bus"]))
    )
    indices_PV = np.setdiff1d(
        indices_PV, indices_slack
    )  # Exclude slack indices from PV indices

    indices_PQ = np.setdiff1d(
        np.arange(num_buses), np.union1d(indices_PV, indices_slack)
    )

    bus_types[indices_PQ] = PQ  # Set PV bus types to 2
    bus_types[indices_PV] = PV  # Set PV bus types to 2
    bus_types[indices_slack] = REF  # Set Slack bus types to 3

    net.bus["type"] = bus_types
    # assign type of the bus connected to each load and generator
    # Pandapower doesnt use node types, but we need to order the different nodes by types in the data matrices that we gonna build
    net.load["type"] = net.bus.type[net.load.bus].to_list()
    net.gen["type"] = net.bus.type[net.gen.bus].to_list()
    net.sgen["type"] = net.bus.type[net.sgen.bus].to_list()

    # there is no more than one load per bus:
    assert net.load.bus.unique().shape[0] == net.load.bus.shape[0]

    # REF bus is bus with ext grid:
    assert (
        np.where(net.bus["type"] == REF)[0]  # REF bus indicated by case file
        == net.ext_grid.bus.values
    ).all()  # Buses connected to an ext grid

    # PQ buses are buses with no gen nor ext_grid, only load or nothing connected to them
    assert (
        (net.bus["type"] == PQ)  # PQ buses indicated by case file
        == ~np.isin(
            range(net.bus.shape[0]),
            np.concatenate(
                [net.ext_grid.bus.values, net.gen.bus.values, net.sgen.bus.values]
            ),
        )
    ).all()  # Buses which are NOT connected to a gen nor an ext grid


def pf_preprocessing(net: pandapowerNet) -> pandapowerNet:
    """
    Sets variables to the results of OPF
    - sgen.p_mw, sgen.q_mvar: active and reactive power generation for static generators
    - gen.p_mw, gen.vm_pu: active power and voltage magnitude for generators
    - shunt.q_mvar, shunt.p_mw: active and reactive power for shunt elements
    - ext_grid.q_mvar, ext_grid.p_mw: active and reactive power for external grids = slack buses
    """
    net.sgen[["p_mw"]] = net.res_sgen[
        ["p_mw"]
    ]  # No need to set q_mvar as Anna did in her code.
    net.gen[["p_mw", "vm_pu"]] = net.res_gen[
        ["p_mw", "vm_pu"]
    ]  # PV node... so we set P and V ;)
    # net.ext_grid[["q_mvar", "p_mw"]] = net.res_ext_grid[["q_mvar", "p_mw"]]  # Slack
    return net


def pf_post_processing(net: pandapowerNet) -> np.ndarray:
    """
    Post-process the PF data to build the final data representation for the given scenario, shape (n_buses, 10)
    columns are (bus, Pd, Qd, Pg, Qg, Vm, Va, PQ, PV, REF)
    rows are buses
    """
    X = np.zeros((net.bus.shape[0], 10))
    all_loads = (
        pd.concat([net.res_load])[["p_mw", "q_mvar", "bus"]].groupby("bus").sum()
    )  # shunt + loads

    all_gens = (
        pd.concat([net.res_gen, net.res_sgen, net.res_ext_grid])[
            ["p_mw", "q_mvar", "bus"]
        ]
        .groupby("bus")
        .sum()
    )

    assert (net.bus.index.values == list(range(X.shape[0]))).all()

    X[:, 0] = net.bus.index.values

    # Active and reactive power demand
    X[all_loads.index, 1] = all_loads.p_mw  # Pd
    X[all_loads.index, 2] = all_loads.q_mvar  # Qd

    # Active and reactive power generated
    X[net.bus.type == PV, 3] = all_gens.p_mw[
        net.res_bus.type == PV
    ]  # active Power generated
    X[net.bus.type == PV, 4] = all_gens.q_mvar[
        net.res_bus.type == PV
    ]  # reactive Power generated
    X[net.bus.type == REF, 3] = all_gens.p_mw[
        net.res_bus.type == REF
    ]  # active Power generated
    X[net.bus.type == REF, 4] = all_gens.q_mvar[
        net.res_bus.type == REF
    ]  # reactive Power generated

    # Voltage
    X[:, 5] = net.res_bus.vm_pu  # voltage magnitude
    X[:, 6] = net.res_bus.va_degree  # voltage angle
    X[:, 7:10] = pd.get_dummies(net.bus["type"]).values

    return X


def get_adjacency_list(net: pandapowerNet) -> list:
    """'
    Get adjacency list for network
    """
    ppc = net._ppc
    Y_bus, Yf, Yt = makeYbus_pypower(ppc["baseMVA"], ppc["bus"], ppc["branch"])

    i, j = np.nonzero(
        Y_bus
    )  # This gives you the row and column indices of non-zero elements
    s = Y_bus[i, j]
    G = np.real(s)
    B = np.imag(s)

    edge_index = np.column_stack((i, j))
    edge_attr = np.stack((G, B)).T
    adjacency_lists = np.column_stack((edge_index, edge_attr))
    return adjacency_lists


def process_scenario(
    net,
    scenarios,
    scenario_index,
    generator,
    no_stats,
    local_csv_data,
    local_adjacency_lists,
    local_stats,
    error_log_file,
):
    """
    Process a load scenario
    """
    net.load.p_mw = scenarios[net.load.bus, scenario_index, 0]
    net.load.q_mvar = scenarios[net.load.bus, scenario_index, 1]
    # Generate perturbed topologies
    perturbed_topologies = generator.generate(net)

    for perturbed_topology in perturbed_topologies:

        try:
            run_opf(perturbed_topology)
        except Exception as e:
            with open(error_log_file, "a") as f:
                f.write(
                    f"Caught an exception at scenario {scenario_index} in run_opf function: {e}\n"
                )
            continue

        net_pf = copy.deepcopy(perturbed_topology)

        net_pf = pf_preprocessing(net_pf)

        try:
            run_pf(net_pf)
        except Exception as e:
            with open(error_log_file, "a") as f:
                f.write(
                    f"Caught an exception at scenario {scenario_index} in run_pf function: {e}\n"
                )
            continue

        # Append processed power flow data
        local_csv_data.extend(pf_post_processing(net_pf))
        local_adjacency_lists.append(get_adjacency_list(net_pf))
        if not no_stats:
            local_stats.update(net_pf)

    return local_csv_data, local_adjacency_lists, local_stats
