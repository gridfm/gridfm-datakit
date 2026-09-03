# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gridfm_datakit.network import load_net_from_pglib
from gridfm_datakit.utils.idx_bus import PD
from gridfm_datakit.utils.idx_gen import GEN_BUS
import os


# %%
# Input dataset for Pd, Qd: (N, L)

# Source directory of the raw opflearn download.
# Override with the OPFLEARN_DIR environment variable.
dir_path = os.environ.get("OPFLEARN_DIR", "data/opflearn/")
if not dir_path.endswith("/"):
    dir_path += "/"
df = pd.read_csv(dir_path + "pglib_opf_case118_ieee.csv")
n_scenarios = df.shape[0]

# %%
net = load_net_from_pglib("case118_ieee")
n_buses = net.buses.shape[0]
n_gens = net.gens.shape[0]
buses_with_load = np.where(net.buses[:, PD] >0)[0]
buses_with_gen = net.gens[:,GEN_BUS].astype(int)

# %%


# %%
pl_columns = [col for col in df.columns if "pl" in col]
ql_columns = [col for col in df.columns if "ql" in col]
pg_columns = [col for col in df.columns if col.endswith("pg")]
qg_columns = [col for col in df.columns if col.endswith("qg")]
v_bus_columns = [col for col in df.columns if col.endswith("v_bus")]

# %%
Pd = np.array(df[pl_columns])
Qd = np.array(df[ql_columns])
Pg = np.array(df[pg_columns])
Qg = np.array(df[qg_columns])
Vm = np.array(df[v_bus_columns].apply(lambda x: x.str.replace(" ", "").astype(complex)))


assert Pd.shape[1] == len(buses_with_load), "Number of buses with load and pd do not match"
assert Qd.shape[1] == len(buses_with_load), "Number of buses with load and qd do not match"
assert Pg.shape[1] == len(buses_with_gen), "Number of buses with gen and pg do not match"
assert Qg.shape[1] == len(buses_with_gen), "Number of buses with gen and qg do not match"
assert Vm.shape[1] == n_buses, "Number of buses and vm do not match"

# %%
pd_bus = np.zeros((n_scenarios, net.buses.shape[0]))
qd_bus = np.zeros((n_scenarios, net.buses.shape[0]))
pg_bus = np.zeros((n_scenarios, net.buses.shape[0]))
qg_bus = np.zeros((n_scenarios, net.buses.shape[0]))

pd_bus[:, buses_with_load] = Pd
qd_bus[:, buses_with_load] = Qd

# this works only if there is only one generator per bus
# if there are multiple generators per bus, we need to sum the pg and qg values
n_gens_per_bus = np.bincount(buses_with_gen.astype(int))
assert (n_gens_per_bus <= 1).all(), "There should be only maximum one generator per bus"
pg_bus[:, buses_with_gen] = Pg
qg_bus[:, buses_with_gen] = Qg


vm_bus = np.abs(Vm)
va_bus = np.rad2deg(np.angle(Vm) * -180/np.pi) # NOTE: CURRENTLY, THESE OPFLEARN DATASETS HAVE INCORRECT VALUES FOR THE COMPLEX BUS VOLTAGES. THE ANGLE OF THE COMPLEX BUS VOLTAGES ('v_bus') MUST BE SCALED BY -180/π TO GET THE CORRECT VALUES. 


# %%

df_bus = pd.DataFrame(
    {
        "scenario": np.repeat(np.arange(n_scenarios), n_buses),
        "bus": np.tile(np.arange(n_buses), n_scenarios),
        "Pd": pd_bus.flatten()* 100,
        "Qd": qd_bus.flatten()* 100,
        "Pg": pg_bus.flatten()* 100,
        "Qg": qg_bus.flatten()* 100,
        "Vm": vm_bus.flatten(),
        "Va": va_bus.flatten(),
    })

df_gen = pd.DataFrame(
    {
        "scenario": np.repeat(np.arange(n_scenarios), n_gens),
        "idx": np.tile(np.arange(n_gens), n_scenarios),
        "bus": np.tile(buses_with_gen, n_scenarios),
        "p_mw": Pg.flatten(order="C")* 100,
        "q_mvar": Qg.flatten(order="C")* 100,
    })


# # %%
# plt.figure(figsize=(22, 8))
# df_gen.boxplot(column="p_mw", by="gen")
# plt.title("PG per Gen")
# plt.suptitle("")
# plt.xticks(rotation=45, ha="right")
# plt.subplots_adjust(bottom=0.3)
# plt.show()


# # %%

# plt.figure(figsize=(22, 8))
# df_bus.boxplot(column="Pd", by="bus")
# plt.title("Pd per Bus")
# plt.suptitle("")
# plt.xticks(rotation=45, ha="right")
# plt.subplots_adjust(bottom=0.3)
# plt.show()



# %%
# save to bus_data.parquet and gen_data.parquet
os.makedirs(dir_path + "converted", exist_ok=True)
df_bus.to_parquet(dir_path + "converted/bus_data.parquet", index=False)
df_gen.to_parquet(dir_path + "converted/gen_data.parquet", index=False)
