# %%
import h5py
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Input dataset for Pd, Qd: (N, L)

# Source directory of the raw pglearn download.
# Override with the PGLEARN_DIR environment variable.
dir_path = os.environ.get("PGLEARN_DIR", "data/pglearn/PGLearn-Small-118_ieee-nminus1/")
if not dir_path.endswith("/"):
    dir_path += "/"
input_data = dir_path + "input.h5"
input_dataset = h5py.File(input_data, 'r')

# Primal dataset for Pg, Qg: (N, G), Va, Vm: (N, N)
primal_data = dir_path + "primal.h5"  
primal_dataset = h5py.File(primal_data, 'r')


# Metadata
meta_data = dir_path + "meta.h5"  
meta_dataset = h5py.File(meta_data, 'r')

# Case file
with open(dir_path+'case.json', 'r') as f:
        case_data = json.load(f)

# %%
pg = np.array(primal_dataset["pg"])
qg = np.array(primal_dataset["qg"])
vm = np.array(primal_dataset["vm"])
va = np.array(primal_dataset["va"])
Pd = np.array(input_dataset["pd"])
qd = np.array(input_dataset["qd"])

n_scenarios = va.shape[0]
n_buses = va.shape[1]

n_gens = pg.shape[1]
n_loads = qd.shape[1]

assert max([len(i) for i in case_data['data']['bus_gens']]) == 1, "There should be only one generator per bus"
buses_with_gens = [i for i in range(n_buses) if case_data['data']['bus_gens'][i]] # this works under the assumption that there is only one generator per bus
buses_with_loads = [i for i in range(n_buses) if case_data['data']['bus_loads'][i]]

assert len(buses_with_gens) == n_gens, "Number of generators is not correct"
assert len(buses_with_loads) == n_loads, "Number of loads is not correct"

assert max([len(i) for i in case_data['data']['bus_gens']]) == 1, "There should be only one generator per bus"
assert max([len(i) for i in case_data['data']['bus_loads']]) == 1, "There should be only one load per bus"

# assert gen_idx are consecutive
assert (np.concatenate(case_data['data']['bus_gens']).astype(int)-1 == list(range(n_gens))).all(), "gen_idx are not consecutive"
assert (np.concatenate(case_data['data']['bus_loads']).astype(int)-1 == list(range(n_loads))).all(), "load_idx are not consecutive"

pg_all = np.zeros((n_scenarios, n_buses))
qg_all = np.zeros((n_scenarios, n_buses))
pd_all = np.zeros((n_scenarios, n_buses))
qd_all = np.zeros((n_scenarios, n_buses))

# %%
dup_gens = np.where(np.bincount(buses_with_gens) > 1)[0]
pd_all[:, buses_with_loads] = Pd
qd_all[:, buses_with_loads] = qd

pg_all[:, buses_with_gens] = pg
qg_all[:, buses_with_gens] = qg


df_bus = pd.DataFrame(
    {
        "scenario": np.repeat(np.arange(n_scenarios), n_buses),
        "bus": np.tile(np.arange(n_buses), n_scenarios),
        "Pd": pd_all.flatten() * 100,
        "Qd": qd_all.flatten()* 100,
        "Pg": pg_all.flatten()* 100,
        "Qg": qg_all.flatten()* 100,
        "Vm": vm.flatten(),
        "Va": np.rad2deg(va.flatten()),
    })



# %%
df_gen = pd.DataFrame(
    {
        "scenario": np.repeat(np.arange(n_scenarios), n_gens),
        "idx": np.tile(np.arange(n_gens), n_scenarios),
        "bus": np.tile(buses_with_gens, n_scenarios),
        "p_mw": pg.flatten(order="C")* 100,
        "q_mvar": qg.flatten(order="C")* 100,
        "gen_status": np.array(input_dataset["gen_status"]).flatten()
    })

# %%
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
df_bus.to_parquet(dir_path + "converted/bus_data.parquet", index=False)
df_gen.to_parquet(dir_path + "converted/gen_data.parquet", index=False)



