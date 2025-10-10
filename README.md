<p align="center">
  <img src="https://raw.githubusercontent.com/gridfm/gridfm-datakit/refs/heads/main/docs/figs/KIT_logo.png" alt="GridFM logo" style="width: 40%; height: auto;"/>
  <br/>
</p>

<p align="center" style="font-size: 25px;">
    <b>gridfm-datakit</b>
</p>

[![Docs](https://img.shields.io/badge/docs-available-brightgreen)](https://gridfm.github.io/gridfm-datakit/)
![Coverage](https://img.shields.io/badge/coverage-76%25-yellow)
![Python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.12-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)


This library is brought to you by the GridFM team to generate power flow data to train machine learning and foundation models.

---



## Comparison with other PF datasets/ libraries

| Feature                                                    | GraphNeuralSolver [\[1\]](https://doi.org/10.1016/j.epsr.2020.106547) | OPFData [\[2\]](https://arxiv.org/abs/2406.07234) | OPFLearn [\[3\]](https://arxiv.org/abs/2111.01228) | PowerFlowNet [\[4\]](https://arxiv.org/abs/2311.03415) | TypedGNN [\[5\]](https://doi.org/10.1016/j.engappai.2022.105567) | PF△ [\[6\]](https://www.climatechange.ai/papers/iclr2025/67) | **PGLearn** [\[7\]](https://openreview.net/pdf?id=cecIf0CKnH) | **gridfm-datakit** [\[8\]](https://www.cell.com/joule/fulltext/S2542-4351(24)00470-7) |
| ---------------------------------------------------------- | ----------------- | ------- | -------- | ------------- | -------- | --- | ----------------------------- | ---------- |
| Generator Profile                                          | ✅                | ❌      | ❌       | ✅            | ✅       | ✅  | ❌                            | ✅         |
| N-1                                                        | ❌                | ✅      | ❌       | ❌            | ✅       | ✅  | ✅                            | ✅         |
| > 1000 Buses                                               | ❌                | ✅      | ✅       | ❌            | ❌       | ✅  | ✅                            | ✅         |
| N-k, k > 1                                                 | ❌                | ❌      | ❌       | ❌            | ❌       | ❌  | ❌                            | ✅         |
| Load Scenarios from Real World Data                        | ❌                | ❌      | ❌       | ❌            | ❌       | ❌  | ❌                            | ✅         |
| Net Param Perturbation                                     | ✅                | ❌      | ❌       | ✅            | ✅       | ❌  | ❌                            | ✅         |
| Multi-processing and scalable to very large (1M+) datasets | ❌                | ❌      | ❌       | ❌            | ❌       | ❌  | ✅                            | ✅         |


# Installation

1. ⭐ Star the [repository](https://github.com/gridfm/gridfm-datakit) on GitHub to support the project!

2. Run:

    ```bash
    python -m pip install --upgrade pip  # Upgrade pip
    pip install gridfm-datakit
    ```

# Getting Started

## Option 1: Run data gen using interactive interface

To use the interactive interface, either open `scripts/interactive_interface.ipynb` or copy the following into a Jupyter notebook and follow the instructions:

```python
from gridfm_datakit.interactive import interactive_interface
interactive_interface()
```


## Option 2: Using the command line interface

Run the data generation routine from the command line:

```bash
gridfm_datakit generate path/to/config.yaml
```


## Configuration Overview

Refer to the sections [Network](network.md), [Load Scenarios](load_scenarios.md), and [Topology perturbations](topology_perturbations.md) for a description of the configuration parameters.

Sample configuration files are provided in `scripts/config`, e.g. `default.yaml`:

```yaml
network:
  name: "case24_ieee_rts" # Name of the power grid network (without extension)
  source: "pglib" # Data source for the grid; options: pglib, pandapower, file
  network_dir: "scripts/grids" # if using source "file", this is the directory containing the network file (relative to the project root)


load:
  generator: "agg_load_profile" # Name of the load generator; options: agg_load_profile, powergraph
  agg_profile: "default" # Name of the aggregated load profile
  scenarios: 200 # Number of different load scenarios to generate
  # WARNING: the following parameters are only used if generator is "agg_load_profile"
  # if using generator "powergraph", these parameters are ignored
  sigma: 0.2 # max local noise
  change_reactive_power: true # If true, changes reactive power of loads. If False, keeps the ones from the case file
  global_range: 0.4 # Range of the global scaling factor. used to set the lower bound of the scaling factor
  max_scaling_factor: 4.0 # Max upper bound of the global scaling factor
  step_size: 0.025 # Step size when finding the upper bound of the global scaling factor
  start_scaling_factor: 0.8 # Initial value of the global scaling factor
  find_limit: false # If set to true, will run opf with pandapower run OPF to find upper limit. Set to false if using Julia to run OPF
  upper_limit: 1.2 # Upper limit for load profile - when using Julia

topology_perturbation:
  type: "random" # Type of topology generator; options: n_minus_k, random, none
  # WARNING: the following parameters are only used if type is not "none"
  k: 1 # Maximum number of components to drop in each perturbation
  n_topology_variants: 5 # Number of unique perturbed topologies per scenario
  elements: ["line", "trafo", "gen", "sgen"] # elements to perturb options: line, trafo, gen, sgen

generation_perturbation:
  type: "cost_permutation" # Type of generation perturbation; options: cost_permutation, cost_perturbation, none
  # WARNING: the following parameters are onlyused if type is "cost_perturbation"
  sigma: 1.0 # Size of range use for sampling scaling factor

settings:
  num_processes: 10 # Number of parallel processes to use
  data_dir: "./data_out" # Directory to save generated data relative to the project root
  large_chunk_size: 50 # Number of load scenarios processed before saving
  no_stats: false # If true, disables statistical calculations
  overwrite: true # If true, overwrites existing files, if false, appends to files
  mode: "unsecure" # Mode of the script; options: secure, unsecure. Unsecure mode generates unsecure scenarios, i.e. scenarios where the the generator setpoints are obtained for the base topology, before the topology is perturbed.
  dcpf: false # If true, also stores the results of dc power flow (in addition to the results AC power flow)
  julia: true # If true, uses Julia PowerModels to run OPF. If false, uses PandaPower OPF function.
  pm_solver: "ipopt" # Ipopt solver as default; update for ma27, ma57. Will be updated to pp.runpm_ac_opf(net, pm_log_level=5, pm_solver=pm_solver) if julia: true in solvers.py
```

<br>

## Output Files

The data generation process writes the following artifacts under:
`{settings.data_dir}/{network.name}/raw`

- **tqdm.log**: Progress bar log.
- **error.log**: Error messages captured during generation.
- **args.log**: YAML dump of the configuration used for this run.
- **scenarios_{generator}.csv**: Load scenarios (per-element time series) produced by the selected load generator.
- **scenarios_{generator}.html**: Plot of the generated load scenarios.
- **scenarios_{generator}.log**: Generator-specific notes (e.g., bounds for the global scaling factor when using `agg_load_profile`).
- **bus_data.csv**: Bus-level features for each processed scenario (columns `BUS_COLUMNS` and, if `settings.dcpf=True`, also `DC_BUS_COLUMNS`).
- **gen_data.csv**: Generator/ext_grid/sgen features per scenario (columns `GEN_COLUMNS`).
- **branch_data.csv**: Branch features per scenario (columns `BRANCH_COLUMNS`).
- **y_bus_data.csv**: Nonzero Y-bus entries per scenario with columns `[scenario, index1, index2, G, B]`.
- **stats.csv**: (if `settings.no_stats=False`) Aggregated statistics collected during generation.
- **stats_plot.html**: (if `settings.no_stats=False`) HTML dashboard of the aggregated statistics.
- **feature_plots/**: Created if `bus_data.csv` exists; contains violin plots per feature named `distribution_{feature_name}_all_buses.png`.
