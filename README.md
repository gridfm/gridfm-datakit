[![Docs](https://img.shields.io/badge/docs-available-brightgreen)](https://gridfm.github.io/gridfm-datakit/)
![Coverage](https://img.shields.io/badge/coverage-76%25-yellow)


# Overview


This library is brought to you by the GridFM team to generate power flow data to train machine learning and foundation models.



<p align="center">
  <img src="https://raw.githubusercontent.com/gridfm/gridfm-datakit/refs/heads/main/docs/figs/pipeline_readme.png" alt=""/>
  <br/>
</p>


---



## Comparison with other PF datasets/ libraries

| Feature                                                    | GraphNeuralSolver [\[1\]](https://doi.org/10.1016/j.epsr.2020.106547) | OPFData [\[2\]](https://arxiv.org/abs/2406.07234) | OPFLearn [\[3\]](https://arxiv.org/abs/2111.01228) | PowerFlowNet [\[4\]](https://arxiv.org/abs/2311.03415) | TypedGNN [\[5\]](https://doi.org/10.1016/j.engappai.2022.105567) | PF△ [\[6\]](https://www.climatechange.ai/papers/iclr2025/67) | **PGLearn** [\[7\]](https://openreview.net/pdf?id=cecIf0CKnH) | **gridfm-datakit** [\[8\]](https://www.cell.com/joule/fulltext/S2542-4351(24)00470-7) |
| ---------------------------------------------------------- | ----------------- | ------- | -------- | ------------- | -------- | --- | ----------------------------- | ---------- |
| Generator Profile                                          | ✅                | ❌      | ❌       | ✅            | ✅       | ✅  | ❌                            | ❌         |
| N-1                                                        | ❌                | ✅      | ❌       | ❌            | ✅       | ✅  | ✅                            | ✅         |
| > 1000 Buses                                               | ❌                | ✅      | ✅       | ❌            | ❌       | ✅  | ✅                            | ✅         |
| N-k, k > 1                                                 | ❌                | ❌      | ❌       | ❌            | ❌       | ❌  | ❌                            | ✅         |
| Load Scenarios from Real World Data                        | ❌                | ❌      | ❌       | ❌            | ❌       | ❌  | ❌                            | ✅         |
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
from gridfm_datakit.interactive_utils import interactive_interface
interactive_interface()
```


## Option 2: Using the command line interface

Run the data generation routine from the command line:

```bash
gridfm_datakit path/to/config.yaml
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
  sigma: 0.05 # max local noise
  change_reactive_power: true # If true, changes reactive power of loads. If False, keeps the ones from the case file
  global_range: 0.4 # Range of the global scaling factor. used to set the lower bound of the scaling factor
  max_scaling_factor: 4.0 # Max upper bound of the global scaling factor
  step_size: 0.025 # Step size when finding the upper bound of the global scaling factor
  start_scaling_factor: 0.8 # Initial value of the global scaling factor

topology_perturbation:
  type: "random" # Type of topology generator; options: n_minus_k, random, none
  # WARNING: the following parameters are only used if type is not "none"
  k: 1 # Maximum number of components to drop in each perturbation
  n_topology_variants: 5 # Number of unique perturbed topologies per scenario
  elements: ["line", "trafo", "gen", "sgen"] # elements to perturb options: line, trafo, gen, sgen

settings:
  num_processes: 10 # Number of parallel processes to use
  data_dir: "./data_out" # Directory to save generated data relative to the project root
  large_chunk_size: 50 # Number of load scenarios processed before saving
  no_stats: false # If true, disables statistical calculations
  overwrite: true # If true, overwrites existing files, if false, appends to files (note that bus_params.csv, edge_params.csv, scenarios_{load.generator}.csv and scenarios_{load.generator}.html will still be overwritten)
  mode: "pf" # Mode of the script; options: contingency, pf
```

<br>

## Output Files

The data generation process produces several output files in the specified data directory:

- **tqdm.log**: Progress bar log.
- **error.log**: Log of the errors raised during data generation.
- **args.log**: Copy of the config file used.
- **pf_node.csv**: Data related to the nodes (buses) in the network, such as voltage levels and power injections.
- **pf_edge.csv**: Branch admittance matrix for each pf case.
- **branch_idx_removed.csv**: List of the indices of the branches (lines and transformers) that got removed when perturbing the topologies.
- **edge_params.csv**: Branch admittance matrix and branch rate limits for the unperturbed topology.
- **bus_params.csv**: Parameters for the buses (voltage limits and the base voltage).
- **scenario_{args.load.generator}.csv**: Load element-level load profile obtained after using the load scenario generator.
- **scenario_{args.load.generator}.html**: Plots of the element-level load profile.
- **scenario_{args.load.generator}.log**: If generator is "agg_load_profile", stores the upper and lower bounds for the global scaling factor.
- **stats.csv**: Stats about the generated data.
- **stats_plot.html**: Plots of the stats about the generated data.
