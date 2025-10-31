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


# Installation

1. ⭐ Star the [repository](https://github.com/gridfm/gridfm-datakit) on GitHub to support the project!

2. Install gridfm-datakit

    ```bash
    python -m pip install --upgrade pip  # Upgrade pip
    pip install gridfm-datakit
    ```

3. Install Julia with Powermodels and Ipopt

    ```bash
    gridfm_datakit setup_pm
    ```

### For Developers

To install the latest development version from GitHub, follow these steps instead of step 2.

```bash
git clone https://github.com/gridfm/gridfm-datakit.git
cd "gridfm-datakit"
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip  # Upgrade pip to ensure compatibility with pyproject.toml
pip3 install -e '.[test,dev]'
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
gridfm_datakit path/to/config.yaml
```


## Configuration Overview

Refer to the sections [Network](network.md), [Load Scenarios](load_scenarios.md), and [Topology perturbations](topology_perturbations.md) for a description of the configuration parameters.

Sample configuration files are provided in `scripts/config`, e.g. `case24_ieee_rts.yaml`:

```yaml
network:
  name: "case24_ieee_rts" # Name of the power grid network (without extension)
  source: "pglib" # Data source for the grid; options: pglib, file
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

topology_perturbation:
  type: "random" # Type of topology generator; options: n_minus_k, random, none
  # WARNING: the following parameters are only used if type is not "none"
  k: 1 # Maximum number of components to drop in each perturbation
  n_topology_variants: 5 # Number of unique perturbed topologies per scenario
  elements: [branch, gen] # elements to perturb. options: branch, gen

generation_perturbation:
  type: "cost_permutation" # Type of generation perturbation; options: cost_permutation, cost_perturbation, none
  # WARNING: the following parameters are onlyused if type is "cost_perturbation"
  sigma: 1.0 # Size of range use for sampling scaling factor

settings:
  num_processes: 10 # Number of parallel processes to use
  data_dir: "./data_out" # Directory to save generated data relative to the project root
  large_chunk_size: 50 # Number of load scenarios processed before saving
  overwrite: true # If true, overwrites existing files, if false, appends to files
  mode: "pf" # Mode of the script; options: pf, opf. pf: power flow data where one or more operating limits – the inequality constraints defined in OPF, e.g., voltage magnitude or branch limits – may be violated. opf:  datapoints for training OPF solvers, with cost-optimal dispatches that satisfy all operating limits (OPF-feasible)
```

<br>

## Output Files

The data generation process writes the following artifacts under:
`{settings.data_dir}/{network.name}/raw`

- **tqdm.log**: Progress bar log.
- **error.log**: Error messages captured during generation.
- **args.log**: YAML dump of the configuration used for this run.
- **scenarios_{generator}.parquet**: Load scenarios (per-element time series) produced by the selected load generator.
- **scenarios_{generator}.html**: Plot of the generated load scenarios.
- **scenarios_{generator}.log**: Generator-specific notes (e.g., bounds for the global scaling factor when using `agg_load_profile`).
- **bus_data.parquet**: Bus-level features for each processed scenario (columns `BUS_COLUMNS` and, if `settings.dcpf=True`, also `DC_BUS_COLUMNS`).
- **gen_data.parquet**: Generator features per scenario (columns `GEN_COLUMNS`).
- **branch_data.parquet**: Branch features per scenario (columns `BRANCH_COLUMNS`).
- **y_bus_data.parquet**: Nonzero Y-bus entries per scenario with columns `[scenario, index1, index2, G, B]`.
- **stats.csv**: (if `settings.no_stats=False`) Aggregated statistics collected during generation.
- **stats_plot.html**: (if `settings.no_stats=False`) HTML dashboard of the aggregated statistics.
- **feature_plots/**: Created if `bus_data.parquet` exists; contains violin plots per feature named `distribution_{feature_name}.png`.
