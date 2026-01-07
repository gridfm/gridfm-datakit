## Option 1: Run data gen using interactive interface

To use the interactive interface, either open `scripts/interactive_interface.ipynb` or copy the following into a Jupyter notebook and follow the instructions:

```python
from gridfm_datakit.interactive import interactive_interface
interactive_interface()
```


## Option 2: Using the command line interface

### Generate Data

Run the data generation routine from the command line:

```bash
gridfm-datakit generate path/to/config.yaml
```
```


## Configuration Overview

Refer to the sections [Network](network.md), [Load Scenarios](load_scenarios.md), and [Topology perturbations](topology_perturbations.md) for a description of the configuration parameters.

Sample configuration files are provided in `scripts/config`, e.g. `default.yaml`:

```yaml
network:
  name: "case24_ieee_rts" # Name of the power grid network (without extension)
  source: "pglib" # Data source for the grid; options: pglib, file
  # WARNING: the following parameter is only used if source is "file"
  network_dir: "scripts/grids" # if using source "file", this is the directory containing the network file (relative to the project root)

load:
  generator: "agg_load_profile" # Name of the load generator; options: agg_load_profile, powergraph
  agg_profile: "default" # Name of the aggregated load profile
  scenarios: 10000 # Number of different load scenarios to generate
  # WARNING: the following parameters are only used if generator is "agg_load_profile"
  # if using generator "powergraph", these parameters are ignored
  sigma: 0.2 # max local noise
  change_reactive_power: true # If true, changes reactive power of loads. If False, keeps the ones from the case file
  global_range: 0.4 # Range of the global scaling factor. used to set the lower bound of the scaling factor
  max_scaling_factor: 4.0 # Max upper bound of the global scaling factor
  step_size: 0.05 # Step size when finding the upper bound of the global scaling factor
  start_scaling_factor: 0.8 # Initial value of the global scaling factor

topology_perturbation:
  type: "random" # Type of topology generator; options: n_minus_k, random, none
  # WARNING: the following parameters are only used if type is not "none"
  k: 2 # Maximum number of components to drop in each perturbation
  n_topology_variants: 10 # Number of unique perturbed topologies per scenario
  elements: [branch, gen] # elements to perturb. options: branch, gen

generation_perturbation:
  type: "cost_permutation" # Type of generation perturbation; options: cost_permutation, cost_perturbation, none
  # WARNING: the following parameter is only used if type is "cost_permutation"
  sigma: 1.0 # Size of range used for sampling scaling factor

admittance_perturbation:
  type: "random_perturbation" # Type of admittance perturbation; options: random_perturbation, none
  # WARNING: the following parameter is only used if type is "random_perturbation"
  sigma: 0.2 # Size of range used for sampling scaling factor

settings:
  num_processes: 10 # Number of parallel processes to use
  data_dir: "./data_out" # Directory to save generated data relative to the project root
  large_chunk_size: 1000 # Number of load scenarios processed before saving
  overwrite: true # If true, overwrites existing files, if false, appends to files
  mode: "pf" # Mode of the script; options: pf, opf. pf: power flow data where one or more operating limits – the inequality constraints defined in OPF, e.g., voltage magnitude or branch limits – may be violated. opf: generates datapoints for training OPF solvers, with cost-optimal dispatches that satisfy all operating limits (OPF-feasible)
  include_dc_res: true # If true, also stores the results of dc power flow and dc optimal power flow
  pf_fast: true # Whether to use fast PF solver by default (compute_ac_pf from powermodels.jl); if false, uses Ipopt-based PF. Some networks e.g. case10000_goc do not work with pf_fast: true
  dcpf_fast: true # Whether to use fast DC PF solver (compute_dc_pf from powermodels.jl); if false, uses optimizer-based DC PF
  enable_solver_logs: false # If true, write OPF/PF solver logs to {data_dir}/solver_log; PF fast ignores logging

```

## Choosing the right mode

The `mode` parameter controls how the power flow scenarios are generated and validated:

### Optimal Power Flow Mode (`mode: "opf"`)
- **Generation**: Solves Optimal Power Flow (OPF) for each perturbed topology
- **Constraints**: Enforces all operational constraints (voltage limits, branch loading, generator limits)
- **Example Use Case**: Generating data to train a neural optimal power flow solver, or a power flow solver for cases within operating limits.
- **Performance**: Slower due to OPF solving for each scenario

### Power Flow Mode (`mode: "pf"`)
- **Generation**: Solves OPF for base topology, then applies topology perturbations and solves Power Flow (without changing the generator setpoints)
- **Constraints**: Since the topology perturbations are performed after solving OPF, the inequality constraints of OPF (e.g. branch loading, voltage magnitude at PQ buses, generator bounds on reactive power, etc) might be violated.
- **Use Case**: Training data for power flow, contingency analysis, etc
- **Performance**: Faster as it avoids re-solving OPF for each perturbed scenario
- **PF Solver Choice**: Controlled by `settings.pf_fast`. If `true`, uses the fast `compute_ac_pf` path. If `false`, uses the Ipopt-based AC PF which is slower for smaller grids but has better convergence properties for large grids.

## Data Validation

The generated data can be validated using the CLI validation command:

```bash
# Validate with default sampling (100 partitions of 200 scenarios)
gridfm-datakit validate ./data_out/case24_ieee_rts/raw

# Validate with custom number of partitions
gridfm-datakit validate ./data_out/case24_ieee_rts/raw --n-partitions 50

# Validate all partitions (slower but complete)
gridfm-datakit validate ./data_out/case24_ieee_rts/raw --n-partitions 0
```

This automatically detects the generation mode and runs appropriate validation checks:

- **Physical Consistency**: Power balance equations, Y-bus matrix consistency
- **Data Integrity**: File completeness, scenario indexing, bus indexing
- **Constraint Validation**: Generator limits, branch loading (OPF mode only)
- **Power Flow Accuracy**: Computed vs stored power flows

## Statistics

After generating data, you can compute and visualize statistics:

```bash
# Generate statistics plots
gridfm-datakit stats ./data_out/case24_ieee_rts/raw
```

This generates `stats_plot.png` showing distributions of:
- Network topology metrics (generator/branch counts)
- Branch loading and overload statistics
- Power balance errors

The plots help assess dataset quality and identify scenarios with constraint violations or balance errors.

To visualize individual feature distributions across buses, run:

```bash
gridfm-datakit plots ./data_out/case24_ieee_rts/raw --sn-mva 100
```

This command saves violin plots for each feature to `feature_plots/` (or a custom directory specified with `--output-dir`).

<br>

# Output Files

The data generation process writes the following artifacts under:
`{settings.data_dir}/{network.name}/raw`

- **tqdm.log**: Progress bar log.
- **error.log**: Error messages captured during generation.
- **args.log**: YAML dump of the configuration used for this run.
- **scenarios_{generator}.parquet**: Load scenarios (per-element time series) produced by the selected load generator.
- **scenarios_{generator}.html**: Plot of the generated load scenarios.
- **scenarios_{generator}.log**: Generator-specific notes (e.g., bounds for the global scaling factor when using `agg_load_profile`).
- **bus_data.parquet**: Bus-level features for each processed scenario (columns `BUS_COLUMNS` and, if `settings.include_dc_res=True`, also `DC_BUS_COLUMNS`).
- **gen_data.parquet**: Generator features per scenario (columns `GEN_COLUMNS`).
- **branch_data.parquet**: Branch features per scenario (columns `BRANCH_COLUMNS`).
- **y_bus_data.parquet**: Nonzero Y-bus entries per scenario with columns `[scenario, index1, index2, G, B]`.
- **stats.parquet**: (if `settings.no_stats=False`) Aggregated statistics collected during generation.
- **stats_plot.html**: (if `settings.no_stats=False`) HTML dashboard of the aggregated statistics.
