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


## Configuration Overview

Refer to the sections [Network](network.md), [Load Scenarios](load_scenarios.md), and [Topology perturbations](topology_perturbations.md) for a description of the configuration parameters.

To generate time-domain trajectories instead of power flow snapshots, add a
`dynamic:` block to the config; the same `generate` command then runs the
dynamic pipeline. See [Dynamic Simulation](dynamic_simulation.md).

To read the grid or solve the power flow with PowSyBl, see [PowSyBl](powsybl.md).

Sample configuration files are provided in `scripts/config`, e.g. `default.yaml`:

```yaml
network:
  name: "case24_ieee_rts" # Name of the power grid network (without extension)
  source: "pglib" # Data source for the grid; options: pglib, file
  reader: "native" # Network reader; options: native, powsybl
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
  outage_count_probabilities: [0.1, 0.8, 0.1] # Optional: P(N-0), P(N-1), P(N-2)

generation_perturbation:
  type: "cost_permutation" # Type of generation perturbation; options: cost_permutation, cost_perturbation, none
  # WARNING: sigma is required only if type is "cost_perturbation"
  # sigma: 1.0 # Size of range used for sampling scaling factor

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
  pf_solver: "powermodel" # Engine solving the power flow in pf mode; options: powermodel, powsybl, lightsim2grid (the OPF is always solved by PowerModels). powsybl requires network.reader: powsybl
  pf_fast: true # Whether to use fast PF solver by default (compute_ac_pf from powermodels.jl); if false, uses Ipopt-based PF. Some networks e.g. case10000_goc do not work with pf_fast: true
  dcpf_fast: true # Whether to use fast DC PF solver (compute_dc_pf from powermodels.jl); if false, uses optimizer-based DC PF
  enable_solver_logs: false # If true, write OPF/PF solver logs to {data_dir}/solver_log; PF fast and DCPF fast ignore logging
  max_iter: 200 # Maximum iterations for Ipopt-based solvers

```

## Parallel generation

The CLI and `generate_power_flow_data_distributed` support two process backends:

```yaml
settings:
  num_processes: 4
  is_multiprocessing: true
```

With `is_multiprocessing: true`, static generation uses a spawned
`multiprocessing.Pool` and a Manager queue to display progress after each load
scenario. The pool and its initialized solver instances are reused across large
chunks. Results are saved in scenario order, regardless of worker completion
order. Worker exceptions stop generation and clean up the pool and Manager.

The default, `false`, uses the existing `ProcessPoolExecutor` backend; with one
worker it processes scenarios in the current process. The Pool backend spawns a
worker even when `num_processes: 1`. Both backends preserve chunk boundaries,
seed derivation, and solver options, including CoinHSL and `opf_formulation`.
The setting does not change the sequential `generate_power_flow_data` API or
the dynamic simulation backend.

`overwrite: false` retains its existing append behavior when checkpointing is disabled.

### Scenario timeouts

To skip load scenarios that hang or exceed a wall-clock limit, use the Pool
backend and set a positive timeout in seconds:

```yaml
settings:
  is_multiprocessing: true
  scenario_timeout_sec: 600
  enable_checkpointing: true
```

The timer covers one complete load scenario: its dispatch OPF and all topology
variants. It begins after worker/solver initialization, immediately before
scenario processing. Initialization, queue wait, and result transfer are outside
this limit. A scenario that exceeds it is skipped in full, so none of its
partially accumulated variants are saved.

The parent kills the offending worker, terminates the remaining pool, and retries
the unsaved large chunk in a fresh pool without known offenders. Other scenarios
from that attempt are recomputed. Already checkpointed chunks remain untouched.
Retries finish even if every scenario in a chunk times out. Ordinary exceptions
and unexpected worker exits stop generation rather than being silently skipped.

Skipped global `load_scenario_idx` values and measured durations are written to
`timed_out_scenarios.jsonl` and reported in `error.log`. With checkpointing enabled,
the checkpoint also persists `skipped_scenarios` immediately, before the chunk is
saved, so resuming does not rerun known offenders. Without checkpointing, the skip
set applies to the current invocation. Logs retain the history of aborted attempts.

Timeout mode seeds each load scenario independently using its global index.
Retries and skipped scenarios therefore do not change the random perturbations of
surviving scenarios. This differs from the legacy chunk-stream seed sequence:
enabling timeouts can change seeded output even when no limit is exceeded. Keep
the timeout setting unchanged when resuming a checkpointed run.

The default `scenario_timeout_sec: null` disables timeouts and retains the
existing seed behavior. Timeouts apply only to distributed static generation with
`is_multiprocessing: true`; sequential and dynamic pipelines reject them.

### Checkpoint and resume

Enable checkpoints for distributed static generation (either process backend):

```yaml
settings:
  enable_checkpointing: true
  overwrite: true
  seed: null
```

The run writes `checkpoint.json` before generating scenarios. It saves the actual
random seed even when `seed` is null, then advances the checkpoint only after all
five output datasets for a large chunk are saved. To resume an interrupted run,
use the same configuration and output directory with `overwrite: false`.
The saved seed is reused; completed large chunks are skipped. An already completed
run returns its artifact paths without regenerating scenarios or starting workers.

The checkpoint records the committed parquet files and output sample count. On
resume, parquet files written after the last committed checkpoint are removed and
that count is restored, allowing an interrupted chunk save to be retried without
duplicate samples. Missing or truncated committed files cause an error.

Keep the same scenario count, worker count, chunk size, network, perturbations,
and solver settings. An explicit seed must match the saved seed; `seed: null`
uses the saved one. You may change solver logging or move the complete output
directory. You may switch process backends when scenario timeouts are disabled.
Keep input-file contents and the software
environment unchanged for reproducible results, and use one generation process
per output directory.

Existing data without a checkpoint cannot be adopted into a resumable run.
Disabling checkpointing on existing checkpointed data is rejected to prevent an
ordinary append from invalidating its state. Use `overwrite: true` for a fresh
run. Checkpointing is disabled by default and applies to the CLI's static pipeline
and `generate_power_flow_data_distributed`, not the sequential or dynamic APIs.

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
- **PF Solver Choice**: `settings.pf_solver` selects the engine: `powermodel` (default), `powsybl`, or `lightsim2grid`. The OPF is always solved by PowerModels. The `powsybl` backend requires `network.reader: "powsybl"`. See [Power flow solver](power_flow_solver.md).
- **PF Fast Path**: With `powermodel`, `settings.pf_fast` selects the fast `compute_ac_pf` path when `true` and the Ipopt-based AC PF when `false`.

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

Running `gridfm-datakit stats` additionally writes `stats.parquet` and `stats_plot.png` to the data directory.
