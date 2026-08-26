# Dynamic simulation example (IEEE14 + Dynawo)

A self-contained, end-to-end run of the dynamic-simulation pipeline: load
scenarios → balanced initial state (OPF → update_powsybl → AC power flow) →
Dynawo time-domain simulation → Parquet + Zarr + metadata + reports.

The scenario is an IEEE14 network where generator `_GEN____2_SM` is disconnected
at `t = 50 s`; the resulting voltage/field transients are recorded per scenario.

## Prerequisites

- `pip install 'gridfm-datakit[dynamic]'` (pulls in `pypowsybl` and `zarr`).
- **Dynawo** installed and referenced from `~/.itools/config.yml`
  (`dynawo: { homeDir: /path/to/dynawo }`). The pipeline checks for it before
  spawning any worker and fails with `Dynawo backend unavailable: …` explaining
  what is missing.
- Julia is bootstrapped automatically on first run (for the OPF step).

## Run

```bash
python scripts/dynamic_example/run.py     # from the project root
# or, from this folder:
python run.py
```

The CLI runs the same pipeline. A config carrying a `dynamic:` block selects it,
so there is no separate subcommand. Note that the CLI does not resolve this
example's folder-relative paths, which is what `run.py` is for:

```bash
gridfm_datakit generate config.yaml       # paths must be absolute
```

Progress logging streams advancement to the console as it runs:

```
HH:MM:SS INFO    gridfm_datakit.dynamic | Dynamic generation: 6 scenarios in 2 chunk(s), 2 worker(s).
HH:MM:SS INFO    gridfm_datakit.dynamic | Chunk 1/2 done (3 scenarios), 3 samples so far.
HH:MM:SS INFO    gridfm_datakit.dynamic | Chunk 2/2 done (3 scenarios), 6 samples so far.
HH:MM:SS INFO    gridfm_datakit.dynamic | Saved 6 samples to .../out/IEEE14/raw/dynamic (6 with dynamic results, 6 reports).
```

## Files

```
dynamic_example/
├── config.yaml     # the run configuration (paths are folder-relative)
├── run.py          # resolves paths, enables logging, calls generate_dynamic_data
├── grids/          # IEEE14.iidm (network) + IEEE14.par (Dynawo parameters)
└── inputs/         # the four dynamic input tables:
    ├── static_element_dynamic_models.csv   # which elements get which dynamic model
    ├── automation_systems.csv              # automation systems (e.g. under-voltage)
    ├── events.csv                          # disturbances (here: a generator disconnect)
    └── variables.csv                       # curves to monitor
```

## Output

Everything the run produces lives under **one root**: `settings.data_dir` (here
`out/`). There is no separate dynamic output directory; the layout reuses the
static pipeline's `{data_dir}/{network.name}/raw/`, with the dynamic artifacts in
a `dynamic/` subfolder of it.

```
out/                                    <- settings.data_dir
└── IEEE14/raw/
    ├── args.log, error.log
    ├── scenarios_agg_load_profile.{parquet,html,log}
    ├── solver_log/                     only when settings.enable_solver_logs is on
    └── dynamic/
        ├── bus_data.parquet / gen_data.parquet / branch_data.parquet
        │   y_bus_data.parquet / runtime_data.parquet
        │       static PF snapshot (features), tagged with
        │       (scenario_index, perturbation_index). This pair replaces the
        │       static pipeline's load_scenario_idx, which cannot tell two
        │       perturbations of one load scenario apart
        ├── dynamic_results.zarr/
        │       curves (n_samples, n_variables, n_timesteps) + scenario_index /
        │       perturbation_index coordinate arrays; join the two modalities on
        │       this key pair
        ├── reports/
        │       one Dynawo report (JSON) per sample: model build-up + convergence
        └── metadata.json
                variable names, dimensions, join-key index, config hash
```

A run whose `variables.csv` declares `FinalStateValue` rows also writes
`final_state_values.parquet` next to those files. This example monitors curves
only, so it produces none.

The dynamic artifacts sit in their own `dynamic/` subfolder rather than directly
in `raw/` because the static pipeline writes `bus_data.parquet` as a *partitioned
directory* while the dynamic pipeline writes it as a flat file: same name,
different kind, so they must not share a directory.

`raw/dynamic/` is owned by the pipeline and recreated on every run, so it never
mixes fresh artifacts with a previous run's leftovers.

### A note on `enable_solver_logs`

It is **off by default** here. Turning it on raises the Julia solver verbosity
to DEBUG, which un-silences PowerModels. PowerModels logs through its own
Julia-level logger, which the file-based log router does **not** reliably
capture, so `[ PowerModels | Info/Warn ]` lines can spill onto the console.
With it off, `init_julia` calls `PowerModels.silence()` at the source, keeping
the console limited to pipeline progress. The Dynawo per-simulation reports
(under `reports/`) are saved regardless of this setting.

## Variations

- **Faster/larger:** change `load.scenarios`, `settings.num_processes`,
  `settings.large_chunk_size`.
- **Topology perturbations:** set `topology_perturbation.type: random` (with
  `k` / `n_topology_variants` / `elements`) to expand each scenario into several
  samples, one Dynawo run per perturbed topology, each labelled by
  `perturbation_index`. This is the only perturbation that adds *dynamic*
  diversity.
- **`admittance_perturbation`:** accepted for parity with the static pipeline.
  Its perturbed `r`/`x` are written to the pypowsybl network with the OPF
  set-points, so they do change the simulated network. But one scenario still
  yields one sample, and the effect on the dynamic outputs is unvalidated.
- **`generation_perturbation` does not work here.** It randomises generator
  *cost*, and `reader: powsybl`, which dynamic runs require, carries no cost
  data, so every generator gets the same placeholder `(c2=0, c1=1, c0=0)`.
  `cost_permutation` is then a strict no-op; `cost_perturbation` only spreads
  synthetic values around that placeholder.
- **Quieter/louder:** `dynamic.logging.verbosity` (`silent|error|warning|info|debug`)
  sets the progress logger's threshold; `settings.enable_solver_logs: true`
  captures solver output under `raw/solver_log/` (see the note above);
  `dynamic.logging.save_reports: false` skips the per-sample report files.
