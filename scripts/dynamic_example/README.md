# Dynamic simulation example (IEEE14 + Dynawo)

A self-contained, end-to-end run of the dynamic-simulation pipeline: load
scenarios → balanced initial state (OPF → update_powsybl → AC power flow) →
Dynawo time-domain simulation → Parquet + Zarr + metadata + reports.

The scenario is an IEEE14 network where generator `_GEN____2_SM` is disconnected
at `t = 50 s`; the resulting voltage/field transients are recorded per scenario.

## Prerequisites

- `pip install 'gridfm-datakit[dynamic]'` (pulls in `pypowsybl[dynamic]` and
  `zarr`; the plain `powsybl` extra is not enough).
- **Dynawo** installed and referenced from `~/.itools/config.yml`
  (`dynawo: { homeDir: /path/to/dynawo }`). Without it pypowsybl raises
  `DynawoSimulationProvider could not be instantiated`.
- Julia is bootstrapped automatically on first run (for the OPF step).

## Run

```bash
python scripts/dynamic_example/run.py     # from the project root
# or, from this folder:
python run.py
```

The CLI runs the same pipeline. A config carrying a `dynamic:` block selects it,
so there is no separate subcommand — but note the CLI does not resolve this
example's folder-relative paths, which is what `run.py` is for:

```bash
gridfm-datakit generate config.yaml       # paths must be absolute
```

Progress logging streams advancement to the console as it runs:

```
HH:MM:SS INFO    gridfm_datakit.dynamic | Dynamic generation: 6 scenarios in 2 chunk(s), 2 worker(s).
HH:MM:SS INFO    gridfm_datakit.dynamic | Chunk 1/2 done (3 scenarios) — 3 samples so far.
HH:MM:SS INFO    gridfm_datakit.dynamic | Chunk 2/2 done (3 scenarios) — 6 samples so far.
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
`out/`). There is no separate dynamic output directory — the layout reuses the
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
        │       (scenario_index, perturbation_index)
        ├── dynamic_results.zarr/
        │       curves (n_samples, n_variables, n_timesteps) + scenario_index /
        │       perturbation_index coordinate arrays; join the two modalities on
        │       this key pair
        ├── reports/
        │       one Dynawo report (JSON) per sample — model build-up + convergence
        └── metadata.json
                variable names, dimensions, join-key index, config hash
```

The dynamic artifacts sit in their own `dynamic/` subfolder rather than directly
in `raw/` because the static pipeline writes `bus_data.parquet` as a *partitioned
directory* while the dynamic pipeline writes it as a flat file — same name,
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
  samples — one Dynawo run per perturbed topology, each labelled by
  `perturbation_index`. This is the only perturbation that adds *dynamic*
  diversity.
- **`generation_perturbation` / `admittance_perturbation` are not supported
  here.** They are accepted for parity with the static pipeline, but
  `generation_perturbation` randomises generation *cost*, so in the dynamic
  pipeline both only shift the OPF dispatch — the initial operating point Dynawo
  starts from — and produce no dynamic-model or event variation. They are not
  validated against the dynamic outputs.
- **Quieter/louder:** `dynamic.logging.verbosity` (`silent|error|warning|info|debug`);
  set `settings.enable_solver_logs: false` to skip solver-log capture;
  `dynamic.logging.save_reports: false` to skip report files.
