# Dynamic simulation example (IEEE14 + Dynawo)

A self-contained, end-to-end run of the dynamic-simulation pipeline: load
scenarios → balanced initial state (OPF → update_powsybl → AC power flow) →
Dynawo time-domain simulation → Parquet + Zarr + metadata + reports.

The scenario is an IEEE14 network where generator `_GEN____2_SM` is disconnected
at `t = 50 s`; the resulting voltage/field transients are recorded per scenario.

## Prerequisites

- `pip install 'pypowsybl[dynamic]'` and the package extras.
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

Full logging is enabled, so you see advancement as it runs:

```
HH:MM:SS INFO    gridfm_datakit.dynamic | Dynamic generation: 6 scenarios in 2 chunk(s), 2 worker(s).
HH:MM:SS INFO    gridfm_datakit.dynamic | Chunk 1/2 done (3 scenarios) — 3 samples so far.
HH:MM:SS INFO    gridfm_datakit.dynamic | Chunk 2/2 done (3 scenarios) — 6 samples so far.
HH:MM:SS INFO    gridfm_datakit.dynamic | Saved 6 samples to .../out (6 with dynamic results, 6 reports).
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

## Output (`out/`)

```
out/
├── bus_data.parquet / gen_data.parquet / branch_data.parquet
│       static PF snapshot (features), tagged with (scenario_index, perturbation_index)
├── dynamic_results.zarr/
│       curves (n_samples, n_variables, n_timesteps) + scenario_index / perturbation_index
│       coordinate arrays; join the two modalities on this key pair
├── reports/
│       one Dynawo report (JSON) per sample — model build-up + convergence
├── metadata.json
│       variable names, dimensions, join-key index, config hash
└── IEEE14/raw/solver_log/
        per-worker capture of OPF / PF / DCPF / Dynawo native output
```

## Variations

- **Faster/larger:** change `load.scenarios`, `settings.num_processes`,
  `settings.large_chunk_size`.
- **Topology perturbations:** set `topology_perturbation.type: random` (with
  `k` / `n_topology_variants` / `elements`) to expand each scenario into several
  samples — one Dynawo run per perturbed topology, each labelled by
  `perturbation_index`.
- **Quieter/louder:** `dynamic.logging.verbosity` (`silent|error|warning|info|debug`);
  set `settings.enable_solver_logs: false` to skip solver-log capture;
  `dynamic.logging.save_reports: false` to skip report files.
