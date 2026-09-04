# Environment used for the paper PowerModels runtime matrix

Cluster jobs completed ~2026-07-13 (`julia_matrix_{small,large}_s{1,2}`).

## Julia runtime (executed)

- **Julia 1.12.6** (`JULIA_CPU_TARGET=generic`)
- `JULIA_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`

The `Manifest.toml` next to the launchers was exported from the job environment and records
`julia_version = "1.11.8"` in its header (the environment was originally resolved under 1.11).
Reproduce with **1.12.6** as used on the timed jobs; instantiate from this Manifest.

## Packages (from `scripts/runtime/pure_julia/Manifest.toml`)

| Package | Version |
|---------|---------|
| PowerModels | 0.21.5 |
| Ipopt.jl | 1.14.0 |
| Ipopt_jll | 300.1400.1901 → **Ipopt 3.14.19** |
| Memento | 1.4.1 |
| PythonCall | 0.9.31 (present in the env; **not** used by the pure-Julia matrix) |

Ipopt linear solver: **MUMPS 5.8.2** (Ipopt default; not overridden). HSL ma57 was not used.

## Hardware (original run)

- AMD EPYC 9634, 84 cores, one LSF host per job
- RAM request / observed peak: small 256 G / ~205 G; large 960 G / ~870 G
- See `lsf_job_wall_times.md`
