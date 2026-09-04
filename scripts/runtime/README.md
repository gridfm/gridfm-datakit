# Traditional-solver runtime (paper)

This directory is the **PowerModels / Julia** side of the GENCO paper runtime comparison.
GENCO GPU numbers live in `gridfm_model_evaluation`; this repo only ships the classical-solver
raw results that those figures consume.

Paper mapping (Joule manuscript):

| Paper object | Protocol | Source here |
|--------------|----------|-------------|
| Fig. runtime vs grid size (PF / OPF) | in-memory = **setup 1** | `outputs_julia/full_matrix/wall_at_best_p_setup1.csv` |
| Fig. wall time vs worker count | in-memory = **setup 1** | `outputs_julia/full_matrix/small|large/setup1/*.csv` |
| Fig. from-disk / in-memory ratio | setup 2 / setup 1 at independently chosen best `p` | `wall_at_best_p_setup1.csv` and `wall_at_best_p_setup2.csv` |
| Speedup tables vs AC-PF / AC-OPF | both protocols | same CSVs; GENCO times from the other repo |

**Raw results:** 56 CSVs under `outputs_julia/full_matrix/{small,large}/{setup1,setup2}/`.
Do not edit them. Summaries (`wall_at_best_p*.csv`) are derived.

## Metric (do not change)

Amortized per-instance wall time = `pf_elapsed_s / n_pfs`.

Best worker count `p` = `argmin(pf_elapsed_s / n_pfs)`.
**Never** use `mean_pf_runtime_s` to pick best `p`.

`pf_elapsed_s` is the timed solve region only (worker init, Julia compile, OPF prep, and `/tmp` staging are outside it).

## What was run

| Item | Value |
|------|--------|
| Engine | `pure_julia/benchmark_distributed.jl` |
| Config | `pure_julia/matrix_config.jl` |
| Modes | `pf`, `dcpf`, `opf`, `dcopf` |
| `p` | 24, 40, …, 216 (step 16; 13 points) |
| Dispatch batch | 32 (small grids) / 1 (GOC 2000/10000) |
| Ipopt | `max_iter=100`, `tol=1e-6`, `print_level=0`, linear solver **MUMPS** (default) |
| AC-PF | `compute_ac_pf` (NLsolve) on IEEE 14–GOC 500; `solve_ac_pf`+Ipopt on GOC 2000/10000 |
| Setup 1 | one parsed **corrected** `.m` base case per worker (`cached-base`) |
| Setup 2 | up to 10 000 `scenario_*_corrected.json` staged to node-local `/tmp`, parse inside the timer |
| Sample counts | 4M / 3M / 2M / 2M / 500k / 50k / 10k (case14 → case10000) |

Paper jobs (LSF): small s1 `951513`, large s1 `951514`, small s2 `955074`, large s2 `955075`.
See `outputs_julia/full_matrix/lsf_job_wall_times.md`.

## Reproduce (same args as the paper)

Needs Julia **1.12.6**, this directory as `JULIA_PROJECT` (`Project.toml` + `Manifest.toml`),
and the seven corrected `.m` files in `gridfm_datakit/grids/` (setup 1).

Setup 2 also needs the finetuning JSON pool (too large for git):

```text
$GRIDFM_DATA_BASE/{pf,opf}/<network>/powermodels/scenario_*_corrected.json
```

Original cluster path: `/dccstor/gridfm/powermodels_data/v4/finetuning`.

```bash
# instantiate once
julia --project=scripts/runtime/pure_julia -e 'using Pkg; Pkg.instantiate()'

# setup 1 (in-memory base case) — no scenario JSON required
bash scripts/runtime/pure_julia/run_small_setup1.sh
bash scripts/runtime/pure_julia/run_large_setup1.sh

# setup 2 (from-disk datakit scenarios)
export GRIDFM_DATA_BASE=/path/to/finetuning
bash scripts/runtime/pure_julia/run_correction.sh          # writes *_corrected.json
bash scripts/runtime/pure_julia/run_small_setup2.sh
bash scripts/runtime/pure_julia/run_large_setup2.sh
```

On LSF (original CCC 84-core 7xx nodes, 256 G small / 960 G large):

```bash
bash scripts/runtime/pure_julia/submit_matrix.sh
```

`--resume` is always on: delete a CSV first if you want that cell fully recomputed.

Rebuild summaries from the 56 CSVs:

```bash
python scripts/runtime/pure_julia/summarize_wall_at_best_p.py --check
```

## Environment

See `outputs_julia/full_matrix/environment_versions.md` and `methodology_parameters.md`.
