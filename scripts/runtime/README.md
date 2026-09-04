# Reproducing the PowerModels runtime results

This directory holds the **classical-solver** (PowerModels) runtime matrix used in the paper figures: amortized per-instance wall time for AC-PF, DC-PF, AC-OPF, and DC-OPF, under an in-memory protocol (setup 1) and a from-disk protocol (setup 2).

GENCO GPU numbers are not produced here.

The 56 committed CSVs **are** the paper raw results. You only need to re-run the jobs if you want to regenerate them.

## 1. Install

Julia **1.12.6**. From the repository root:

```bash
julia --project=scripts/runtime/pure_julia -e 'using Pkg; Pkg.instantiate()'
```

Setup 1 also needs the seven corrected `.m` files in `gridfm_datakit/grids/` (they are tracked on this branch).

## 2. Read the paper numbers

They live at `scripts/runtime/outputs_julia/full_matrix/`. Do not edit the 56 CSVs.

| File | What it is |
|------|------------|
| `{small,large}/setup1/benchmark_<grid>_<mode>.csv` | In-memory protocol: worker sweep for AC-PF, DC-PF, AC-OPF, DC-OPF (main scaling figures + appendix worker-count figure) |
| `{small,large}/setup2/benchmark_<grid>_<mode>.csv` | From-disk protocol: same modes (loading-ratio figure and speedup table) |
| `wall_at_best_p_setup1.csv` | In-memory protocol: wall time at the best `p` (main PF/OPF scaling figures) |
| `wall_at_best_p_setup2.csv` | From-disk protocol: same, independently chosen best `p` |
| `wall_at_best_p_long.csv` | Long form of both (used for the loading-ratio figure) |
| `wall_at_best_p.md` | Human-readable best-`p` table |
| `methodology_parameters.md` | Shared, GENCO, and PowerModels parameters used for the matrix |

**Metric.** Amortized per-instance runtime = `pf_elapsed_s / n_pfs`. Best `p` = `argmin` of that ratio. Never use `mean_pf_runtime_s`. Init, compile, and `/tmp` staging are outside `pf_elapsed_s`. The 56 sweep CSVs are the source of truth; the `wall_at_best_p*` files are derived from them.

```bash
python scripts/runtime/pure_julia/summarize_wall_at_best_p.py --check
```

| Grid | Instances | Scope |
|------|----------:|-------|
| case14_ieee | 4,000,000 | small |
| case30_ieee | 3,000,000 | small |
| case57_ieee | 2,000,000 | small |
| case118_ieee | 2,000,000 | small |
| case500_goc | 500,000 | small |
| case2000_goc | 50,000 | large |
| case10000_goc | 10,000 | large |

Modes: `pf`, `dcpf`, `opf`, `dcopf`. AC-PF uses NLsolve (`--pf-fast`) through case500 and Ipopt on case2000/10000.

## 3. Re-run the matrix

**Setup 1 (in-memory)** — repeated solves of the corrected base `.m` case. No scenario JSON.

```bash
bash scripts/runtime/pure_julia/run_small_setup1.sh
bash scripts/runtime/pure_julia/run_large_setup1.sh
```

**Setup 2 (from-disk)** — parse a datakit scenario per solve. The JSON pool is not in git:

```text
$GRIDFM_DATA_BASE/{pf,opf}/<network>/powermodels/scenario_*_corrected.json
```

Original path: `/dccstor/gridfm/powermodels_data/v4/finetuning`.

```bash
export GRIDFM_DATA_BASE=/path/to/finetuning
bash scripts/runtime/pure_julia/run_correction.sh
bash scripts/runtime/pure_julia/run_small_setup2.sh
bash scripts/runtime/pure_julia/run_large_setup2.sh
```

On LSF (84 cores; 256 G small / 960 G large):

```bash
bash scripts/runtime/pure_julia/submit_matrix.sh
```

`--resume` is always on: delete a CSV first to recompute that cell.

## Notes

- Ipopt: `max_iter=100`, `tol=1e-6`, linear solver MUMPS (default). Dispatch batch 32 (small) / 1 (large). One BLAS thread per worker.
- Hardware of the original run: AMD EPYC 9634, 84 cores. Versions: `outputs_julia/full_matrix/environment_versions.md`.
- Full parameter table: [`outputs_julia/full_matrix/methodology_parameters.md`](outputs_julia/full_matrix/methodology_parameters.md).
