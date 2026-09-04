# Reproducing the PowerModels runtime results

This directory holds the **classical-solver** (PowerModels) runtime matrix used in
the paper figures: amortized per-instance wall time for AC-PF, DC-PF, AC-OPF, and
DC-OPF, under an in-memory protocol (setup 1) and a from-disk protocol (setup 2).

The 56 committed CSVs **are** the paper raw results. You only need to re-run the
jobs if you want to regenerate them.

This tree lives on the **`genco-paper-repro`** branch, not on default `main`:

```bash
git clone -b genco-paper-repro https://github.com/gridfm/gridfm-datakit.git
```

GENCO GPU numbers are **not** produced here. Data-diversity figures:
[`paper/repro/README.md`](../../paper/repro/README.md).

## 1. Install

Julia **1.12.6**, the version used on the timed jobs. From the repository root:

```bash
julia --project=scripts/runtime/pure_julia -e 'using Pkg; Pkg.instantiate()'
```

Exact package pins, Ipopt/MUMPS versions, thread settings, and original hardware
are in
[`outputs_julia/full_matrix/environment_versions.md`](outputs_julia/full_matrix/environment_versions.md).
Instantiate from [`pure_julia/Manifest.toml`](pure_julia/Manifest.toml). That
Manifest was resolved under Julia 1.11 (its header says `1.11.8`); reproduce with
**1.12.6**.

Setup 1 also needs the seven corrected `.m` files in `gridfm_datakit/grids/`
(they are tracked on `genco-paper-repro`).

## 2. Read the paper numbers

They live at `scripts/runtime/outputs_julia/full_matrix/`. Do not edit the 56 CSVs.

| File | What it is |
|------|------------|
| `{small,large}/setup1/benchmark_<grid>_<mode>.csv` | In-memory protocol: worker sweep for AC-PF, DC-PF, AC-OPF, DC-OPF (main scaling figures + appendix worker-count figure) |
| `{small,large}/setup2/benchmark_<grid>_<mode>.csv` | From-disk protocol: same modes (loading-ratio figure and speedup table) |
| `wall_at_best_p_setup1.csv` | In-memory protocol: wall time at the best `p` |
| `wall_at_best_p_setup2.csv` | From-disk protocol: same, independently chosen best `p` |
| `wall_at_best_p_long.csv` | Long form of both (loading-ratio figure) |
| `wall_at_best_p.md` | Human-readable best-`p` table |
| [`environment_versions.md`](outputs_julia/full_matrix/environment_versions.md) | Julia, PowerModels, Ipopt, MUMPS, threads, and hardware of the original run |
| [`methodology_parameters.md`](outputs_julia/full_matrix/methodology_parameters.md) | Shared, GENCO, and PowerModels parameters used for the matrix |
| [`lsf_job_wall_times.md`](outputs_julia/full_matrix/lsf_job_wall_times.md) | LSF job IDs, wall, and peak RAM of the four paper jobs |

**Metric.** Amortized per-instance runtime = `pf_elapsed_s / n_pfs`. Best `p` =
`argmin` of that ratio. Never use `mean_pf_runtime_s`. Init, compile, and `/tmp`
staging are outside `pf_elapsed_s`. The 56 sweep CSVs are the source of truth;
the `wall_at_best_p*` files are derived from them.

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

Modes: `pf`, `dcpf`, `opf`, `dcopf`. AC-PF uses NLsolve (`--pf-fast`) through
case500 and Ipopt on case2000/10000. Worker sweep: `p = 24, 40, …, 216`
(13 points). Dispatch batch 32 (small, including case500) / 1 (case2000/10000).

## 3. Re-run the matrix

**Setup 1 (in-memory)** — repeated solves of the corrected base `.m` case. No
scenario JSON.

```bash
bash scripts/runtime/pure_julia/run_small_setup1.sh
bash scripts/runtime/pure_julia/run_large_setup1.sh
```

**Setup 2 (from-disk)** — parse a datakit scenario per solve. The JSON pool is
not in git. The 10,000 corrected files per grid used in the paper are on Hugging
Face:
[`gridfm/reproducibility-powermodels-setup2`](https://huggingface.co/datasets/gridfm/reproducibility-powermodels-setup2)
(~156 GiB for all seven networks × PF and OPF).

```bash
hf download gridfm/reproducibility-powermodels-setup2 --repo-type dataset \
    --local-dir /path/to/finetuning
export GRIDFM_DATA_BASE=/path/to/finetuning   # required; there is no cluster default
# already scenario_*_corrected.json — skip run_correction.sh
bash scripts/runtime/pure_julia/run_small_setup2.sh
bash scripts/runtime/pure_julia/run_large_setup2.sh
```

Expected layout:

```text
$GRIDFM_DATA_BASE/{pf,opf}/<network>/powermodels/scenario_*_corrected.json
```

`GRIDFM_DATA_BASE` must be set for setup 2 and for `run_correction.sh`. At most
10,000 distinct scenarios per network (indices wrap). Stage to node-local `/tmp`
before the timed region.

If you convert from parquet yourself instead of using the Hugging Face snapshot:

```bash
export GRIDFM_DATA_BASE=/path/to/finetuning
python scripts/convert/batch_convert_finetune.py --pf-base "$GRIDFM_DATA_BASE/pf" --opf-base "$GRIDFM_DATA_BASE/opf"
bash scripts/runtime/pure_julia/run_correction.sh
bash scripts/runtime/pure_julia/run_small_setup2.sh
bash scripts/runtime/pure_julia/run_large_setup2.sh
```

On LSF (84 cores; 256 G small / 960 G large; original hosts were CCC 7xx):

```bash
bash scripts/runtime/pure_julia/submit_matrix.sh
```

`--resume` is always on: delete a CSV first to recompute that cell.

The converter implementation is `gridfm_datakit/convert/`; `scripts/convert/` is
the CLI over the finetuning tree.


## Notes

- Ipopt: `max_iter=100`, `tol=1e-6`, linear solver MUMPS (default; not ma57).
  One BLAS thread per worker (`JULIA_NUM_THREADS=1` and matching OpenBLAS/OMP/MKL
  flags in `pure_julia/env.sh`).
- Original hardware: AMD EPYC 9634, 84 cores. Details:
  [`outputs_julia/full_matrix/environment_versions.md`](outputs_julia/full_matrix/environment_versions.md).
- Observed job RAM: small ~205 GB, large ~870 GB. See
  [`outputs_julia/full_matrix/lsf_job_wall_times.md`](outputs_julia/full_matrix/lsf_job_wall_times.md).
- Full parameter table:
  [`outputs_julia/full_matrix/methodology_parameters.md`](outputs_julia/full_matrix/methodology_parameters.md).
