# Handoff: prepare a GitHub reproducibility branch from `paper_backup`

**For:** the agent that will clean this repo and push a public reproducibility branch.  
**Repo:** `gridfm/gridfm-datakit` (`origin` = `git@github.com:gridfm/gridfm-datakit.git`).  
**Local branch under discussion:** `paper_backup` at `d154788` (not on GitHub yet).  
**Do not push `paper_backup` as-is.** It has the paper PowerModels runtime results, but it also has junk, gitignore traps, and missing env/grid files.

---

## Direct answer

`paper_backup` contains **the paper PowerModels Julia runtime scripts and the 56 CSVs that those scripts produced**. That is the most important artifact and it **is** on the branch (force-added despite `.gitignore`).

It does **not** contain everything needed to *re-run* those scripts from a fresh clone:

| Needed to re-run the paper matrix | On `paper_backup`? |
|-----------------------------------|--------------------|
| Engine + launchers + exact args   | Yes (`scripts/runtime/pure_julia/`) |
| Paper result CSVs + wall-at-best-p tables | Yes (`scripts/runtime/outputs_julia/full_matrix/`, 56/56, 13 `p` points each) |
| Julia `Project.toml` / `Manifest.toml` | **No** (lives in gitignored `venv/julia_env/`) |
| Corrected `.m` grids for setup 1  | **No** (gitignored `gridfm_datakit/grids/*.m`) |
| Finetuning scenario JSONs for setup 2 | **No** (external `/dccstor/gridfm/powermodels_data/v4/finetuning`) |
| LSF logs of the four paper jobs   | **No** (only under `~/.lsbatch/`) |
| GENCO inference scripts/outputs   | **No** (other repo: `gridfm_model_evaluation`) |

Treat `paper_backup` as the **source snapshot**, not the branch to publish. Build a new branch from it, keep the runtime keep-set, drop the junk, add the missing env/grids, then push.

---

## Git facts (do not merge `main` first)

```
paper_backup  d154788  Add Julia benchmark matrix outputs   (HEAD, local only)
              070deb0  Backup of paper branch working state
              7a9c3c5  add per-solve scenario loading…      (also local branch `paper`)
              4b2fa22  added converter                      (origin/paper)
              206f8cc  updated scripts
              02cb650  tag 1.0.4 / old main
```

- `paper_backup` is **5 commits ahead of local `main`**, **3 ahead of `origin/paper`**, **92 commits behind `origin/main`**.
- The paper numbers were produced by the code on this lineage, **not** current `origin/main`.
- **Do not rebase onto `origin/main`.** A reproducibility branch must pin the code that produced the CSVs. Security-only cherry-picks from main are optional and out of scope unless the user asks.

Suggested published branch name: `genco-paper-repro`, created from `paper_backup` after the cleanup below.

---

## Priority: paper runtime (keep this, make it clone-visible)

This is what the paper PowerModels tables come from. Full-matrix completion ~2026-07-13. Jobs: small s1 `951513`, large s1 `951514`, small s2 `955074`, large s2 `955075`.

### Tracked keep-set (already on `paper_backup`)

```
scripts/runtime/pure_julia/
  benchmark_distributed.jl   # engine (Distributed + pmap)
  matrix_config.jl           # counts, p-sweep, batch, pf_fast, DATA_BASE
  run_matrix.jl              # expands config → exact CLI args
  scenario_staging.jl        # setup2: stage ≤10k JSON to /tmp
  correct_scenarios.jl       # rewrite scenario_*_corrected.json (setup2 prereq)
  run_correction.sh
  run_{small,large}_setup{1,2}.sh
  submit_matrix.sh           # LSF wrapper for the 5 jobs

scripts/runtime/outputs_julia/full_matrix/
  {small,large}/{setup1,setup2}/benchmark_<network>_<mode>.csv   # 56 files
  wall_at_best_p.md
  wall_at_best_p_setup1.csv
  wall_at_best_p_setup2.csv
  wall_at_best_p_long.csv
  lsf_job_wall_times.md
  methodology_parameters.md
  environment_versions.md    # currently 1 line — expand it (see gaps)

scripts/runtime/experiments/powermodels_genco_handoff.md
```

CSV check: **56 files**, each **14 lines** (header + `p` = 24,40,…,216). Header:

```
p,opf_elapsed_s,init_elapsed_s,pf_elapsed_s,n_pfs,...,mean_pf_runtime_s,mean_pf_solve_time_s,...,mean_parse_time_s,successful_count,failed_count,first_error
```

**Metric rule (paper):** best `p` = `argmin(pf_elapsed_s / n_pfs)`. Never use `mean_pf_runtime_s` to pick best p. Timed region is `pf_elapsed_s` only (init, OPF prep, `/tmp` staging are outside it).

`.gitignore` still has `scripts/runtime/outputs_julia/`. The CSVs are tracked only because they were `git add -f`. **Fix gitignore** so a later `git add` cannot drop them, e.g.:

```
scripts/runtime/outputs_julia/
!scripts/runtime/outputs_julia/full_matrix/
!scripts/runtime/outputs_julia/full_matrix/**
```

Do **not** add `scripts/runtime/trash/` or the old Python+juliacall CSVs. Those are not the paper matrix.

### Exact command chain that produced the 56 CSVs

User-facing entry:

```bash
bash scripts/runtime/pure_julia/submit_matrix.sh
```

That submits (LSF `bsub`):

| Job name | Script | `-M` / `-n` | Notes |
|----------|--------|-------------|-------|
| `julia_matrix_correct` | `run_correction.sh` | 128G / 84 | setup2 only; non-7xx hosts |
| `julia_matrix_small_s1` | `run_small_setup1.sh` | 256G / 84 | no dep |
| `julia_matrix_large_s1` | `run_large_setup1.sh` | 960G / 84 | no dep |
| `julia_matrix_small_s2` | `run_small_setup2.sh` | 256G / 84 | after correction |
| `julia_matrix_large_s2` | `run_large_setup2.sh` | 960G / 84 | after correction |

Each `run_*_setup*.sh` does:

```bash
source "$REPO_ROOT/venv/bin/activate"
export JULIA_PROJECT="$REPO_ROOT/venv/julia_env"
export JULIA_CPU_TARGET=generic
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export JULIA_DEPOT_PATH="${TMPDIR:-/tmp}/gridfm_julia_${LSB_JOBID}_$$:$HOME/.julia_shared:$HOME/.julia"
julia --project="$JULIA_PROJECT" run_matrix.jl --scope {small|large} --setup {setup1|setup2}
```

`run_matrix.jl` then runs, for every network in that scope × modes `pf,dcpf,opf,dcopf`:

```text
julia benchmark_distributed.jl
  --setup cached-base|per-solve-load
  --network <case14_ieee|…|case10000_goc>
  --mode <pf|dcpf|opf|dcopf>
  --n-pfs <count from matrix_config.jl>
  --process-start 24 --process-stop 216 --process-step 16
  --dispatch-batch-size 32        # small
  --dispatch-batch-size 1         # large
  --output-csv scripts/runtime/outputs_julia/full_matrix/{scope}/{setup}/benchmark_<net>_<mode>.csv
  --resume
  --init-timeout-s 900
  --pf-fast | --no-pf-fast        # pf mode only; see table
  setup1: --case-file gridfm_datakit/grids/pglib_opf_<network>_corrected.m
  setup2: --data-base /dccstor/gridfm/powermodels_data/v4/finetuning
```

Ipopt (when used): `--max-iter 100 --tol 1e-6 --print-level 0` (CLI defaults in `benchmark_distributed.jl`). Linear solver **not set** → Ipopt default **MUMPS 5.8.2**. Do not enable HSL ma57.

`matrix_config.jl` (the paper counts):

| network | n_pfs | scope | pf_fast |
|---------|------:|-------|---------|
| case14_ieee | 4_000_000 | small | true (`compute_ac_pf`) |
| case30_ieee | 3_000_000 | small | true |
| case57_ieee | 2_000_000 | small | true |
| case118_ieee | 2_000_000 | small | true |
| case500_goc | 500_000 | small | true |
| case2000_goc | 50_000 | large | **false** (`solve_ac_pf` + Ipopt) |
| case10000_goc | 10_000 | large | **false** |

Modes: PF/DCPF scenarios from `…/finetuning/pf/<network>/`; OPF/DCOPF from `…/finetuning/opf/<network>/`. Pool ≤10_000 files; extra solves wrap with modulo. Setup2 stages that pool to node-local `/tmp` **before** the timed region.

`--resume` is always on: to fully redo a cell, **move/delete that CSV first**.

Hardcoded paths to parameterize before public repro (do not leave IBM CCC paths as the only option):

- `submit_matrix.sh`: `REPO_ROOT="/u/apu/gridfm-datakit"` (the four `run_*_setup*.sh` already compute repo root; only submit is sticky)
- `matrix_config.jl` / `correct_scenarios.jl` / `benchmark_distributed.jl`: `DATA_BASE = "/dccstor/gridfm/powermodels_data/v4/finetuning"`
- LSF host `select[hname=='cccxc702' … 'cccxc716']`

Keep those as **documented defaults for the original run**, but add env vars / flags so someone else can point at their data.

---

## File catalog: `paper_backup` vs `main`

Everything below is relative to `git diff --name-status main...paper_backup`. Grouped by what the cleanup agent should do.

### A. Keep — paper runtime (see previous section)

Already listed. This is the reason the branch exists.

### B. Keep — data generation, conversion, and related launchers (code, not the 10k JSON files)

These are on `paper_backup` (some already on `origin/paper`):

| Path | Role |
|------|------|
| `scripts/data_gen/launcher/core/launch_finetuning_data_gen_pf.py` | Writes per-grid YAML from `default_pf.yaml` and `bsub`s `gridfm_datakit generate`. **case10000 PF is commented out** in the current file; the paper still used case10000 PF scenarios that already exist on dccstor. |
| `scripts/data_gen/launcher/core/launch_finetuning_data_gen_opf.py` | Same for OPF. Current file has **only case10000 uncommented** (other grids commented). Do not treat this file as a complete replay of all OPF grids without restoring the commented tuples. |
| `scripts/config/default_pf.yaml` | PF generation defaults (`scenarios: 10000`, perturbations, `pf_fast`, etc.). Launchers override `data_dir` to `/dccstor/gridfm/powermodels_data/v4/finetuning/pf`. |
| `scripts/config/default_opf.yaml` | OPF generation defaults. Launchers override `data_dir` to `…/v4/finetuning/opf`. |
| `gridfm_datakit/convert/parquet_to_powermodels.py` | Parquet row → PowerModels JSON. |
| `gridfm_datakit/convert/batch_parquet_to_powermodels.py` | Batch wrapper. |
| `gridfm_datakit/convert/roundtrip_check.py` | Roundtrip validation. |
| `scripts/convert/batch_convert_finetune.py` | Convert first 10k PF/OPF parquet scenarios → JSON under the finetuning trees. Args: `--max-samples 10000`. |
| `scripts/convert/validate_parquet_roundtrip.py` + `submit_validate_parquet_roundtrip.sh` | Roundtrip job. |
| `scripts/runtime/pure_julia/correct_scenarios.jl` | After JSON convert: write `scenario_*_corrected.json` (setup2 reads **corrected** files). |
| `scripts/data_gen/launcher/contingency/*` | Contingency finetuning launchers (`case300` train/test, Texas). **Keep.** |
| `scripts/data_gen/launcher/pretraining/*` | Pretraining / pretraining-eval PF launchers. **Keep.** |
| `pfdelta/build_task_splits_from_data_processed.py` | PF-delta task-split builder. **Keep.** |

**Data-gen Ipopt `max_iter` in the YAML launchers is not the runtime-benchmark `max_iter=100`.** Example PF launcher: case14 `50`, case118 `100`, case500 `120`. OPF launcher case10000: `320`. Document that split; do not “unify” them.

Generated per-grid YAMLs under `scripts/config/finetuning/` and `scripts/config/generated/` are **not tracked** (`generated/` is gitignored). The launchers recreate them.

### C. Keep with care — supporting paper-adjacent code

| Path | Role | Action |
|------|------|--------|
| `docs/dc_slack_residual_correction.md` | Documents DC slack residual fix. | Keep. |
| `dc_res_lsf/compute_dc_residuals.py` + `dc_res_lsf.sh` | Job that produced residual stats. | Keep scripts; **drop** `lsf_13100.err/.out` unless the paper cites that exact log. Keep `out_n1000/` only if the paper uses those numbers. |
| `gridfm_datakit/utils/stats.py` | Small change vs main (stats used by residual/docs). | Keep. |
| `scripts/utils/*` | Moved helpers (`compare_parquet_files.py`, `parse_ipopt_logs.py`, …). | Keep the move; drop `scripts/utils/matteo_config.yaml` if it is personal (already deleted on `paper_backup`). |
| `tests/test_parquet_to_powermodels.py` | Converter tests. | Keep the test. |
| `tests/fixtures/parquet_json_roundtrip/` | Tiny parquet fixtures (not the 10k scenario pool). | Keep; they are already on `origin/paper`. |

### D. Drop before GitHub (junk / oversized / not paper)

| Path | Why drop |
|------|----------|
| `runtime.zip` (42 MB) | Snapshot of old `scripts/runtime/**` including huge LSF/roundtrip logs (`bsub.out` ~200 MB uncompressed). Not the paper matrix. GitHub warn ≥50 MB. |
| `tests/test_parquet_to_powermodels.log` (6 MB) | Accidental log check-in. |
| `cmd_benchmark` | Personal scratch `bsub` history (juliacall + GENCO). Not a launcher. |
| `qeon/IEEE118_input_features.csv` + `IEEE118_target_outputs.csv` (~5.5 MB) | Unrelated dump. |
| `v_and_s`, `v_and_s_all` | Scratch. |

### E. Do not add from the working tree (gitignored, not the paper matrix)

```
scripts/runtime/trash/          # pilots, python+juliacall, superseded launchers
scripts/runtime/outputs/        # older python CSVs
venv/                           # except copy Project.toml+Manifest.toml out
~/.lsbatch/julia_matrix_*.out   # optional; summaries already in lsf_job_wall_times.md
```

`scripts/runtime/trash/README.md` is a useful map of what was moved; you may copy a short “out of tree” note into `experiments/` instead of publishing `trash/`.

### F. Already deleted vs `origin/paper` (correct)

`070deb0` removed the tracked Python+juliacall matrix under `scripts/runtime/benchmark_juliacall_matrix/`. That is **intentional**. The paper classical solver numbers are **pure Julia**, not juliacall. Do not restore those files on the repro branch.

---

## Gaps to close on the repro branch (required)

### 1. Julia environment (currently untracked)

Copy into the repo, e.g. `scripts/runtime/pure_julia/`:

- `/u/apu/gridfm-datakit/venv/julia_env/Project.toml`
- `/u/apu/gridfm-datakit/venv/julia_env/Manifest.toml`

Documented paper versions:

- Runtime Julia used on the cluster: **1.12.6** (`environment_versions.md`)
- `Manifest.toml` header currently says `julia_version = "1.11.8"` — **do not hide this**. Pin what was actually executed (1.12.6) and keep the Manifest as the package set (PowerModels **0.21.5**, Ipopt.jl **1.14.0**, Ipopt_jll **300.1400.1901** → Ipopt 3.14.19).
- Threading: `JULIA_NUM_THREADS=1` and BLAS/OMP/MKL = 1 on workers.

Expand `environment_versions.md` from this Manifest (the tracked file is currently a single line). Include Python only if you keep data-gen in the same branch (`pyproject.toml` is unchanged vs old main / tag 1.0.4).

Point `JULIA_PROJECT` at the in-repo project, not `venv/julia_env`, so a clone works.

### 2. Setup-1 grids (currently gitignored)

Un-ignore at least the **corrected** cases the matrix reads:

```
gridfm_datakit/grids/pglib_opf_case{14,30,57,118}_ieee_corrected.m
gridfm_datakit/grids/pglib_opf_case{500,2000,10000}_goc_corrected.m
```

They exist on disk today; only `__init__.py` is tracked. Confirm PGLIB license text is already in the repo before adding `.m` files.

### 3. Setup-2 scenario data (too large for GitHub)

Do **not** commit `/dccstor/gridfm/powermodels_data/v4/finetuning`. Document:

```
{DATA_BASE}/{pf|opf}/<network>/powermodels/scenario_*_corrected.json
```

up to 10_000 per (split, network). Pipeline if someone must rebuild:

1. `launch_finetuning_data_gen_{pf,opf}.py` → parquet under `…/v4/finetuning/{pf,opf}`
2. `scripts/convert/batch_convert_finetune.py --max-samples 10000`
3. `julia correct_scenarios.jl --workers N`
4. Then `run_*_setup2.sh`

Without those JSON files, **setup1 can still be reproduced** from the `.m` cases; setup2 cannot.

### 4. Parameterize CCC-only bits

Minimum: `DATA_BASE`, `REPO_ROOT`, LSF resource strings. Keep original values in comments / `methodology_parameters.md`.

### 5. Optional but useful

- Copy the four LSF `.out` footers into `outputs_julia/full_matrix/lsf/` **or** leave `lsf_job_wall_times.md` as the record (already has wall days + RAM).
- A 20-line `scripts/runtime/README.md`: how to read the CSVs, how to launch, what is out of tree.

---

## What this repo will still not reproduce

**GENCO** numbers live in `/u/apu/gridfm_model_evaluation` (`scripts/benchmark_inference/pf_benchmark_matrix/…`). `cmd_benchmark` has the GPU `bsub` lines; do not treat that as the GENCO repro package. Shared sample counts are in `matrix_config.jl` (4M/3M/2M/2M/500k/50k/10k). GENCO comparison notes: `experiments/powermodels_genco_handoff.md`.

Canvases used while planning counts (`twenty-second-benchmark-counts.canvas.tsx`) are **outside git** under Cursor project files. The **ran** counts are only `matrix_config.jl`.

---

## Suggested agent procedure

1. Create branch `genco-paper-repro` from `paper_backup` (`d154788`). **Do not merge `origin/main`.**
2. Delete paths in section D (`runtime.zip`, test log, `cmd_benchmark`, `qeon/`, `v_and_s*`). **Keep** `pfdelta/build_task_splits_from_data_processed.py` and all `scripts/data_gen/launcher/{core,contingency,pretraining}/*`.
3. Copy Julia `Project.toml` + `Manifest.toml` into `scripts/runtime/pure_julia/`. Retarget launchers’ `JULIA_PROJECT`.
4. Force-add the seven corrected `.m` grids; relax `.gitignore` for those names.
5. Fix `.gitignore` so `outputs_julia/full_matrix/**` stays tracked.
6. Expand `environment_versions.md`; add a short `scripts/runtime/README.md` with the command chain and metric rule from this note.
7. Replace hardcoded `REPO_ROOT` in `submit_matrix.sh`; keep `DATA_BASE` overridable.
8. Confirm `git ls-files scripts/runtime/outputs_julia | wc -l` is **63** (56 CSVs + 7 markdown/csv summaries) and every CSV has 13 p-rows.
9. `git push -u origin genco-paper-repro` only after the user confirms. Open a PR against `main` **or** leave as a protected paper branch — ask; merging into `main` is a product decision, not required for reproducibility.

Do not rewrite the 56 CSVs. Do not “fix” `pf_fast` on case2000/10000. Do not switch Ipopt to ma57. Do not use `mean_pf_runtime_s` in any derived table you add.
