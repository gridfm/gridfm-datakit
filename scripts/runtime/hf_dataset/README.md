---
license: apache-2.0
tags:
  - power-systems
  - power-flow
  - optimal-power-flow
  - powermodels
---

# gridfm-datakit PowerModels setup-2 scenarios

Paper: **[gridfm-datakit-v1: A Python Library for Scalable and Realistic Power Flow and Optimal Power Flow Data Generation](https://arxiv.org/abs/2512.14658)**

The **10,000 PowerModels JSON scenarios per grid** used in the classical-solver
runtime (setup 2 / from-disk) of that paper: AC-PF, DC-PF, AC-OPF, and DC-OPF
wall-time figures. Files are the corrected pool
(`scenario_*_corrected.json`). Indices wrap when the matrix asks for more than
10,000 solves.

Code and the 56 result CSVs live in `scripts/runtime/` on the
**`genco-paper-repro`** branch of
[`gridfm/gridfm-datakit`](https://github.com/gridfm/gridfm-datakit/tree/genco-paper-repro/scripts/runtime)
(see that directory’s README). Default `main` does not include this tree.

Data-diversity figures (different snapshot, `case118_ieee` parquet) live at
[`gridfm/reproducibility-datakit-technical-report`](https://huggingface.co/datasets/gridfm/reproducibility-datakit-technical-report).

## Layout

Matches `$GRIDFM_DATA_BASE/{pf,opf}/<network>/powermodels/scenario_*_corrected.json`.

| Path | Scenarios | Size |
|------|----------:|-----:|
| `pf/case14_ieee/powermodels/` | 10,000 | 0.09 GiB |
| `pf/case30_ieee/powermodels/` | 10,000 | 0.18 GiB |
| `pf/case57_ieee/powermodels/` | 10,000 | 0.33 GiB |
| `pf/case118_ieee/powermodels/` | 10,000 | 0.85 GiB |
| `pf/case500_goc/powermodels/` | 10,000 | 3.5 GiB |
| `pf/case2000_goc/powermodels/` | 10,000 | 14.5 GiB |
| `pf/case10000_goc/powermodels/` | 10,000 | 58.6 GiB |
| `opf/case14_ieee/powermodels/` | 10,000 | 0.09 GiB |
| `opf/case30_ieee/powermodels/` | 10,000 | 0.18 GiB |
| `opf/case57_ieee/powermodels/` | 10,000 | 0.33 GiB |
| `opf/case118_ieee/powermodels/` | 10,000 | 0.85 GiB |
| `opf/case500_goc/powermodels/` | 10,000 | 3.5 GiB |
| `opf/case2000_goc/powermodels/` | 10,000 | 14.5 GiB |
| `opf/case10000_goc/powermodels/` | 10,000 | 58.6 GiB |

**Total:** 140,000 files, about **156 GiB**. PF and OPF splits are both required: PF/DCPF jobs read `pf/`, OPF/DCOPF jobs read `opf/`. GOC 2000/10000 are most of the bytes.

## Download

The full tree is large (GOC 2000/10000 dominate). Download only what you need:

```bash
# IEEE + case500 only (small jobs)
hf download gridfm/reproducibility-powermodels-setup2 --repo-type dataset \
    --local-dir /path/to/finetuning \
    --exclude "pf/case2000_goc/**" --exclude "pf/case10000_goc/**" \
    --exclude "opf/case2000_goc/**" --exclude "opf/case10000_goc/**"

# one network
hf download gridfm/reproducibility-powermodels-setup2 --repo-type dataset \
    --local-dir /path/to/finetuning \
    --include "pf/case14_ieee/**" --include "opf/case14_ieee/**"
```

Then:

```bash
export GRIDFM_DATA_BASE=/path/to/finetuning   # required; there is no cluster default
# already scenario_*_corrected.json — do not run run_correction.sh
bash scripts/runtime/pure_julia/run_small_setup2.sh
```

Setup 1 does not use this dataset (it solves the corrected `.m` base case in memory).

## Notes

- Uncorrected `scenario_*.json` files are **not** included; only `*_corrected.json`.
- The `apache-2.0` licence applies to this converted snapshot and the datakit code.
  Underlying PGLib / GOC network models retain their original terms.
