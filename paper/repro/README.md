# Reproducing the figures

This directory generates the data-diversity figures of *gridfm-datakit-v1*
([arXiv:2512.14658](https://arxiv.org/abs/2512.14658)): the entropy spider plots,
the feature violin plots, the branch-flow entropy barplot, and the branch-loading
histograms.

## 1. Install

Python 3.10–3.12. From the repository root:

```bash
pip install -e '.[dev,test]'
pip install seaborn        # used by plot_branch_loading.py; not a package dependency
```

Run everything from the repository root with `PYTHONPATH` set to it:

```bash
export PYTHONPATH=$PWD
```

The scripts import `gridfm_datakit` for `load_net_from_pglib` and the bus/generator
index constants. Depending on how the package was installed, the import may only
resolve from the repository root.

## 2. Get the data

The figures are computed from a snapshot of sampled parquet (~979 MB), hosted at
[`gridfm/gridfm-datakit-paper-figures`](https://huggingface.co/datasets/gridfm/gridfm-datakit-paper-figures)
(private — you need access and a HuggingFace token):

```bash
hf download gridfm/gridfm-datakit-paper-figures \
    --repo-type dataset \
    --local-dir paper/repro/dataset_sampled
```

The scripts expect it at `paper/repro/dataset_sampled/` (the default), or point them
elsewhere with `--data-dir`.

The snapshot holds six datasets on `case118_ieee`, each downsampled to exactly
10,000 scenarios × 118 buses so that every library contributes equally to the
diversity metrics:

| File prefix | Mode | Library |
|---|---|---|
| `gridfm_datakit_pf` | PF | this library, `mode: pf` |
| `gridfm_datakit_opf` | OPF | this library, `mode: opf` |
| `pfdelta` | PF | PFΔ |
| `opfdata` | OPF | OPFData |
| `pglearn` | OPF | PGLearn |
| `opflearn` | OPF | OPF-Learn |

Each has `_bus_data.parquet` and `_gen_data.parquet`; the two PF datasets also have
`_branch_data.parquet` (needed for the branch figures).

### The full, non-downsampled data

The same HuggingFace repo also holds the complete datasets under `full/`, for anyone
who needs more than the 10,000 scenarios used by the figures. **You do not need this to
reproduce the figures** — it is 15 GB against the snapshot's 1 GB.

```
gridfm/gridfm-datakit-paper-figures
├── *_bus_data.parquet, *_gen_data.parquet, *_branch_data.parquet   <- the 1 GB snapshot (repo root)
└── full/
    ├── gridfm_datakit_pf/     this library, mode: pf    199,207 scenarios
    ├── gridfm_datakit_opf/    this library, mode: opf   195,753 scenarios
    ├── opfdata/               OPFData                   300,000 scenarios
    ├── pglearn/               PGLearn                    96,852 scenarios
    ├── opflearn/              OPF-Learn                  10,000 scenarios
    └── pfdelta/               PFΔ                        29,000 scenarios
```

Fetch one dataset rather than the whole repo:

```bash
hf download gridfm/gridfm-datakit-paper-figures \
    --repo-type dataset \
    --include "full/gridfm_datakit_pf/*" \
    --local-dir /path/to/somewhere
```

The two `gridfm_datakit_*` directories are complete generation output, so they carry more
than the four tables the figures use:

| File | Contents |
|---|---|
| `bus_data.parquet` | per-bus `Pd, Qd, Pg, Qg, Vm, Va`, bus-type one-hots, voltage limits, shunts |
| `gen_data.parquet` | per-generator dispatch, limits, cost coefficients, status |
| `branch_data.parquet` | per-branch flows, impedances, admittance entries, `rate_a`, status |
| `y_bus_data.parquet` | nonzero admittance-matrix entries per scenario |
| `runtime_data.parquet` | per-sample AC and DC solver time |
| `stats.parquet`, `stats_plot.png` | output of the `gridfm_datakit stats` command |
| `scenarios_agg_load_profile.parquet` | the sampled load scenarios themselves |
| `args.log` | the resolved config for the run (see [`configs/`](configs/)) |
| `tqdm.log`, `error.log`, `solver_log/` | progress, per-sample failures, Ipopt logs |

`include_dc_res: true` was set for both runs, so the bus/gen/branch tables also carry
DC-PF or DC-OPF columns (`Va_dc`, `Pg_dc`, `p_mw_dc`, `pf_dc`, `pt_dc`) as ML baselines.

The `gridfm_datakit_*` tables are single parquet files. The PFΔ tables are instead
**partitioned parquet directories** (`scenario_partition=*/` subdirectories, 145 per
table), so point `pandas.read_parquet` at the `*.parquet` directory rather than at an
individual part file — it handles both layouts identically.

The four external baselines under `full/` are the converted parquet only (`bus`/`gen`,
plus `branch` and `y_bus` for PFΔ) — not the original downloads. They are third-party
datasets reshaped into this repository's schema; see their upstream licences before
redistributing.

## 3. Generate the figures

```bash
OUT=paper/repro/out

# Entropy spider plots (PF and OPF)
python paper/repro/plot_spider.py --mode pf  --metric entropy --output-dir $OUT
python paper/repro/plot_spider.py --mode opf --metric entropy --output-dir $OUT

# Feature violin plots: Pd, Qd, Pg, Qg, Vm, Va for PF and OPF
python paper/repro/plot_violin.py --mode pf  --output-dir $OUT
python paper/repro/plot_violin.py --mode opf --output-dir $OUT

# Branch flow entropy barplot (Pf, Qf)
python paper/repro/plot_bar_branch.py --metric entropy --output-dir $OUT

# Branch loading histograms
python paper/repro/plot_branch_loading.py --output-dir $OUT
```

That writes 17 PDFs into `paper/repro/out/`:

| Script | Output |
|---|---|
| `plot_spider.py` | `spider_plot_entropy_{pf,opf}.pdf` |
| `plot_violin.py` | `{Pd,Qd,Pg,Qg,Vm,Va}_violin_{pf,opf}.pdf` |
| `plot_bar_branch.py` | `barplot_branch_entropy_pf.pdf` |
| `plot_branch_loading.py` | `branch_loading_{datakit,pfdelta}.pdf` |

Every script takes `--output-dir`, `--data-dir`, and `--datasets` (to restrict the
comparison to a subset). `plot_spider.py` and `plot_bar_branch.py` also take
`--metric std` for the standard-deviation variant; omitting `--metric` produces both.

The first spider or violin run calls `load_net_from_pglib("case118_ieee")`, which
downloads the PGLib case into the installed package's `grids/` directory and may
resolve the pinned Julia packages. That is a one-off, cached cost — none of these
scripts solve OPF or PF.

## What the numbers mean

`plot_branch_loading.py` prints the overload statistics alongside the histograms:

```
Loading gridfm_datakit_pf...
  percentage of overloaded branches per scenario: 1.2158602150537636
  percentage of scenarios with overloaded branches: 79.07
Loading pfdelta...
  percentage of overloaded branches per scenario: 8.027258064516129
  percentage of scenarios with overloaded branches: 100.0
```

Branch loading is `max(S_from, S_to) / rate_a` with `S = sqrt(P² + Q²)`; a branch is
overloaded when it exceeds 1.

**Entropy.** `plot_utils.py` implements the mean normalized Shannon entropy: a 100-bin
histogram per bus over a fixed domain, entropy in bits, averaged across buses and
normalized by `log2(100)`, giving a value in `[0, 1]`. Domains are `(-π, π]` for `Va`
(treated as circular via `entropy_circular_from_deg_fixed`), the network voltage bounds
for `Vm`, and per-bus empirical min/max **pooled across all compared datasets** for
`Pd, Qd, Pg, Qg`. Pooling the domain is what makes the values comparable across
libraries instead of reflecting per-dataset binning.

## Rebuilding the snapshot from raw output

Only needed if you are generating new datasets rather than reproducing the published
figures.

```bash
python paper/repro/prepare_datasets.py --base-path /path/to/raw/datasets [--seed 0]
```

It finds the smallest scenario count across the datasets and downsamples all of them to
it. The gridfm-datakit inputs themselves are generated with the configs in
[`configs/`](configs/) (`gridfm_datakit generate paper/repro/configs/case118_ieee_pf.yaml`);
the full non-downsampled data is on HuggingFace under `full/`. Note that the subset is drawn randomly: a fresh run yields a different sample and
therefore slightly different figures. `--seed` makes a run repeatable but cannot
recover the published snapshot — use the HuggingFace snapshot for that.

The external libraries must be converted to this repository's parquet schema first:

| Library | Converter |
|---|---|
| PFΔ | `pfdelta/batch_convert_pfdelta.py` |
| OPFData | `opf_data/batch_convert.py` |
| PGLearn | `paper/repro/pg_learn_conversion.py` (set `PGLEARN_DIR`) |
| OPF-Learn | `paper/repro/opf_learn_conversion.py` (set `OPFLEARN_DIR`) |

## Notes

- **Bus selection in the violin plots is pinned.** Each violin panel shows 10 buses,
  fixed in `PINNED_BUSES` in `plot_violin.py` so the figures are stable across runs.
  Pass `--no-pin` to sample buses randomly instead.
- **`plot_branch_loading.py` forces the `Agg` matplotlib backend** before importing
  pyplot. On macOS the default `macosx` backend applies Retina scaling to saved vector
  output, which changes the rendering. Do not reorder those imports.
- **`plot_spider_branch.py` and `plot_violin_branch.py`** produce branch-level spider
  and violin figures that are not part of the paper. They are kept because they were
  part of the original analysis. Careful: `plot_spider_branch.py` writes the same
  filename as `plot_bar_branch.py` (`barplot_branch_{metric}_pf.pdf`), so run them into
  separate `--output-dir`s.

For how these outputs were checked against the published figures, see
[`verify_reproducibility.md`](verify_reproducibility.md).
