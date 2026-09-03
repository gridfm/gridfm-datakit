# Reproducing the paper's figures

**This is the branch used to generate the results published in
*gridfm-datakit-v1: A Python Library for Scalable and Realistic Power Flow and
Optimal Power Flow Data Generation* ([arXiv:2512.14658](https://arxiv.org/abs/2512.14658)).**

Everything needed to regenerate the paper's computed figures lives in this
directory. All 17 published panels reproduce **pixel-for-pixel**.

```bash
# from the repository root
PYTHONPATH=$PWD bash paper/repro/verify.sh
```

This regenerates every figure and compares it against the published PDFs,
printing one line per panel and exiting non-zero if any differs. Expected output:

```
All 17 panels are pixel-identical to the published figures.
```

## Where to find each figure

Reference PDFs (as published) are in `paper/extracted/figures/comparison_plots/`.
Regenerated figures go to `paper/repro/out/` by default.

| Paper figure | What it shows | Command | Output file(s) |
|---|---|---|---|
| **Figure 2** | Spider plot, normalized mean feature entropy (PF and OPF) | `python paper/repro/plot_spider.py --mode pf --metric entropy`<br>`python paper/repro/plot_spider.py --mode opf --metric entropy` | `spider_plot_entropy_pf.pdf`<br>`spider_plot_entropy_opf.pdf` |
| **Figure 3** | Reactive power generation ($Q_g$) violins, PF | `python paper/repro/plot_violin.py --mode pf` | `Qg_violin_pf.pdf` |
| **Figure 4** | Branch flow entropy barplot ($P_f$, $Q_f$) | `python paper/repro/plot_bar_branch.py --metric entropy` | `barplot_branch_entropy_pf.pdf` |
| **Figure 5** | Branch loading histograms (log-scale y) | `python paper/repro/plot_branch_loading.py` | `branch_loading_datakit.pdf`<br>`branch_loading_pfdelta.pdf` |
| **Figure 8** (appendix) | Feature distribution violins, 6 features × PF/OPF | `python paper/repro/plot_violin.py --mode pf`<br>`python paper/repro/plot_violin.py --mode opf` | `{Pd,Qd,Pg,Qg,Vm,Va}_violin_{pf,opf}.pdf` |

Every script takes `--output-dir` and `--data-dir`. Add `--output-dir paper/repro/out`
to match what `verify.sh` does.

`plot_branch_loading.py` also prints the overload statistics quoted in Section 5:

```
Loading gridfm_datakit_pf...
  percentage of overloaded branches per scenario: 1.2158602150537636
  percentage of scenarios with overloaded branches: 79.07
Loading pfdelta...
  percentage of overloaded branches per scenario: 8.027258064516129
  percentage of scenarios with overloaded branches: 100.0
```

matching the paper's "1.2 % of the branches are overloaded ... 79 % of the scenarios
have at least one branch overloading" and "all scenarios have overloads ... 8 % of all
branches are overloaded".

## Setup

```bash
# Python 3.10-3.12; run everything from the repository root
pip install -e '.[dev,test]'
pip install seaborn          # needed by plot_branch_loading.py only; not a package dependency
brew install poppler         # provides pdftoppm, used by verify.sh
```

Run from the repository root with `PYTHONPATH=$PWD`. The scripts import
`gridfm_datakit` (for `load_net_from_pglib` and the bus/gen index constants), and
depending on how the package was installed the import may only resolve from the
repository root.

The first spider or violin run calls `load_net_from_pglib("case118_ieee")`, which
downloads the PGLib case into the installed package's `grids/` directory and may
resolve the pinned Julia packages. That is a one-off cost and is cached; none of the
figure scripts solve OPF or PF.

## Input data

The figures are computed from a **snapshot of sampled parquet** in
`paper/repro/dataset_sampled/` — six datasets on `case118_ieee`, each downsampled to
exactly 10,000 scenarios × 118 buses:

| Dataset | Mode | Source |
|---|---|---|
| `gridfm_datakit_pf` | PF | this library, `mode: pf` |
| `gridfm_datakit_opf` | OPF | this library, `mode: opf` |
| `pfdelta` | PF | PFΔ, converted by `pfdelta/batch_convert_pfdelta.py` |
| `opfdata` | OPF | OPFData, converted by `opf_data/batch_convert.py` |
| `pglearn` | OPF | PGLearn, converted by `pg_learn_conversion.py` |
| `opflearn` | OPF | OPF-Learn, converted by `opf_learn_conversion.py` |

This directory is ~979 MB and is **gitignored**, so it does not travel with the branch.
It is the authoritative input: obtain it from the authors, or rebuild it with
`prepare_datasets.py` subject to the caveat below.

`prepare_datasets.py` rebuilds the snapshot from raw generated output:

```bash
python paper/repro/prepare_datasets.py --base-path /path/to/raw/datasets [--seed 0]
```

## Reproducibility notes

Read these before concluding that something failed to reproduce.

1. **Compare pixels, not PDF bytes.** PDFs embed a creation timestamp, so `md5` of two
   PDFs never matches even when the figures are identical. `verify.sh` rasterises both
   sides with `pdftoppm -r 100` and compares the PNGs.

2. **Violin bus selections are pinned.** Upstream, `plot_violin.py` chose which 10 buses
   to show with an *unseeded* `np.random.choice`, so figures 3 and 8 differed on every
   run. The exact selections were recovered from the published PDFs and are pinned in
   `PINNED_BUSES` in `plot_violin.py`; list order is the x-axis order. Pass `--no-pin`
   to re-sample buses randomly instead.

3. **Re-sampling the dataset snapshot will not reproduce the figures.**
   `prepare_datasets.py` draws its scenario subset with `np.random.choice` and the
   original run recorded no seed, so a fresh run yields a different 10,000-scenario
   subset and therefore different (though qualitatively similar) figures. `--seed` only
   makes future runs self-consistent. Use the committed snapshot for exact reproduction.

4. **`plot_branch_loading.py` forces the `Agg` matplotlib backend.** On macOS the
   default `macosx` backend applies Retina scaling to saved vector output, which changes
   the rendered figure. This is set before `pyplot` is imported; do not reorder those
   imports.

5. **Provenance.** Branched from `main` at `35da4b3` (Release 1.1.0). Do not use
   `gridfm_datakit.__version__` to record provenance — it reports `0.1.0` while
   `pyproject.toml` says `1.1.0`; cite the git SHA instead.

## Entropy metric

Implemented in `plot_utils.py` (`entropy_from_samples_fixed`,
`entropy_circular_from_deg_fixed`), following the paper's Appendix C: a 100-bin
histogram per bus over a fixed domain, Shannon entropy in bits, averaged across buses
and normalized by $\log_2(100)$. Domains are $(-\pi, \pi]$ for $V_a$ (treated as
circular), network voltage bounds for $V_m$, and per-bus empirical min/max pooled
across all compared datasets for $P_d, Q_d, P_g, Q_g$ — so values are comparable across
datasets rather than driven by binning differences.

## Out of scope

- **Figure 1** and the appendix library-comparison table are hand-authored HTML, not
  computed. Sources: `paper/extracted/summary.txt` and `paper/extracted/output.txt`.
- **Figures 6 and 7** are screenshots of the `gridfm_datakit stats` output and of a
  generated dataset, not computed figures.
- **Table 1** (samples / CPU-hours / convergence rate) required roughly 6,000 CPU-hours
  across four grids. `scripts/summary_data_gen.py` recomputes it from the run logs
  (`args.log`, `tqdm.log`, `n_scenarios.txt`, `solver_log/`) of completed generation runs;
  it does not re-run the generation.

## Also here, but not used in the paper

`plot_spider_branch.py` and `plot_violin_branch.py` produce branch-level spider and
violin figures that no published figure uses. They are kept because they were part of
the original analysis.

Note that `plot_spider_branch.py` writes the **same filename** as
`plot_bar_branch.py` (`barplot_branch_{metric}_pf.pdf`). The paper's Figure 4 is the one
from `plot_bar_branch.py`; run them into separate `--output-dir`s to avoid clobbering.
