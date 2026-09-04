# Verifying reproducibility

Record of how the output of `paper/repro/` was checked against the figures published in
*gridfm-datakit-v1* ([arXiv:2512.14658](https://arxiv.org/abs/2512.14658)).

**Result: all 17 panels are pixel-identical to the published figures.**

This document is provenance, not part of the reproduction workflow — to generate the
figures, see [`README.md`](README.md).

## Panels checked

| Paper figure | Panels |
|---|---|
| Figure 2 | `spider_plot_entropy_pf`, `spider_plot_entropy_opf` |
| Figure 3 | `Qg_violin_pf` |
| Figure 4 | `barplot_branch_entropy_pf` |
| Figure 5 | `branch_loading_datakit`, `branch_loading_pfdelta` |
| Figure 8 (appendix) | `{Pd,Qd,Pg,Qg,Vm,Va}_violin_{pf,opf}` — 12 panels |

## Method

The reference PDFs are the ones submitted with the paper. They are not kept in this
branch; obtain them from the paper's source archive (the `figures/comparison_plots/`
directory of the arXiv submission) and point `REF` at them.

**Compare rendered pixels, not PDF bytes.** PDFs embed a creation timestamp, so `md5` of
two PDFs never matches even when the figures are identical. Rasterize both sides first:

```bash
REF=/path/to/paper/figures/comparison_plots
OUT=paper/repro/out

for name in spider_plot_entropy_pf spider_plot_entropy_opf \
            Qg_violin_pf barplot_branch_entropy_pf \
            branch_loading_datakit branch_loading_pfdelta \
            Pd_violin_pf Qd_violin_pf Pg_violin_pf Vm_violin_pf Va_violin_pf \
            Pd_violin_opf Qd_violin_opf Pg_violin_opf Qg_violin_opf \
            Vm_violin_opf Va_violin_opf; do
    pdftoppm -r 100 -png "$REF/$name.pdf" "/tmp/ref_$name"
    pdftoppm -r 100 -png "$OUT/$name.pdf" "/tmp/mine_$name"
    if [ "$(md5 -q /tmp/ref_$name-1.png)" = "$(md5 -q /tmp/mine_$name-1.png)" ]; then
        printf '%-12s %s\n' IDENTICAL "$name"
    else
        printf '%-12s %s\n' DIFFERS "$name"
    fi
done
```

`pdftoppm` comes from poppler (`brew install poppler`). 100 dpi is enough to catch any
difference in data, geometry, or styling.

## Statistics

`plot_branch_loading.py` reproduces the values quoted in Section 5 of the paper:

| Quantity | Paper | Reproduced |
|---|---|---|
| datakit: branches overloaded | 1.2 % | 1.2159 % |
| datakit: scenarios with ≥1 overload | 79 % | 79.07 % |
| PFΔ: branches overloaded | 8 % | 8.0273 % |
| PFΔ: scenarios with ≥1 overload | all | 100.0 % |

## Issues found and fixed while porting

The scripts came from the authors' working tree and were not reproducible as-is. Three
defects had to be fixed:

1. **Unseeded bus selection in the violin plots.** `plot_violin.py` chose its 10
   displayed buses with `np.random.choice` and no seed, so figures 3 and 8 differed on
   every run — one run produced buses `45, 102, 68, 64, 71, 7, 14, 33, 69, 84` against
   the published `35, 23, 106, 75, 72, 109, 11, 90, 69, 84`. The published selections
   were recovered from the figure PDFs with `pdftotext` and are pinned in `PINNED_BUSES`
   in `plot_violin.py`; list order is the x-axis order. `--no-pin` restores the random
   behaviour.

2. **Matplotlib backend affected the rendering.** The branch-loading figures matched
   when generated in an ad-hoc script but not once packaged. Cause: the default `macosx`
   backend applies Retina scaling to saved vector output. `plot_branch_loading.py` now
   calls `matplotlib.use("Agg")` before importing pyplot. The other 15 panels were
   unaffected, as only these use seaborn.

3. **Figure 5 had no script.** It existed only as notebook cells with hardcoded absolute
   paths and duplicated logic; extracted to `plot_branch_loading.py`, preserving the
   loading formula and plot styling exactly.

Also de-hardcoded: absolute paths in `prepare_datasets.py` (now `--base-path` /
`--output-dir` / `--seed`) and in the PGLearn and OPF-Learn conversion scripts (now the
`PGLEARN_DIR` and `OPFLEARN_DIR` environment variables); `--data-dir` defaults were
unified across the plot scripts.

## Caveats on exactness

- **Exact reproduction depends on the dataset snapshot**, not just the code.
  `prepare_datasets.py` draws its scenario subset with an unseeded `np.random.choice`,
  and the original run recorded no seed, so re-running it yields a different
  10,000-scenario subset and different (though qualitatively similar) figures. The
  snapshot on HuggingFace is the authoritative input.
- **Provenance:** branched from `main` at `35da4b3` (Release 1.1.0). Do not use
  `gridfm_datakit.__version__` to record provenance — it reports `0.1.0` while
  `pyproject.toml` says `1.1.0`. Cite the git SHA.

## Not reproduced here

- **Figure 1** and the appendix library-comparison table are hand-authored HTML, not
  computed output.
- **Figures 6 and 7** are screenshots of the `gridfm_datakit stats` output and of a
  generated dataset.
- **Table 1** (samples / CPU-hours / convergence rate) took roughly 6,000 CPU-hours
  across four grids. `scripts/summary_data_gen.py` recomputes it from the logs
  (`args.log`, `tqdm.log`, `n_scenarios.txt`, `solver_log/`) of completed generation
  runs; it does not re-run the generation.
