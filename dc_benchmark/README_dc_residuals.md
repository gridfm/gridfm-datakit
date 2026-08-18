# DC residuals on `.pt` scenarios

Script: `compute_dc_residuals.py`

## What it does

For every `data_index_<i>.pt` in a processed dataset, it solves DC power flow and
measures how far that DC solution is from satisfying the real (AC) power balance.

You get one number per scenario: the average active power mismatch across all buses,
in MW. 

Everything comes from the `.pt` files. No `.m` or parquet files are needed.

## How to run it

```bash
python dc_benchmark/compute_dc_residuals.py \
    --processed-dir dc_benchmark/data/case118_ieee/processed
```

Try it on a few scenarios first:

```bash
python dc_benchmark/compute_dc_residuals.py --limit 20
```

Options:

| flag | meaning |
|---|---|
| `--processed-dir` | where the `.pt` files are (defaults to case118_ieee) |
| `--output-dir` | where to write results (defaults to `--processed-dir`) |
| `--limit N` | only do the first N scenarios (0 = all) |

Takes a few minutes for ~1000 scenarios.

## What you get

**`dc_residuals.csv`** — one row per scenario:

```
pt_index,mean_p_residual_lossless,mean_p_residual_full_y
0,0.21182616022344847,0.15840249937142767
1,0.35446750482216566,0.26382235778976953
```

**`dc_residuals_stats.txt`** — mean/median/max over all scenarios.

## Why two columns

DC power flow doesn't decide how much the slack bus generates, so we have to fill that
in afterwards. There are two ways, and they give slightly different answers:

- **`mean_p_residual_lossless`** — what datakit does today. Leaves a leftover error at
  the slack bus that isn't really the DC model's fault; it's a bookkeeping artifact.
- **`mean_p_residual_full_y`** — fills in the slack so its error is exactly zero.

**Use `full_y`** if you're comparing a neural solver against DC and want a fair number.
On case118 it comes out about 19% lower.

Everything else about the two columns is identical, so the gap between them is purely
the slack convention.

