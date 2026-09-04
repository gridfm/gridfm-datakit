# Zeroing the slack-bus residual in DC power flow statistics

## What this is

`compute_stats_from_data` scores DC power flow solutions in the AC model: it evaluates the
per-bus active power mismatch `P_mis_dc` using the full complex Y matrix at the DC
operating point (`Vm = 1.0`, `Va = Va_dc`). A large part of the residual that shows up at
the reference (slack) bus is a bookkeeping artifact, not a real modelling error, and it can
be removed exactly.

## Why the slack residual is an artifact

A DC power flow does not enforce the slack bus balance — the slack is the free variable
that absorbs whatever the rest of the system does not supply. Its generation therefore has
to be reconstructed after the solve, and there are two inconsistent ways to do it:

- **What datakit does today** (`apply_slack_single_gen` in
  `gridfm_datakit/process/process_network.py`): `Pg_slack` is closed against PowerModels'
  *lossless* DC flows, `p = -b_series * dtheta`, for which `pf + pt = 0` on every branch.
- **What the residual is scored with**: the full complex Y, which retains series
  resistance, so `pf + pt != 0`.

The whole model difference — the losses the lossless DC flows dropped — lands on the one
bus whose generation was a free variable. It is pure convention mismatch.

## Why zeroing that one entry is exact

`compute_bus_balance` (`gridfm_datakit/utils/power_balance.py`) computes, per bus:

```
P_mis[i] = | (Pg[i] - Pd[i]) - P_out[i] - P_sh[i] |
```

`Pg[i]` appears in exactly one row — bus `i`'s own equation. So changing `Pg` at the slack
bus only can only change `P_mis` at the slack bus, and nothing else. Angles, loads, shunts,
non-slack setpoints and every branch flow are untouched (`compute_branch_powers_vectorized`
reads only `Vm`, `Va` and the Y entries — never `Pg`).

Therefore these two are algebraically identical:

1. Recompute `Pg_slack = Pd + P_shunt + sum(full-Y flows out of slack)`, making the slack
   residual `0` by construction.
2. Take the existing lossless-slack residuals, **set the slack bus entry to 0**, and
   aggregate over all N buses (slack included).

Route 2 needs no re-solve, which is what is implemented.

> **Active power only.** `Q_mis_ac` uses `Qg`, which comes from the stored AC solution and
> is not a reconstructed DC free variable. There is nothing for it to absorb, so the same
> zeroing is **not** justified for reactive power and is never applied to it.

## Usage

Opt-in flag, default `False`, so existing behaviour is byte-identical:

```python
from gridfm_datakit.utils.stats import compute_stats_from_data, plot_stats

# Statistics only — writes nothing to disk.
stats = compute_stats_from_data(
    data_dir,
    sn_mva=100.0,
    n_partitions=10,
    zero_slack_dc_residual=True,
)

# Plots + stats.parquet. When the flag is on, outputs are suffixed
# (stats_plot_slack_corrected.png, stats_slack_corrected.parquet) so a corrected
# run never overwrites the uncorrected artifacts already in the directory.
plot_stats(data_dir, sn_mva=100.0, n_partitions=10, zero_slack_dc_residual=True)
```

The correction is applied to `balance_dc` before the per-scenario groupby, so both the
mean and the max aggregations pick it up. It joins on `(scenario, bus)` rather than
relying on row order, because `compute_bus_balance` returns a frame with a fresh index
from an internal merge.

Verified properties: AC active and reactive metrics are unchanged (exact `allclose`), and
the correction is monotone — it never increases either metric, since it zeroes a
non-negative term.

## Measured effect

`/dccstor/gridfm/powermodels_data/v4/finetuning/pf/<case>/raw`, 10 partitions,
2000 scenarios per case, `sn_mva = 100`. The same partitions are sampled for both
variants (reseeded per case), so the columns are directly comparable. Values in MW.

| Case | Mean residual (raw) | Mean (corrected) | Max residual (raw) | Max (corrected) |
|---|---|---|---|---|
| case118_ieee | 2.2624 | **2.0768** | 27.834 | **23.307** |
| case2000_goc | 1.0347 | **1.0346** | 105.251 | **105.251** |
| case10000_goc | 0.6355 | **0.6345** | 127.104 | **127.104** |

"Mean" = per-scenario mean over all N buses, then averaged over scenarios.
"Max" = per-scenario max over buses, then averaged over scenarios.
Worst-case-over-all-scenarios max: 67.41 -> 48.89 (case118), unchanged for the other two.

## Interpreting this table — read before quoting it

The correction is large for case118 and negligible for the two large cases. This is
structural, not a defect:

| Case | Slack residual removed | Fraction of scenarios where slack was the worst bus |
|---|---|---|
| case118_ieee | 24.7 MW | **60.8%** |
| case2000_goc | 0.14 MW | 0.0% |
| case10000_goc | 12.2 MW | 0.0% |

Two independent dilutions stack on the large cases:

- **The mean divides by N.** Even case10000's 12.2 MW slack residual becomes
  `12.2 / 10000 ~= 0.001` MW of mean — invisible, despite being a real 12 MW error at
  that bus.
- **The max is set by other buses entirely.** On case118 the slack is the single worst bus
  61% of the time, so zeroing it visibly moves the max. On case2000 and case10000 it is
  *never* the worst bus — some other bus carries a 105-127 MW residual — so the max is
  bit-identical by construction, not by coincidence.

Consequences for reporting:

- The "DC's apparent error is partly a slack bookkeeping artifact" argument is strong on
  case118 and essentially irrelevant to the large cases **under these two metrics**.
- To show the artifact on the large grids, report the slack-bus residual itself (the
  middle column above), not a mean over 10 000 buses that divides it away.
- The 105-127 MW max residuals on the large cases are a genuinely different phenomenon —
  real DC linearization error at non-slack buses — which this correction does not and
  should not address.
