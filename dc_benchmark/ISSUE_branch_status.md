# `compute_branch_powers_vectorized` doesn't check line status

## Summary

`compute_branch_powers_vectorized` (`gridfm_datakit/utils/power_balance.py:74`)
computes branch flows purely from the `Yff/Yft/Ytf/Ytt` columns of `branch_df`. It
never looks at `br_status`.

This is correct today only because of an unstated invariant: `branch_vectors`
(`gridfm_datakit/network.py:766`) multiplies the series admittance and charging
susceptance by `stat = branch[:, BR_STATUS]`, so an out-of-service branch has all four
admittances exactly zero, and its computed flow is therefore zero as well.

That invariant is real but implicit, and it is only preserved as long as every caller
feeds admittances that came through `branch_vectors`. Nothing enforces it.

## Why it matters

Any caller that builds `branch_df` from admittances not produced by `branch_vectors`
gets silently wrong flows on out-of-service branches — no error, no warning, just a
phantom flow on a line that is supposed to be disconnected.

I hit exactly this while scoring DC solutions from `.pt` files. Reconstructing the
admittances from the stored tensors and disabling two branches on `case118_ieee` gave
those branches **7.47 MW of flow**, which then propagated into the per-bus balance and
into the slack generation derived from it. Measured, not hypothetical.

The failure is silent, and it corrupts a physical quantity rather than crashing, which
makes it the worst kind to leave to convention.

This also matters more now than it used to: with configurable outage sampling (#63),
datasets containing out-of-service branches are expected rather than exceptional.

## Reproducer

```python
from gridfm_datakit.network import load_net_from_pglib, branch_vectors
from gridfm_datakit.utils import idx_brch as ib

net = load_net_from_pglib("case118_ieee")
net.branches[[3, 17], ib.BR_STATUS] = 0
Ytt, Yff, Yft, Ytf = branch_vectors(net.branches, net.branches.shape[0])
# Yff[3] == Yft[3] == Ytf[3] == Ytt[3] == 0  -> the invariant holds here
```

The invariant holds through the normal pipeline: `process_network.py:532` calls
`branch_vectors` and writes those zeroed admittances directly into the stored branch
columns. So `stats.py` is fine today. The risk is entirely about callers that bypass
that path.

## Proposed fix

Two options, not mutually exclusive.

**1. Assert the invariant (cheap, catches misuse immediately).**
If `branch_df` carries a `br_status` column, assert that out-of-service rows have zero
admittance:

```python
if "br_status" in branch_df.columns:
    off = branch_df["br_status"].to_numpy() != 1
    if off.any():
        assert not np.any(
            branch_df.loc[off, ["Yff_r", "Yff_i", "Yft_r", "Yft_i",
                                "Ytf_r", "Ytf_i", "Ytt_r", "Ytt_i"]].to_numpy()
        ), "out-of-service branches must have zero admittance"
```

This turns a silent wrong number into a loud failure, and documents the contract.

**2. Enforce it (robust, works regardless of caller).**
Zero the flows for out-of-service branches inside the function, so the result is
correct by construction rather than by convention:

```python
if "br_status" in branch_df.columns:
    off = branch_df["br_status"].to_numpy() != 1
    pf[off] = qf[off] = pt[off] = qt[off] = 0.0
```

My preference is **(2) with the docstring stating the contract**, because it makes the
function correct on its own terms instead of depending on how the caller built its
input. (1) is a good minimum if you'd rather keep the function purely computational.

Either way the docstring should say explicitly that admittances are expected to be
status-masked, since that assumption is currently invisible at the call site.

## Note on scope

`compute_bus_balance` inherits this: it consumes whatever `flows` it is handed, so it is
correct exactly when the flows are. No separate fix needed there if the above lands.
