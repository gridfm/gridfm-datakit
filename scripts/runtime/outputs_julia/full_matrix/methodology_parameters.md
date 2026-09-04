# Benchmark methodology parameters

### Shared

| Parameter | Choice | Why |
|-----------|--------|-----|
| Solves per network | 4M / 3M / 2M / 2M / 500k / 50k / 10k (case14→case10000) | Enough wall time for stable estimates on fast configs; same budget for slow modes and for both methods |
| Distinct scenarios | ≤10 000 (wrap if more solves requested) | Balance diversity vs memory / staging cost |
| Scenario order | No shuffle | Avoid extra I/O variability |
| Setup‑2 storage | Stage to node-local `/tmp` before the run | Local disk is much faster than the shared filesystem |
| Data root | `…/finetuning`; GENCO uses PF graphs; PowerModels uses PF or OPF split by mode | Match each solver’s intended scenario set |

### Parallelism & compile

| Parameter | GENCO | PowerModels | Why |
|-----------|-------|-------------|-----|
| Sweep | Batch size 64–16384 (IEEE) / 16–16384 (GOC), powers of two | Workers `p` = 24…216 (step 16) | Wide enough to see the plateau; coarse enough to stay tractable |
| Prefetch / dispatch | 32 DataLoader workers | Dispatch batch 32 (small, incl. case500) / 1 (case2000/10000) | Keep workers busy without excess queueing or stragglers |
| Compilation / solver | `torch.compile` (`reduce-overhead`) | Fast AC PF on case14–500; Ipopt on case2000/10000 | Use the best reliable stack per network size |

### PowerModels / Ipopt

Applies to Ipopt-backed modes: AC PF when `pf_fast=false` (case2000/10000), AC OPF, and DC OPF. Fast AC PF (`compute_ac_pf`) and DC PF (`compute_dc_pf`) do not use Ipopt.

| Parameter | Choice | Why |
|-----------|--------|-----|
| Ipopt `max_iter` | 100 | Caps worst-case wall time per solve; enough for typical `LOCALLY_SOLVED` on these networks |
| Ipopt `tol` | 1e-6 | Standard NLP convergence tolerance |
| Ipopt `print_level` | 0 | No solver logging on the timed path |
| Linear solver | MUMPS 5.8.2 (Ipopt default; not overridden) | Shipped with `Ipopt_jll`; HSL ma57 is often faster but needs a separate HSL license and is more expensive to deploy |
| AC PF (small) | `PowerModels.compute_ac_pf` (NLsolve Newton) | Fast native PF; used for case14–case500 |
| AC PF (large) | `solve_ac_pf` + Ipopt | More robust on case2000 / case10000 |
| DC PF | `compute_dc_pf` | Linear DC model; no Ipopt |
| AC / DC OPF | `solve_ac_opf` / `solve_dc_opf` + Ipopt | Same Ipopt settings for all OPF solves |

### Setup 1 (data ready)

| Parameter | GENCO | PowerModels | Why |
|-----------|-------|-------------|-----|
| Prepared data | ≤10 000 graphs in RAM | One base network per worker | GENCO: avoid a single repeated sample; PM: keep memory low with many processes |

### Setup 2 (load during the run)

| Parameter | Choice | Why |
|-----------|--------|-----|
| Staged pool | ≤10 000 files on `/tmp` | Too small → little diversity and heavy cache reuse; too large → costly staging |
