# Wall time at best p (full matrix)

Best **p** = `argmin(pf_elapsed_s / n_pfs)` (wall µs/solve).
**Never** use `mean_pf_runtime_s` for best-p selection.

Wall = `pf_elapsed_s` (timed solve region only; init/staging excluded).

Modes: AC-OPF / DC-OPF / AC-PF / DC-PF.

## Setup1 (cached-base)

| grid | AC-OPF | DC-OPF | AC-PF | DC-PF | n | best p |
|------|-------:|-------:|------:|------:|--:|--------|
| case14_ieee | 11.4m | 2.9m | 28.1s | 19.6s | 4,000,000 | 216/216/120/40 |
| case30_ieee | 16.7m | 2.9m | 33.9s | 18.4s | 3,000,000 | 216/216/136/88 |
| case57_ieee | 21.0m | 3.4m | 33.3s | 13.9s | 2,000,000 | 216/216/200/104 |
| case118_ieee | 1.43h | 8.4m | 1.8m | 36.3s | 2,000,000 | 216/216/120/184 |
| case500_goc | 1.64h | 11.6m | 1.8m | 21.2s | 500,000 | 216/200/216/216 |
| case2000_goc | 55.2m | 4.0m | 11.1m | 11.0s | 50,000 | 152/152/152/152 |
| case10000_goc | 1.33h | 5.7m | 10.1m | 11.7s | 10,000 | 184/184/152/152 |

## Setup2 (per-solve-load)

| grid | AC-OPF | DC-OPF | AC-PF | DC-PF | n | best p |
|------|-------:|-------:|------:|------:|--:|--------|
| case14_ieee | 12.2m | 3.1m | 34.8s | 23.5s | 4,000,000 | 152/152/120/72 |
| case30_ieee | 15.9m | 3.1m | 43.6s | 23.5s | 3,000,000 | 152/152/152/104 |
| case57_ieee | 20.9m | 3.8m | 44.5s | 22.5s | 2,000,000 | 152/168/168/136 |
| case118_ieee | 1.15h | 8.8m | 1.9m | 45.5s | 2,000,000 | 184/152/168/152 |
| case500_goc | 1.58h | 9.2m | 2.3m | 48.4s | 500,000 | 200/200/184/152 |
| case2000_goc | 58.6m | 4.4m | 11.4m | 24.2s | 50,000 | 216/184/168/152 |
| case10000_goc | 1.49h | 6.0m | 10.7m | 25.3s | 10,000 | 200/216/200/152 |

