# Generation configs

The exact configurations used to generate the two gridfm-datakit datasets compared in
*gridfm-datakit-v1* ([arXiv:2512.14658](https://arxiv.org/abs/2512.14658)), on
`case118_ieee`.

| File | Mode | Samples generated | Converged |
|---|---|---|---|
| `case118_ieee_pf.yaml` | `pf` | 200,000 | 199,207 |
| `case118_ieee_opf.yaml` | `opf` | 200,000 | 195,753 |

Each run is 10,000 load scenarios × 20 topology variants. The published figures use
10,000 scenarios sampled from each.

To re-run a generation:

```bash
gridfm_datakit generate scripts/datakit_report/configs/case118_ieee_pf.yaml
```

## Provenance

These YAML files were reconstructed from the `args.log` that each original run wrote,
and every field has been checked to match it. The original logs are kept alongside as
`*.args.log` so the record is not just a transcription:

- `case118_ieee_pf.args.log` — run started 2025-11-12 11:54:10
- `case118_ieee_opf.args.log` — run started 2025-11-12 12:10:23

## The settings that matter

The two configs are identical except for `settings.mode` and `settings.data_dir`. The
perturbation settings follow the paper's Section 5 footnote, chosen for a fair comparison
against the other libraries:

| Setting | Value | Why |
|---|---|---|
| `topology_perturbation.k` | 1 | N-1, matching every other library except OPF-Learn |
| `load.sigma` | 0.2 | Local load noise, as in PGLearn and OPFData |
| `load.global_range` | 0.4 | Global scaling range, as in PGLearn |
| `generation_perturbation.type` | `cost_permutation` | Generator costs permuted, as in PFΔ |
| `admittance_perturbation.sigma` | 0.2 | A feature absent in the other libraries |

`mode` is the substantive difference between the two datasets:

- **`pf`** solves one ACOPF per load scenario on the base topology, fixes the generator
  setpoints, then applies the topology perturbation and solves an AC power flow. Because
  dispatch is not re-optimised for the new topology, some samples violate OPF inequality
  constraints — which is the point of the PF dataset.
- **`opf`** solves an ACOPF for every topology variant, so every sample is cost-optimal
  and within all operating limits. This is roughly 20× more OPF solves than `pf` mode.

## Reproducibility caveat

Neither original run set `settings.seed` (the configs record `seed: null`), so re-running
these configs produces statistically equivalent but not identical data. Setting an
integer seed makes a run repeatable, but only if every other setting — including
`num_processes` and `large_chunk_size`, which determine the chunk boundaries that seed the
per-chunk RNG — is unchanged. It cannot recover the original datasets.

The generated data itself is on HuggingFace at
[`gridfm/reproducibility-datakit-technical-report`](https://huggingface.co/datasets/gridfm/reproducibility-datakit-technical-report)
under `full/gridfm_datakit_pf/` and `full/gridfm_datakit_opf/`.
