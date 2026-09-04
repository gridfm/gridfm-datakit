#!/bin/bash
# LSF batch job: DC residuals on 1000 randomly sampled case500_goc scenarios.
set -euo pipefail

OUT=/u/apu/gridfm-datakit/dc_res_lsf/out_n1000
mkdir -p "$OUT"

echo "host: $(hostname)  started: $(date)"

PYTHONPATH=/u/apu/gridfm-datakit \
/u/apu/gridfm_model_evaluation/venv/bin/python /u/apu/gridfm-datakit/dc_res_lsf/compute_dc_residuals.py \
  --processed-dir /dccstor/gridfm/powermodels_data/v4/finetuning/pf/case500_goc/processed \
  --output-dir "$OUT" \
  --sample 1000 --seed 0

echo "finished: $(date)"
