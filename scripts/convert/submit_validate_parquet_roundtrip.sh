#!/usr/bin/env bash
# Full parquet→JSON→solve roundtrip validation (10 random scenarios × all pf/opf cases).
set -euo pipefail

REPO_ROOT="/u/apu/gridfm-datakit"
SCRIPT="$REPO_ROOT/scripts/convert/validate_parquet_roundtrip.py"
LOG_DIR="${HOME}/.lsbatch"
JOB_NAME="${JOB_NAME:-validate_parquet_roundtrip}"

HOST_SELECT="select[hname!='cccxc701' && hname!='cccxc702' && hname!='cccxc703' && hname!='cccxc704' && hname!='cccxc705' && hname!='cccxc706' && hname!='cccxc707' && hname!='cccxc708' && hname!='cccxc709' && hname!='cccxc710' && hname!='cccxc711' && hname!='cccxc712' && hname!='cccxc713' && hname!='cccxc714' && hname!='cccxc715' && hname!='cccxc716']"
SPAN='span[hosts=1]'

mkdir -p "$LOG_DIR"

# ~160 solves; large GOC OPF dominates. Pad wall clock.
bsub \
  -q normal \
  -n 1 \
  -M 64G \
  -W 24:00 \
  -R "$SPAN $HOST_SELECT" \
  -J "$JOB_NAME" \
  -o "$LOG_DIR/${JOB_NAME}_%J.out" \
  -e "$LOG_DIR/${JOB_NAME}_%J.err" \
  bash -lc "
    set -euo pipefail
    export PATH=\"/u/apu/.juliaup/bin:\$PATH\"
    export JULIA_PROJECT=\"$REPO_ROOT/venv/julia_env\"
    export PYTHON_JULIACALL_EXE=\"\$(command -v julia)\"
    export PYTHON_JULIACALL_PROJECT=\"\$JULIA_PROJECT\"
    export JULIA_DEPOT_PATH=\"\${TMPDIR:-/tmp}/gridfm_julia_\${LSB_JOBID}:\$HOME/.julia_shared:\$HOME/.julia\"
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    cd \"$REPO_ROOT\"
    \"$REPO_ROOT/venv/bin/python\" \"$SCRIPT\" \
      --data-root /dccstor/gridfm/powermodels_data/v4/finetuning \
      --n-scenarios 10 \
      --seed 0 \
      --modes pf opf
  "
