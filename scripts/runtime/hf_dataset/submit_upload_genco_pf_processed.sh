#!/usr/bin/env bash
# Upload the 10k processed PF graphs to Hugging Face (resumable).
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/runtime/hf_dataset/upload_genco_pf_processed.py"
LOG_DIR="${LSB_LOG_DIR:-$HOME/.lsbatch}"
JOB_NAME="${JOB_NAME:-hf_upload_genco_pf_processed}"

# Same host class as other I/O jobs (not the 7xx runtime nodes).
HOST_SELECT="select[hname!='cccxc701' && hname!='cccxc702' && hname!='cccxc703' && hname!='cccxc704' && hname!='cccxc705' && hname!='cccxc706' && hname!='cccxc707' && hname!='cccxc708' && hname!='cccxc709' && hname!='cccxc710' && hname!='cccxc711' && hname!='cccxc712' && hname!='cccxc713' && hname!='cccxc714' && hname!='cccxc715' && hname!='cccxc716']"
SPAN='span[hosts=1]'

mkdir -p "$LOG_DIR"

# ~34 GiB, 70k files in 200-file commits. Resume-safe if the job is killed.
bsub \
  -q normal \
  -n 1 \
  -M 32G \
  -W 72:00 \
  -R "$SPAN $HOST_SELECT" \
  -J "$JOB_NAME" \
  -o "$LOG_DIR/${JOB_NAME}_%J.out" \
  -e "$LOG_DIR/${JOB_NAME}_%J.err" \
  bash -lc "
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export PATH=\"\$HOME/.local/bin:\$PATH\"
    cd \"$REPO_ROOT\"
    echo \"host=\$(hostname) job=\${LSB_JOBID:-local} started=\$(date -Is)\"
    python3 \"$SCRIPT\"
    echo \"finished=\$(date -Is)\"
  "
