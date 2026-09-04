#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
require_gridfm_data_base()

echo "host=$(hostname) job=${LSB_JOBID:-local} started=$(date -Is)"
julia --project="$JULIA_PROJECT" "$SCRIPT_DIR/run_matrix.jl" \
  --scope large \
  --setup setup2
echo "finished=$(date -Is)"
