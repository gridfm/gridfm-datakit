#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

echo "host=$(hostname) job=${LSB_JOBID:-local} started=$(date -Is)"
julia --project="$JULIA_PROJECT" "$SCRIPT_DIR/run_matrix.jl" \
  --scope small \
  --setup setup1
echo "finished=$(date -Is)"
