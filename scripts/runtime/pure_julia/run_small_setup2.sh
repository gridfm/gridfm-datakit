#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

cd "$REPO_ROOT"
source "$REPO_ROOT/venv/bin/activate"

export JULIA_PROJECT="$REPO_ROOT/venv/julia_env"
export JULIA_CPU_TARGET=generic
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

JULIA_JOB_DEPOT="${TMPDIR:-/tmp}/gridfm_julia_${LSB_JOBID:-local}_$$"
mkdir -p "$JULIA_JOB_DEPOT"
trap 'rm -rf "$JULIA_JOB_DEPOT"' EXIT
export JULIA_DEPOT_PATH="$JULIA_JOB_DEPOT:$HOME/.julia_shared:$HOME/.julia"

echo "host=$(hostname) job=${LSB_JOBID:-local} started=$(date -Is)"
julia --project="$JULIA_PROJECT" "$SCRIPT_DIR/run_matrix.jl" \
  --scope small \
  --setup setup2
echo "finished=$(date -Is)"
