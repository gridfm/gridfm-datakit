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

WORKERS="${CORRECTION_WORKERS:-84}"
host="$(hostname -s)"
slots_on_host=""
if [[ -n "${LSB_MCPU_HOSTS:-}" ]]; then
  read -ra parts <<< "$LSB_MCPU_HOSTS"
  for ((i = 0; i + 1 < ${#parts[@]}; i += 2)); do
    if [[ "${parts[i]}" == "$host" ]]; then
      slots_on_host="${parts[i + 1]}"
      break
    fi
  done
fi
if [[ -z "$slots_on_host" && -n "${LSB_DJOB_NUMPROC:-}" ]]; then
  slots_on_host="$LSB_DJOB_NUMPROC"
fi
if [[ -z "$slots_on_host" ]]; then
  slots_on_host="$(nproc)"
fi
if (( WORKERS > slots_on_host )); then
  WORKERS="$slots_on_host"
fi
echo "host=$host job=${LSB_JOBID:-local} lsb_slots=${LSB_DJOB_NUMPROC:-na} host_slots=$slots_on_host workers=$WORKERS started=$(date -Is)"
julia --project="$JULIA_PROJECT" "$SCRIPT_DIR/correct_scenarios.jl" --workers "$WORKERS"
echo "finished=$(date -Is)"
