#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
require_gridfm_data_base()

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
