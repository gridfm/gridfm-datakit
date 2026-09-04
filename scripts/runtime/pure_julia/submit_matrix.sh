#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LSB_LOG_DIR:-$HOME/.lsbatch}"

# Original paper LSF host set (CCC 7xx). Override resources if reproducing elsewhere.
HOST_SELECT="select[hname=='cccxc702' || hname=='cccxc703' || hname=='cccxc704' || hname=='cccxc705' || hname=='cccxc706' || hname=='cccxc707' || hname=='cccxc708' || hname=='cccxc709' || hname=='cccxc710' || hname=='cccxc711' || hname=='cccxc712' || hname=='cccxc713' || hname=='cccxc714' || hname=='cccxc715' || hname=='cccxc716']"
SPAN='span[hosts=1]'
EXCLUDE_7XX="select[hname!='cccxc701' && hname!='cccxc702' && hname!='cccxc703' && hname!='cccxc704' && hname!='cccxc705' && hname!='cccxc706' && hname!='cccxc707' && hname!='cccxc708' && hname!='cccxc709' && hname!='cccxc710' && hname!='cccxc711' && hname!='cccxc712' && hname!='cccxc713' && hname!='cccxc714' && hname!='cccxc715' && hname!='cccxc716']"
PACK_ONE_HOST='span[ptile=84]'
CORR_SLOTS="${CORRECTION_SLOTS:-84}"
CORR_MEM="${CORRECTION_MEM:-128G}"
CORR_WORKERS="${CORRECTION_WORKERS:-84}"

mkdir -p "$LOG_DIR"

submit() {
  local name="$1"
  local mem="$2"
  local slots="$3"
  local body="$4"
  local deps="${5:-}"
  local resources="${6:-$SPAN $HOST_SELECT}"
  local -a args=(
    -q normal
    -R "$resources"
    -M "$mem"
    -n "$slots"
    -J "$name"
    -o "$LOG_DIR/${name}_%J.out"
  )
  if [[ -n "$deps" ]]; then
    local dep_expr=""
    IFS=',' read -ra dep_ids <<< "$deps"
    for dep_id in "${dep_ids[@]}"; do
      [[ -n "$dep_expr" ]] && dep_expr+=" && "
      dep_expr+="done($dep_id)"
    done
    args+=(-w "$dep_expr")
  fi
  bsub "${args[@]}" "$body"
}

CORR_JOB=$(submit "julia_matrix_correct" "$CORR_MEM" "$CORR_SLOTS" \
  "export CORRECTION_WORKERS=$CORR_WORKERS; bash $SCRIPT_DIR/run_correction.sh" \
  "" "$PACK_ONE_HOST $EXCLUDE_7XX" | awk '{print $2}' | tr -d '<>')

# Historical LSF Max Memory: small ~205 GB, large ~870 GB (see outputs_julia/full_matrix/lsf_job_wall_times.md)
SMALL_S1_JOB=$(submit "julia_matrix_small_s1" "256G" "84" \
  "bash $SCRIPT_DIR/run_small_setup1.sh" | awk '{print $2}' | tr -d '<>')
LARGE_S1_JOB=$(submit "julia_matrix_large_s1" "960G" "84" \
  "bash $SCRIPT_DIR/run_large_setup1.sh" | awk '{print $2}' | tr -d '<>')
SMALL_S2_JOB=$(submit "julia_matrix_small_s2" "256G" "84" \
  "bash $SCRIPT_DIR/run_small_setup2.sh" "$CORR_JOB" | awk '{print $2}' | tr -d '<>')
LARGE_S2_JOB=$(submit "julia_matrix_large_s2" "960G" "84" \
  "bash $SCRIPT_DIR/run_large_setup2.sh" "$CORR_JOB" | awk '{print $2}' | tr -d '<>')

cat <<EOF
Submitted Julia benchmark matrix:
  correction:        $CORR_JOB  log=$LOG_DIR/julia_matrix_correct_${CORR_JOB}.out  (non-7xx, $CORR_SLOTS slots)
  small setup1:      $SMALL_S1_JOB  log=$LOG_DIR/julia_matrix_small_s1_${SMALL_S1_JOB}.out
  large setup1:      $LARGE_S1_JOB  log=$LOG_DIR/julia_matrix_large_s1_${LARGE_S1_JOB}.out
  small setup2:      $SMALL_S2_JOB  log=$LOG_DIR/julia_matrix_small_s2_${SMALL_S2_JOB}.out  (after $CORR_JOB)
  large setup2:      $LARGE_S2_JOB  log=$LOG_DIR/julia_matrix_large_s2_${LARGE_S2_JOB}.out  (after $CORR_JOB)
EOF

bjobs -w "$CORR_JOB" "$SMALL_S1_JOB" "$LARGE_S1_JOB" "$SMALL_S2_JOB" "$LARGE_S2_JOB" 2>/dev/null || true
