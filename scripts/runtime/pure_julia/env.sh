# Shared environment for PowerModels Julia runtime jobs.
# Caller must set SCRIPT_DIR to this directory.

REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/venv/bin/activate" ]]; then
  # Optional local venv; not required if `julia` is already on PATH.
  # shellcheck disable=SC1091
  source "$REPO_ROOT/venv/bin/activate"
fi

# Paper package set lives next to these scripts (see Project.toml / Manifest.toml).
export JULIA_PROJECT="${JULIA_PROJECT:-$SCRIPT_DIR}"
export JULIA_CPU_TARGET="${JULIA_CPU_TARGET:-generic}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Setup 2 has no default path. Set it before run_*_setup2.sh or run_correction.sh:
#   export GRIDFM_DATA_BASE=/path/to/finetuning
# Expected layout: $GRIDFM_DATA_BASE/{pf,opf}/<network>/powermodels/scenario_*_corrected.json
require_gridfm_data_base() {
  if [[ -z "${GRIDFM_DATA_BASE:-}" ]]; then
    echo "error: GRIDFM_DATA_BASE is not set." >&2
    echo "Download https://huggingface.co/datasets/gridfm/reproducibility-powermodels-setup2" >&2
    echo "and export GRIDFM_DATA_BASE=/path/to/that/tree." >&2
    echo "Expected: \$GRIDFM_DATA_BASE/{pf,opf}/<network>/powermodels/scenario_*_corrected.json" >&2
    exit 1
  fi
}

JULIA_JOB_DEPOT="${TMPDIR:-/tmp}/gridfm_julia_${LSB_JOBID:-local}_$$"
mkdir -p "$JULIA_JOB_DEPOT"
trap 'rm -rf "$JULIA_JOB_DEPOT"' EXIT
export JULIA_DEPOT_PATH="$JULIA_JOB_DEPOT:$HOME/.julia_shared:$HOME/.julia"
