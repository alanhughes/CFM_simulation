#!/bin/bash
#SBATCH -p serial
#SBATCH -J cfm_grid
#SBATCH -a 0-49

# Site names in sites.yaml (the coordinates come from sites.yaml).
# Keep #SBATCH -a equal to 0-(number of sites x physrho values x RCMs - 1).
SITES=(J108 J208 J409 J109 Y208)

# CFM parameters
PHYSRHO_VALUES=("GSFC2020" "HLdynamic" "Crocus" "Barnola1991" "Ligtenberg2011")
RCM_VALUES=("RACMO" "MAR")

# Shared parameters
LIQUID="bucket"

NUM_LOCATIONS=${#SITES[@]}
NUM_PHYSRHO=${#PHYSRHO_VALUES[@]}
NUM_RCM=${#RCM_VALUES[@]}
TOTAL_TASKS=$((NUM_LOCATIONS * NUM_PHYSRHO * NUM_RCM))

if (( SLURM_ARRAY_TASK_ID >= TOTAL_TASKS )); then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} exceeds TOTAL_TASKS=${TOTAL_TASKS}" >&2
    exit 1
fi

COMBOS_PER_LOCATION=$((NUM_PHYSRHO * NUM_RCM))
LOCATION_INDEX=$((SLURM_ARRAY_TASK_ID / COMBOS_PER_LOCATION))
COMBO_INDEX=$((SLURM_ARRAY_TASK_ID % COMBOS_PER_LOCATION))
PHYSRHO_INDEX=$((COMBO_INDEX / NUM_RCM))
RCM_INDEX=$((COMBO_INDEX % NUM_RCM))

SITE=${SITES[$LOCATION_INDEX]}
PHYSRHO=${PHYSRHO_VALUES[$PHYSRHO_INDEX]}
RCM=${RCM_VALUES[$RCM_INDEX]}

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# Override with e.g. `CFM_CONDA_ENV=CFM3.10.11 sbatch scripts/run_cfm_grid.sh`
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CFM_CONDA_ENV:-cfm}"

# SLURM runs a spooled copy of this script, so locate the repo from the submit directory
# (submit from the repo root) unless CFM_SIM_ROOT is set
REPO_ROOT="${CFM_SIM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_ROOT}/config.yaml" ]]; then
    echo "config.yaml not found in ${REPO_ROOT}; submit from the repo root or set CFM_SIM_ROOT" >&2
    exit 1
fi
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "Task ${SLURM_ARRAY_TASK_ID}/${TOTAL_TASKS}: location=${LOCATION_INDEX}, site=${SITE}, physrho=${PHYSRHO}, rcm=${RCM}, liquid=${LIQUID}"

python scripts/run_cfm.py \
    --site  "$SITE"   \
    --physrho "$PHYSRHO" \
    --rcm   "$RCM"    \
    --liquid "$LIQUID"
