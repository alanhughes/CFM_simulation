#!/bin/bash
#SBATCH -p serial
#SBATCH -J cfm_grid
#SBATCH -a 0-49

# Lat/lon pairs — same index = same location
LATITUDES=(-67.10815 -68.30038 -68.39473 -68.39467 -67.56443)
LONGITUDES=(-61.28748 -64.14850 -64.71308 -64.71017 -63.25733)

# CFM parameters
PHYSRHO_VALUES=("GSFC2020" "HLdynamic" "Crocus" "Barnola1991" "Ligtenberg2011")
RCM_VALUES=("RACMO" "MAR")

# Shared parameters
LIQUID="bucket"

NUM_LOCATIONS=${#LATITUDES[@]}
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

LAT=${LATITUDES[$LOCATION_INDEX]}
LON=${LONGITUDES[$LOCATION_INDEX]}
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
echo "Task ${SLURM_ARRAY_TASK_ID}/${TOTAL_TASKS}: location=${LOCATION_INDEX}, lat=${LAT}, lon=${LON}, physrho=${PHYSRHO}, rcm=${RCM}, liquid=${LIQUID}"

python scripts/run_cfm.py \
    --lat   "$LAT"    \
    --lon   "$LON"    \
    --physrho "$PHYSRHO" \
    --rcm   "$RCM"    \
    --liquid "$LIQUID"
