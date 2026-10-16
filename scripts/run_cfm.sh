#!/bin/bash

# this script runs the run_cfm.py script for all combinations of site, physrho, and RCM values. You have the option
# to change the liquid scheme (e.g. to something other than "bucked") but currently they don't seem to be working in the
# CFM. I've let Max Stevens (maxstev@umd.edu) know about this and he said he would look into it.

# Site names in sites.yaml. The coordinates come from sites.yaml.
# To use a batch of sites.yaml: mapfile -t SITES < <(python scripts/list_sites.py --batch batch_1)
# SITES=(CI-0 CI-22 CI-120 WI-0 WI-70)
# SITES=(J108 J208 J409 J109 Y208)
SITES=(JP21 SI47)
PHYSRHO_VALUES=("GSFC2020" "HLdynamic" "Crocus" "Barnola1991" "Ligtenberg2011")
RCM_VALUES=("RACMO" "MAR")
LIQUID_VALUE="bucket"

# Counters for tracking results
TOTAL_RUNS=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0

# for logging individual runs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "$LOG_DIR"

# python scripts open config.yaml relative to the working directory
cd "$REPO_ROOT"

# for summary logging
exec > >(tee "$LOG_DIR/run_cfm_summary_${TIMESTAMP}.log") 2>&1

# Function to run a single CFM simulation
run_cfm_job() {
    local site=$1
    local physrho=$2
    local rcm=$3

    local log_file="$LOG_DIR/cfm_${site}_${physrho}_${rcm}_${TIMESTAMP}.log"

    echo "Running CFM for site=$site, physrho=$physrho, rcm=$rcm (logging to $log_file)"

    if python "${SCRIPT_DIR}/run_cfm.py" \
        --site "$site" \
        --physrho "$physrho" \
        --rcm "$rcm" \
        --liquid "$LIQUID_VALUE" \
        > "$log_file" 2>&1; then
        echo "SUCCESS: site=$site, physrho=$physrho, rcm=$rcm"
        ((SUCCESSFUL_RUNS++))
    else
        echo "FAILED: site=$site, physrho=$physrho, rcm=$rcm"
        ((FAILED_RUNS++))
    fi
}

# loop over all sites and physrho values
for SITE in "${SITES[@]}"; do
    for PHYSRHO in "${PHYSRHO_VALUES[@]}"; do
        for RCM in "${RCM_VALUES[@]}"; do
            ((TOTAL_RUNS++))
            run_cfm_job "$SITE" "$PHYSRHO" "$RCM" || true
        done
    done
done

# Print summary
echo "=================================="
echo "Execution Summary:"
echo "Total runs: $TOTAL_RUNS"
echo "Successful: $SUCCESSFUL_RUNS"
echo "Failed: $FAILED_RUNS"
echo "=================================="
