#!/bin/bash

# this script runs the run_cfm.py script for all combinations of lat/lon, physrho, and RCM values. You have the option
# to change the liquid scheme (e.g. to something other than "bucked") but currently they don't seem to be working in the
# CFM. I've let Max Stevens (maxstev@umd.edu) know about this and he said he would look into it.

# Define arrays of latitude/longitude pairs (same index = same location)
# LATITUDES=(-66.403 -66.588 -67.000 -67.444 -67.500)
# LONGITUDES=(-63.376 -63.212 -61.486 -64.953 -63.336)
# LATITUDES=(-67.10815 -68.30038 -68.39473 -68.39467 -67.56443)
# LONGITUDES=(-61.28748 -64.14850 -64.71308 -64.71017 -63.25733)
LATITUDES=(-68.0086 -68.0408)
LONGITUDES=(-64.3668 -64.3236)
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
    local lat=$1
    local lon=$2
    local physrho=$3
    local rcm=$4

    local log_file="$LOG_DIR/cfm_lat${lat}_lon${lon}_${physrho}_${rcm}_${TIMESTAMP}.log"

    echo "Running CFM for lat=$lat, lon=$lon, physrho=$physrho, rcm=$rcm (logging to $log_file)"

    if python "${SCRIPT_DIR}/run_cfm.py" \
        --lat "$lat" \
        --lon "$lon" \
        --physrho "$physrho" \
        --rcm "$rcm" \
        --liquid "$LIQUID_VALUE" \
        > "$log_file" 2>&1; then
        echo "SUCCESS: lat=$lat, lon=$lon, physrho=$physrho, rcm=$rcm"
        ((SUCCESSFUL_RUNS++))
    else
        echo "FAILED: lat=$lat, lon=$lon, physrho=$physrho, rcm=$rcm"
        ((FAILED_RUNS++))
    fi
}

# loop over all latitude/longitude pairs and physrho values
for i in "${!LATITUDES[@]}"; do
    LAT=${LATITUDES[$i]}
    LON=${LONGITUDES[$i]}
    for PHYSRHO in "${PHYSRHO_VALUES[@]}"; do
        for RCM in "${RCM_VALUES[@]}"; do
            ((TOTAL_RUNS++))
            run_cfm_job "$LAT" "$LON" "$PHYSRHO" "$RCM" || true
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
