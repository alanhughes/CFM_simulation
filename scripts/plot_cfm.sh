#!/bin/bash

# this script runs the plot_cfm.py script for all combinations of lat/lon and physrho values, using a single RCM and
# liquid type. It logs the output to a timestamped log file in the logs directory.
# the plots which come from this aren't polished and would need to be edited for presentation, but act as a starting 
# point to quickly sanity check CFM output.

# Create log file with timestamp
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"

# python scripts open config.yaml relative to the working directory
cd "$REPO_ROOT"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/plot_cfm_$(date +%Y%m%d_%H%M%S).log"

# Redirect all output to log file and terminal
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== CFM Plot Script Started at $(date) ==="
echo "Log file: $LOG_FILE"

# Define arrays of latitude/longitude pairs (same index = same location)
LATITUDES=(-66.403 -66.588 -67.000 -67.444 -67.500)         
LONGITUDES=(-63.376 -63.212 -61.486 -64.953 -63.336)
PHYSRHO_VALUES=("HLdynamic" "GSFC2020" "Crocus" "Barnola1991" "Ligtenberg2011")
RCM_VALUE="RACMO" # or "MAR"
LIQUID_VALUE="bucket"

# Function to run a single CFM plotting script
run_cfm_plot() {
    local lat=$1
    local lon=$2
    shift 2
    local physrho_values=("$@")

    if python "${SCRIPT_DIR}/plot_cfm.py" \
        --lat "$lat" \
        --lon "$lon" \
        --physrhols "${physrho_values[@]}" \
        --rcm "$RCM_VALUE" \
        --liquid "$LIQUID_VALUE"; then
        echo "SUCCESS: lat=$lat, lon=$lon, physrho=${physrho_values[*]}"
    else
        echo "FAILED: lat=$lat, lon=$lon, physrho=${physrho_values[*]}"
    fi
}

# loop over all latitude/longitude pairs, using all physrho values for each location
for i in "${!LATITUDES[@]}"; do
    LAT=${LATITUDES[$i]}
    LON=${LONGITUDES[$i]}
    run_cfm_plot "$LAT" "$LON" "${PHYSRHO_VALUES[@]}"
done

echo "=== CFM Plot Script Completed at $(date) ==="
