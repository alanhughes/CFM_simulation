#!/bin/bash

# this script runs the plot_cfm.py script for all combinations of site and physrho values, using a single RCM and
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

# Site names in sites.yaml (the active sites)
mapfile -t SITES < <(python "${SCRIPT_DIR}/list_sites.py")
PHYSRHO_VALUES=("HLdynamic" "GSFC2020" "Crocus" "Barnola1991" "Ligtenberg2011")
RCM_VALUE="RACMO" # or "MAR"
LIQUID_VALUE="bucket"

# Function to run a single CFM plotting script
run_cfm_plot() {
    local site=$1
    shift 1
    local physrho_values=("$@")

    if python "${SCRIPT_DIR}/plot_cfm.py" \
        --site "$site" \
        --physrhols "${physrho_values[@]}" \
        --rcm "$RCM_VALUE" \
        --liquid "$LIQUID_VALUE"; then
        echo "SUCCESS: site=$site, physrho=${physrho_values[*]}"
    else
        echo "FAILED: site=$site, physrho=${physrho_values[*]}"
    fi
}

# loop over all sites, using all physrho values for each site
for SITE in "${SITES[@]}"; do
    run_cfm_plot "$SITE" "${PHYSRHO_VALUES[@]}"
done

echo "=== CFM Plot Script Completed at $(date) ==="
