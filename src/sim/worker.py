"""Runs one CFM simulation in this process. scripts/run_cfm.py starts it with the python of the chosen CFM version, as
one process can only import one CFM:

    <python> -m sim.worker --lat <lat> --lon <lon> [--site <name>] --physrho <physrho> --rcm <rcm> --liquid <liquid> --cfm <label>
"""

import argparse
import logging

from cfm_config import cfm_version
from sim.run import CFMRun

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one CFM simulation with this process's python")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--site", help="Site name in sites.yaml, recorded in the manifest (lat/lon are given too)")
    parser.add_argument("--physrho", required=True)
    parser.add_argument("--rcm", required=True)
    parser.add_argument("--liquid", required=True)
    parser.add_argument("--cfm", required=True, help="Label of the CFM version in cfm_versions")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    CFMRun(
        args.lat, args.lon, args.physrho, args.rcm, args.liquid, cfm_version(args.cfm), force=args.force, site=args.site
    ).run()
