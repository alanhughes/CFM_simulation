"""Runs one CFM simulation in this process. scripts/run_cfm.py starts it with the python of the chosen CFM version, as
one process can only import one CFM:

    <python> -m sim.worker --lat <lat> --lon <lon> [--site <name>] --physrho <physrho> --rcm <rcm> --liquid <liquid>
        --cfm <label> --start-year <year> --end-year <year> [--requested-end-year <year> --end-year-source <source>]

The years are the ones that scripts/run_cfm.py resolved (force.years.run_years, DISC-001). The worker uses them as they
are and does not resolve them again, so the run uses the forcing CSV that run_cfm.py made. --requested-end-year and
--end-year-source (config or cli) only go into the manifest.
"""

import argparse
import logging

from cfm_config import cfm_version, config
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
    parser.add_argument("--start-year", type=int, required=True, help="first year of the run, used as it is")
    parser.add_argument("--end-year", type=int, required=True, help="last year of the run, used as it is")
    parser.add_argument("--requested-end-year", type=int, help="the end year that was asked for (default: --end-year)")
    parser.add_argument("--end-year-source", choices=("config", "cli"), default="cli", help="where it came from")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # the run code reads the years from config
    config["start_year"], config["end_year"] = args.start_year, args.end_year
    CFMRun(
        args.lat,
        args.lon,
        args.physrho,
        args.rcm,
        args.liquid,
        cfm_version(args.cfm),
        force=args.force,
        site=args.site,
        requested_end_year=args.end_year if args.requested_end_year is None else args.requested_end_year,
        end_year_source=args.end_year_source,
    ).run()
