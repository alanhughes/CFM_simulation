"""
Reads in and processes MAR forcing data at specified borehole location, then saves to
processed data directory as pandas dataframe. Borehole location, dataframe column names, and save location
are specified in config file.

Then runs the CFM model using the processed forcing data and configuration from config file. Saves CFM output in
location specified in config file. Also saves the CFM configuration used for the run as a JSON file in the output directory.

Usage:
    python run_cfm.py
    python run_cfm.py --site <site name in sites.yaml> --physrho <physrho_value> --rcm <rcm_name> --liquid <liquid>
    python run_cfm.py ... --force    # rerun even if a complete run exists

Each argument that is not given comes from config.yaml (default_site, cfm_config.physRho, rcm_name, cfm_config.liquid).
To run at a new location, add it to sites.yaml and sites.lock first.

"""

from force.process import ProcessMAR, ProcessRACMO, ProcessRACMO23
from sim.run import CFMRun
import argparse
import logging

logging.basicConfig(level=logging.INFO)

from cfm_config import config, get_site

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CFM simulation with optional command line arguments")
    parser.add_argument("--site", type=str, help="Site name in sites.yaml (default: default_site in config.yaml)")
    parser.add_argument("--physrho", type=str, help="Physical densification scheme")
    parser.add_argument("--rcm", type=str, help="RCM name to use for forcing data (e.g., MAR or RACMO)")
    parser.add_argument("--liquid", type=str, help="Meltwater scheme to use (currently only bucket)")
    parser.add_argument("--force", action="store_true", help="Rerun the CFM even if a complete run exists")

    args = parser.parse_args()

    site = get_site(args.site or config["default_site"])
    borehole_lat, borehole_lon = site.lat, site.lon
    physRho = args.physrho or config["cfm_config"]["physRho"]
    rcm_name = args.rcm or config["rcm_name"]
    liquid = args.liquid or config["cfm_config"]["liquid"]
    logging.info(
        f"Site {site.name} (lat={borehole_lat}, lon={borehole_lon}), physrho={physRho}, rcm={rcm_name}, liquid={liquid}"
    )

    if rcm_name == "MAR":
        ProcessMAR(borehole_lat, borehole_lon).process()
    elif rcm_name == "RACMO":
        ProcessRACMO(borehole_lat, borehole_lon).process()
    elif rcm_name == "RACMO23":
        ProcessRACMO23(borehole_lat, borehole_lon).process()
    else:
        logging.error(f"Invalid RCM name: {rcm_name}. Please choose either 'MAR' or 'RACMO' or 'RACMO23'.")
        exit(1)

    CFMRun(borehole_lat, borehole_lon, physRho, rcm_name, liquid, force=args.force).run()
