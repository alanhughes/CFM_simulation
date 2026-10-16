"""
Reads in and processes MAR forcing data at specified borehole location, then saves to
processed data directory as pandas dataframe. Borehole location, dataframe column names, and save location
are specified in config file.

Then runs the CFM model using the processed forcing data and configuration from config file. Saves CFM output in
location specified in config file. Also saves the CFM configuration used for the run as a JSON file in the output directory.

Usage:
    python run_cfm.py
    python run_cfm.py --lat <latitude> --lon <longitude> --physrho <physrho_value> -rcm <rcm_name>
    python run_cfm.py ... --force    # rerun even if a complete run exists

"""

from force.process import ProcessMAR, ProcessRACMO, ProcessRACMO23
from sim.run import CFMRun
import argparse
import logging

logging.basicConfig(level=logging.INFO)

from cfm_config import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CFM simulation with optional command line arguments")
    parser.add_argument("--lat", type=float, help="Borehole latitude")
    parser.add_argument("--lon", type=float, help="Borehole longitude")
    parser.add_argument("--physrho", type=str, help="Physical densification scheme")
    parser.add_argument("--rcm", type=str, help="RCM name to use for forcing data (e.g., MAR or RACMO)")
    parser.add_argument("--liquid", type=str, help="Meltwater scheme to use (currently only bucket)")
    parser.add_argument("--force", action="store_true", help="Rerun the CFM even if a complete run exists")

    args = parser.parse_args()

    if (
        args.lat is not None
        and args.lon is not None
        and args.physrho is not None
        and args.rcm is not None
        and args.liquid is not None
    ):
        borehole_lat = args.lat
        borehole_lon = args.lon
        physRho = args.physrho
        rcm_name = args.rcm
        liquid = args.liquid
        logging.info(
            f"Using command line arguments: lat={borehole_lat}, lon={borehole_lon}, physrho={physRho}, rcm={rcm_name}, liquid={liquid}"
        )
    else:
        borehole_lat = config["borehole_lat"]
        borehole_lon = config["borehole_lon"]
        rcm_name = config["rcm_name"]
        physRho = config["cfm_config"]["physRho"]
        liquid = config["cfm_config"]["liquid"]
        logging.info("Using borehole location and physrho from config.yaml")

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
