"""
Reads in and processes MAR forcing data at specified borehole location, then saves to
processed data directory as pandas dataframe. Borehole location, dataframe column names, and save location
are specified in config file.

Then runs the CFM model using the processed forcing data and configuration from config file. Saves the CFM output, the
CFM configuration JSON and the manifest in {CFM_data_path}/runs/<model point>/<physRho>_<liquid>_<start>-<end>/<cfm label>/<hash8>/
(FEAT-001). A run with the same hash that is complete is skipped.

The CFM runs in a subprocess (sim.worker) with the python of the CFM version from --cfm (default_cfm_version if not
given), as one process can only import one CFM.

Usage:
    python run_cfm.py
    python run_cfm.py --site <site name in sites.yaml> --physrho <physrho_value> --rcm <rcm_name> --liquid <liquid>
    python run_cfm.py ... --force    # rerun even if a complete run exists
    python run_cfm.py ... --cfm <label>    # CFM version from cfm_versions in config.yaml

Each argument that is not given comes from config.yaml (default_site, cfm_config.physRho, rcm_name, cfm_config.liquid).
To run at a new location, add it to sites.yaml and sites.lock first.

"""

from force.process import ProcessMAR, ProcessRACMO, ProcessRACMO23
from sim.identity import check_cfm_identity, check_wrapper_identity, measure_cfm_identity, wrapper_identity
import argparse
import logging
import os
import subprocess
import sys

import sim

logging.basicConfig(level=logging.INFO)

from cfm_config import cfm_version, config, get_site

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CFM simulation with optional command line arguments")
    parser.add_argument("--site", type=str, help="Site name in sites.yaml (default: default_site in config.yaml)")
    parser.add_argument("--physrho", type=str, help="Physical densification scheme")
    parser.add_argument("--rcm", type=str, help="RCM name to use for forcing data (e.g., MAR or RACMO)")
    parser.add_argument("--liquid", type=str, help="Meltwater scheme to use (currently only bucket)")
    parser.add_argument("--force", action="store_true", help="Rerun the CFM even if a complete run exists")
    parser.add_argument("--cfm", type=str, help="CFM version (a label in cfm_versions); default_cfm_version if not given")

    args = parser.parse_args()

    site = get_site(args.site or config["default_site"])
    borehole_lat, borehole_lon = site.lat, site.lon
    physRho = args.physrho or config["cfm_config"]["physRho"]
    rcm_name = args.rcm or config["rcm_name"]
    liquid = args.liquid or config["cfm_config"]["liquid"]
    logging.info(
        f"Site {site.name} (lat={borehole_lat}, lon={borehole_lon}), physrho={physRho}, rcm={rcm_name}, liquid={liquid}"
    )

    # check the CFM checkout and src/ now, not after the forcing has been processed (up to 40 minutes for RACMO)
    version = cfm_version(args.cfm)
    check_cfm_identity(measure_cfm_identity(version))
    check_wrapper_identity(wrapper_identity())

    if rcm_name == "MAR":
        ProcessMAR(borehole_lat, borehole_lon).process()
    elif rcm_name == "RACMO":
        ProcessRACMO(borehole_lat, borehole_lon).process()
    elif rcm_name == "RACMO23":
        ProcessRACMO23(borehole_lat, borehole_lon).process()
    else:
        logging.error(f"Invalid RCM name: {rcm_name}. Please choose either 'MAR' or 'RACMO' or 'RACMO23'.")
        exit(1)

    # the worker imports this repo's src (the one this script imported), with the CFM version's python
    src_path = os.path.dirname(os.path.dirname(os.path.abspath(sim.__file__)))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [src_path, os.environ.get("PYTHONPATH")]))}
    command = [version["python"], "-m", "sim.worker", "--lat", str(borehole_lat), "--lon", str(borehole_lon)]
    command += ["--site", site.name, "--physrho", physRho, "--rcm", rcm_name, "--liquid", liquid, "--cfm", version["label"]]
    if args.force:
        command.append("--force")
    logging.info(f"Running CFM version {version['label']!r} with {version['python']}.")
    sys.exit(subprocess.run(command, env=env).returncode)
