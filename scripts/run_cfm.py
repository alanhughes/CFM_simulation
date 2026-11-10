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
    python run_cfm.py ... --start-year 1979 --end-year 2020    # other years
    python run_cfm.py ... --cfm-overrides configs/power_law.yaml    # cfm_config keys for this run only

Each argument that is not given comes from config.yaml (default_site, cfm_config.physRho, rcm_name, cfm_config.liquid).

The years are start_year and end_year of config.yaml, but the end year is clipped to the last complete year of the RCM
data, with a warning: the ANT-12 (RACMO) data stops at 2023-12-31, so a RACMO run is 1979-2023 (DISC-001). An end
year from --end-year is not clipped; if the data does not include it, the run stops. The years are resolved once here
(force.years.run_years) and given to sim.worker, which does not resolve them again.
To run at a new location, add it to sites.yaml and sites.lock first.

--cfm-overrides <file.yaml> gives cfm_config keys for this run only, without an edit of config.yaml: a YAML mapping of
cfm_config key to value, for example "Ksat_darcy: power_law". The keys go on top of cfm_config and the
cfm_config_by_liquid section of the scheme, before the run hash is made, so they are in the run's cfm_config and hash.
A value replaces the earlier value in full (a list such as outputs is not merged). The file must not set liquid or
physRho (use --liquid and --physrho), resultsFolder, or a key that the run sets from the forcing
(sim.run.REFUSED_CFM_OVERRIDE_KEYS). The file is checked here, before the forcing is made, and the run stops if it is
not correct. The manifest records the file stem as run.config_name and the absolute path as run.cfm_overrides; they
are not in the hash. The worker reads the file when it starts, so do not change the file while a run starts.

"""

import argparse
import logging
import os
import subprocess
import sys
from typing import List, Optional, Sequence

import sim
from cfm_config import ConfigError, Site, cfm_version, config, get_site
from force.process import ProcessMAR, ProcessRACMO, ProcessRACMO23
from force.years import RunYears, add_year_arguments, run_years, set_config_years
from sim.manifest import LIQUID_SCHEMES
from sim.run import load_cfm_overrides
from sim.identity import check_cfm_identity, check_wrapper_identity, measure_cfm_identity, wrapper_identity

logging.basicConfig(level=logging.INFO)

PROCESSORS = {"MAR": ProcessMAR, "RACMO": ProcessRACMO, "RACMO23": ProcessRACMO23}


def worker_command(
    version: dict,
    site: Site,
    physRho: str,
    rcm_name: str,
    liquid: str,
    years: RunYears,
    force: bool = False,
    cfm_overrides: Optional[str] = None,
) -> List[str]:
    """The sim.worker command line of one run. The worker uses the years given here; it does not resolve them.
    cfm_overrides is the path of the --cfm-overrides file, or None."""
    command = [version["python"], "-m", "sim.worker", "--lat", str(site.lat), "--lon", str(site.lon)]
    command += ["--site", site.name, "--physrho", physRho, "--rcm", rcm_name, "--liquid", liquid]
    command += ["--cfm", version["label"], "--start-year", str(years.start_year), "--end-year", str(years.end_year)]
    command += ["--requested-end-year", str(years.requested_end_year), "--end-year-source", years.end_year_source]
    if cfm_overrides is not None:
        command += ["--cfm-overrides", cfm_overrides]
    if force:
        command.append("--force")
    return command


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run CFM simulation with optional command line arguments")
    parser.add_argument("--site", type=str, help="Site name in sites.yaml (default: default_site in config.yaml)")
    parser.add_argument("--physrho", type=str, help="Physical densification scheme")
    parser.add_argument("--rcm", type=str, help="RCM name to use for forcing data (e.g., MAR or RACMO)")
    parser.add_argument("--liquid", type=str, choices=LIQUID_SCHEMES, help="Meltwater scheme to use")
    parser.add_argument("--force", action="store_true", help="Rerun the CFM even if a complete run exists")
    parser.add_argument(
        "--cfm", type=str, help="CFM version (a label in cfm_versions); default_cfm_version if not given"
    )
    parser.add_argument(
        "--cfm-overrides",
        type=str,
        metavar="FILE.yaml",
        help="YAML file of cfm_config keys for this run, on top of config.yaml (in cfm_config and the run hash); "
        "its stem is recorded as config_name in the manifest",
    )
    add_year_arguments(parser)

    args = parser.parse_args(argv)

    site = get_site(args.site or config["default_site"])
    physRho = args.physrho or config["cfm_config"]["physRho"]
    rcm_name = args.rcm or config["rcm_name"]
    liquid = args.liquid or config["cfm_config"]["liquid"]
    if rcm_name not in PROCESSORS:
        parser.error(f"Invalid RCM name: {rcm_name}. Choose one of {', '.join(PROCESSORS)}.")
    # check the overrides file now, not after the forcing has been processed. The worker gets the absolute path, as it
    # can start in a different folder
    cfm_overrides = None
    if args.cfm_overrides is not None:
        try:
            overrides = load_cfm_overrides(args.cfm_overrides)
        except ConfigError as error:
            parser.error(str(error))
        cfm_overrides = os.path.abspath(args.cfm_overrides)
        logging.info(f"cfm_config overrides from {cfm_overrides}: {overrides}")
    logging.info(
        f"Site {site.name} (lat={site.lat}, lon={site.lon}), physrho={physRho}, rcm={rcm_name}, liquid={liquid}"
    )

    # one decision on the years for the forcing here and the run in the worker (DISC-001)
    years = run_years(rcm_name, args.start_year, args.end_year)
    set_config_years(years)
    logging.info(f"{rcm_name}: years {years.start_year}-{years.end_year}")

    # check the CFM checkout and src/ now, not after the forcing has been processed (up to 40 minutes for RACMO)
    version = cfm_version(args.cfm)
    check_cfm_identity(measure_cfm_identity(version))
    check_wrapper_identity(wrapper_identity())

    PROCESSORS[rcm_name](site.lat, site.lon).process()

    # the worker imports this repo's src (the one this script imported), with the CFM version's python
    src_path = os.path.dirname(os.path.dirname(os.path.abspath(sim.__file__)))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [src_path, os.environ.get("PYTHONPATH")]))}
    command = worker_command(
        version, site, physRho, rcm_name, liquid, years, force=args.force, cfm_overrides=cfm_overrides
    )
    logging.info(f"Running CFM version {version['label']!r} with {version['python']}.")
    return subprocess.run(command, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
