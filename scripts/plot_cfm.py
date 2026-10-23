"""Script to plot the results of the CFM simulations for different physical densification schemes.
The script loads the CFM output data, creates plots for density profiles, DIP time series, and change in firn thickness
over time, and saves the figures to the specified figure paths. The script can be run with command line arguments for
the site (by name in sites.yaml) and physical densification schemes, or it can use the values specified in the
config.yaml file if no command line arguments are provided.

Usage:
    python plot_cfm.py --site CI-0 --physrhols GSFC2020 Crocus --rcm RACMO --liquid bucket [--cfm <label>]

The runs are found from their manifests (sim.runs.find_run), with the CFM version from --cfm (default_cfm_version if
not given).
"""

import argparse
import logging

from cfm_config import config, get_site
from plot.results import ResultsPlotter

logging.basicConfig(level=logging.INFO)


def default_phys_rho(cfm_config: dict) -> list:
    """The densification schemes to plot when no argument is given: the one scheme in cfm_config.physRho (BUG-201)."""
    return [cfm_config["physRho"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CFM plotting with optional command line arguments")
    parser.add_argument("--site", type=str, help="Site name in sites.yaml")
    parser.add_argument(
        "--physrhols",
        nargs="+",
        type=str,
        help="List of physical densification schemes",
    )
    parser.add_argument(
        "--rcm",
        type=str,
        help="RCM name to use for forcing data (e.g., MAR or RACMO)",
    )
    parser.add_argument(
        "--liquid",
        type=str,
        help="Meltwater scheme to use (currently only bucket)",
    )

    parser.add_argument(
        "--cfm", type=str, help="CFM version (a label in cfm_versions); default_cfm_version if not given"
    )

    args = parser.parse_args()
    if args.site is not None and args.physrhols is not None and args.rcm is not None and args.liquid is not None:
        logging.info(
            f"Using command line arguments: site={args.site}, physrho={args.physrhols}, rcm={args.rcm}, liquid={args.liquid}"
        )
        site = get_site(args.site)
        phys_rho = args.physrhols
        rcm_name = args.rcm
        liquid = args.liquid
    else:
        site = get_site(config["default_site"])
        phys_rho = default_phys_rho(config["cfm_config"])
        rcm_name = config["rcm_name"]
        liquid = config["cfm_config"]["liquid"]
        logging.info("Using borehole location and physrho from config.yaml")

    ResultsPlotter(site.name, phys_rho, rcm_name, liquid, cfm=args.cfm).plot()
