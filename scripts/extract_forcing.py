"""Make the RACMO or MAR forcing CSVs for many sites in one pass over the share, with the fast h5py extraction (FEAT-003).

By default this does all the active sites in sites.yaml. With the cache (forcing_cache: true, the default), the first
run also keeps all grid cells in a box around the sites in cache.h5, so a later site near them takes seconds.
The CSVs and JSON files go to the forcing cache of FEAT-002, the same files as ProcessRACMO and ProcessMAR make. Existing CSVs are
checked and skipped.

Usage (from the repo root):
    python scripts/extract_forcing.py                       # rcm_name from config, all active sites
    python scripts/extract_forcing.py --rcm RACMO23 --sites CI-0 WI-70
    python scripts/extract_forcing.py --rcm RACMO --no-cache --processes 4
    python scripts/extract_forcing.py --rcm MAR
    python scripts/extract_forcing.py --rcm MAR --end-year 2020

The years are start_year and end_year of config.yaml, but the end year is clipped to the last complete year of the RCM
data, with a warning (DISC-001): the ANT-12 (RACMO) data stops at 2023-12-31, so its CSVs are 1979-2023, the years of
scripts/run_cfm.py. --start-year and --end-year replace them and are used as they are. The years set the CSV name
(<i>_<j>_<start>_<end>.csv) and the date check (BUG-003).
"""

import argparse
import logging
from typing import Dict, Optional, Sequence

from cfm_config import Site, config, get_site, sites
from force.extract import DATASETS, process_fast
from force.process import ProcessMAR, ProcessRACMO, ProcessRACMO23
from force.years import add_year_arguments, run_years, set_config_years

logging.basicConfig(level=logging.INFO)

PROCESSORS = {"MAR": ProcessMAR, "RACMO": ProcessRACMO, "RACMO23": ProcessRACMO23}


def select_sites(names: Optional[Sequence[str]] = None) -> Dict[str, Site]:
    """The active sites in sites.yaml, or the given site names (active or not)."""
    if names:
        return {name: get_site(name) for name in names}
    return sites()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--rcm", default=config["rcm_name"], choices=sorted(DATASETS), help="dataset (default: rcm_name)"
    )
    parser.add_argument("--sites", nargs="+", help="site names in sites.yaml (default: all active sites)")
    parser.add_argument("--no-cache", action="store_true", help="read the share directly, do not use cache.h5")
    parser.add_argument("--processes", type=int, help="worker processes (default: forcing_processes, or 8)")
    parser.add_argument("--overwrite", action="store_true", help="write the CSVs again if they exist")
    add_year_arguments(parser)
    args = parser.parse_args(argv)

    # process_fast, ExtractSettings and the CSV path all read the years from config
    years = run_years(args.rcm, args.start_year, args.end_year)
    set_config_years(years)
    logging.info(f"{args.rcm}: years {years.start_year}-{years.end_year}")

    selected = select_sites(args.sites)
    processors = [PROCESSORS[args.rcm](site.lat, site.lon) for site in selected.values()]
    for name, processor in zip(selected, processors):
        logging.info(f"{name}: model point {processor.model_point.tag}")
    overrides = {"processes": args.processes} if args.processes else {}
    use_cache = False if args.no_cache else None
    written = process_fast(processors, use_cache=use_cache, overwrite=args.overwrite, **overrides)
    logging.info(f"{len(written)} forcing CSVs written")


if __name__ == "__main__":
    main()
