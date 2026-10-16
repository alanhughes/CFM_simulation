"""Make the RACMO forcing CSVs for many sites in one pass over the share, with the fast h5py extraction (FEAT-003).

By default this does all the active sites in sites.yaml. With the cache (forcing_cache: true, the default), the first
run also keeps all grid cells in a box around the sites in cache.h5, so a later site near them takes seconds.
The CSVs and JSON files go to the forcing cache of FEAT-002, the same files as ProcessRACMO makes. Existing CSVs are
checked and skipped.

Usage (from the repo root):
    python scripts/extract_forcing.py                       # rcm_name from config, all active sites
    python scripts/extract_forcing.py --rcm RACMO23 --sites CI-0 WI-70
    python scripts/extract_forcing.py --rcm RACMO --no-cache --processes 4
    python scripts/extract_forcing.py --rcm RACMO --sites CI-0 WI-70 --end-year 2023   # ANT-12 stops in 2023

--start-year and --end-year replace start_year and end_year of config.yaml for this process only. They set the years
of the CSV name (<i>_<j>_<start>_<end>.csv) and of the date check (BUG-003).
"""

import argparse
import logging
from typing import Dict, Optional, Sequence

from cfm_config import Site, config, get_site, sites
from force.extract import DATASETS, process_fast
from force.process import ProcessRACMO, ProcessRACMO23

logging.basicConfig(level=logging.INFO)

PROCESSORS = {"RACMO": ProcessRACMO, "RACMO23": ProcessRACMO23}


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
    parser.add_argument("--start-year", type=int, help="first year (default: start_year in config.yaml)")
    parser.add_argument("--end-year", type=int, help="last year (default: end_year in config.yaml)")
    args = parser.parse_args(argv)

    # process_fast, ExtractSettings and the CSV path all read the years from config
    if args.start_year is not None:
        config["start_year"] = args.start_year
    if args.end_year is not None:
        config["end_year"] = args.end_year
    if config["start_year"] > config["end_year"]:
        parser.error(f"start year {config['start_year']} is after end year {config['end_year']}")
    logging.info(f"{args.rcm}: years {config['start_year']}-{config['end_year']}")

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
