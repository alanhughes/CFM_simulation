"""Make the RACMO forcing CSVs for many sites in one pass over the share, with the fast h5py extraction (FEAT-003).

By default this does all borehole_sites in config.yaml. With the cache (forcing_cache: true, the default), the first
run also keeps all grid cells in a box around the sites in cache.h5, so a later site near them takes seconds.
The CSVs go to the same paths as the CSVs of ProcessRACMO (force_data_save_path_pattern). Existing CSVs are skipped.

Usage (from the repo root):
    python scripts/extract_forcing.py                       # rcm_name from config, all borehole_sites
    python scripts/extract_forcing.py --rcm RACMO23 --sites CI-0 WI-70
    python scripts/extract_forcing.py --rcm RACMO --no-cache --processes 4
"""

import argparse
import logging
from typing import Dict, Optional, Sequence, Tuple

from cfm_config import config
from force.extract import DATASETS, write_forcing_csvs

logging.basicConfig(level=logging.INFO)


def site_targets(rcm_name: str, names: Optional[Sequence[str]] = None) -> Dict[str, Tuple[float, float]]:
    """CSV path -> (lat, lon) for the borehole_sites in config (all, or the given names).

    The only place in this script that knows about borehole_sites and the CSV names; FEAT-002 replaces both.
    """
    sites = config["borehole_sites"]
    unknown = sorted(set(names or []) - set(sites))
    if unknown:
        raise SystemExit(f"Not in borehole_sites: {unknown}")
    targets = {}
    for name in names or sites:
        lat, lon = sites[name]
        path = config["force_data_save_path_pattern"].format(
            CFM_data_path=config["CFM_data_path"],
            rcm_name=rcm_name,
            borehole_lat=lat,
            borehole_lon=lon,
            start_year=config["start_year"],
            end_year=config["end_year"],
        )
        targets[path] = (lat, lon)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--rcm", default=config["rcm_name"], choices=sorted(DATASETS), help="dataset (default: rcm_name)"
    )
    parser.add_argument("--sites", nargs="+", help="names in borehole_sites (default: all)")
    parser.add_argument("--no-cache", action="store_true", help="read the share directly, do not use cache.h5")
    parser.add_argument("--processes", type=int, help="worker processes (default: forcing_processes, or 8)")
    parser.add_argument("--overwrite", action="store_true", help="write the CSVs again if they exist")
    args = parser.parse_args()

    overrides = {"processes": args.processes} if args.processes else {}
    # the cache box covers all borehole_sites, also when only some of them are asked for
    all_sites = list(site_targets(args.rcm).values())
    written = write_forcing_csvs(
        args.rcm,
        site_targets(args.rcm, args.sites),
        overwrite=args.overwrite,
        use_cache=False if args.no_cache else None,
        cache_sites=all_sites,
        **overrides,
    )
    logging.info(f"{len(written)} forcing CSVs written")


if __name__ == "__main__":
    main()
