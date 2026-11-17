"""Write an RCM index of the configured RCM data folder, for a machine with no RCM data, such as the HEC (FEAT-012).

The index (see force.rcm_index) holds the release, the file names of the data folder and the 2D latitude/longitude of
the grid. With the data path key of the RCM (for example RACMO_data_path) set to the index folder, scripts/run_cfm.py
finds the same model point, release and years as with the data folder, and reads the forcing CSV from the forcing
cache. It cannot make forcing.

Usage (from the repo root, on a machine with the RCM data):
    python scripts/make_rcm_index.py --rcm RACMO
    python scripts/make_rcm_index.py --rcm RACMO --out {CFM_share_data_path}/forcing/RACMO/<release>/rcm_index

The default --out is {forcing_data_path}/<rcm>/<release>/rcm_index. Then the index is next to the forcing CSVs of its
release, and a copy of the forcing folder (same relative path) also copies the index. scripts/sync_runs.py does not
copy it, as it copies only the .csv and .json files: give --out on the share, or copy the folder.

After the write, the script reads the index again and compares it with the data folder: the model point of each site
in sites.yaml (active or not) and the days of the data (force.years.data_days). It stops with an error if one differs.
The script stops if --out exists; it never replaces an index.
"""

import argparse
import dataclasses
import logging
import os
import sys

from cfm_config import config, sites
from force.cache import forcing_root
from force.model_point import _RCM_GRIDS, _grid, _release_and_files, clear_cache, model_point_at
from force.rcm_index import is_rcm_index, write_rcm_index
from force.years import data_days

logging.basicConfig(level=logging.INFO, format="%(message)s")


def check_index(rcm_name: str, data_path: str, index_path: str) -> int:
    """Compare the model points of all the sites and the data days of the index with the data folder. Returns the
    number of sites compared. Stops with an error at the first difference."""
    key = _RCM_GRIDS[rcm_name]["path_key"]
    results = {}
    for path in (data_path, index_path):
        clear_cache()
        cfg = {**config, key: path}
        saved = config[key]
        config[key] = path
        try:
            points = {name: model_point_at(s.lat, s.lon, rcm_name) for name, s in sites(active_only=False).items()}
            results[path] = (points, data_days(rcm_name, cfg))
        finally:
            config[key] = saved
    clear_cache()
    (points, days), (index_points, index_days) = results[data_path], results[index_path]
    if days != index_days:
        raise RuntimeError(f"The index gives the data days {index_days}, the data folder {days}")
    for name, mp in points.items():
        if dataclasses.astuple(mp) != dataclasses.astuple(index_points[name]):
            raise RuntimeError(f"{name}: the index gives {index_points[name]}, the data folder {mp}")
    logging.info(
        f"The index agrees with the data folder: {len(points)} sites, data {days.first_day} to {days.last_day}"
    )
    return len(points)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rcm", required=True, choices=sorted(_RCM_GRIDS), help="the RCM of the data folder")
    parser.add_argument("--out", help="the index folder (default: {forcing_data_path}/<rcm>/<release>/rcm_index)")
    args = parser.parse_args(argv)

    key = _RCM_GRIDS[args.rcm]["path_key"]
    data_path = config[key]
    if is_rcm_index(data_path):
        raise SystemExit(f"{key} ({data_path}) is an RCM index. Set it to the RCM data folder.")
    release, _ = _release_and_files(args.rcm, data_path)
    _, coord_file, lat2d, lon2d = _grid(args.rcm, data_path)
    out = args.out or os.path.join(forcing_root(), args.rcm, release, "rcm_index")
    write_rcm_index(
        out, args.rcm, release, data_path, os.listdir(data_path), os.path.basename(coord_file), lat2d, lon2d
    )
    logging.info(f"Wrote the {args.rcm} index of {data_path} to {out}")
    check_index(args.rcm, data_path, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
