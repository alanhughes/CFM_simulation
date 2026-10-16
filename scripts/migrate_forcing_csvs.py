"""Plan the move of old forcing CSVs (cfm_input/<rcm>_<lat>_<lon>_<start>_<end>.csv) into the forcing cache (FEAT-002).

The default is a dry run: the script reads the old CSVs and the RCM coordinate arrays, and prints what it would do.
It does not write any file.

With --copy the script copies each CSV with the action COPY into the cache and writes the JSON file next to it. It
never moves or deletes a source file. It never copies a CSV with the action SKIP. It stops before any copy if the plan
has a CONFLICT (a different file is already in the cache, or different CSVs go to one cache file). It writes each file
to a temporary name and then renames it.

For each old CSV, the script:
1. Finds the site in sites.yaml with the same coordinates.
2. Finds the model point (nearest cell, BUG-002) and the cell that the old code selected (search in degrees).
3. SKIPs the CSV if the two cells are different. The CSV holds the data of the old cell, not of the model point.
   (Example: CI-0 and WI-70 on ANT-12. The reference runs in speersm/ used this data.)
4. Reads the dates in the CSV. If the data does not cover start_year to end_year (BUG-003), but covers complete years
   from start_year to an earlier end year, it proposes the cache name with that end year. The old code kept only the
   years in the range, so the data is the same as an extraction with the earlier end year.
5. Groups the CSVs that go to one cache file. Identical CSVs (same SHA-256) give one copy. Different CSVs are a
   CONFLICT.
6. With --verify, compares the T2m column of the first year with the RCM data at the model point and at the old cell
   (RACMO and RACMO23 only; reads one variable at one point from the share).

Usage:
    python scripts/migrate_forcing_csvs.py [--dry-run] <cfm_input folder> [<cfm_input folder> ...] [--verify]
    python scripts/migrate_forcing_csvs.py --copy <cfm_input folder> [<cfm_input folder> ...]
"""

import argparse
import datetime
import glob
import json
import os
import re
import shutil
import socket
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from cfm_config import sites
from force.cache import ForcingCoverageError, check_date_coverage, forcing_csv_path, forcing_json_path, sha256_file
from force.model_point import ModelPoint, _grid, model_point_at, rcm_data_path

OLD_NAME = re.compile(
    r"^(?P<rcm>MAR|RACMO|RACMO23)_(?P<lat>-?[\d.]+)_(?P<lon>-?[\d.]+)_(?P<start>\d{4})_(?P<end>\d{4})\.csv$"
)


@dataclass
class Plan:
    source: str
    rcm: str
    site: str
    model_point: Optional[ModelPoint] = None
    old_cell: Optional[tuple] = None
    first_date: str = ""
    last_date: str = ""
    target: Optional[str] = None
    action: str = ""
    note: str = ""
    sha256: str = ""


def old_cell(rcm_name: str, lat: float, lon: float) -> tuple:
    """The cell that the code before the BUG-002 fix selected: the smallest distance in degrees."""
    _, _, lat2d, lon2d = _grid(rcm_name, rcm_data_path(rcm_name))
    distance = np.sqrt((lat2d - lat) ** 2 + (lon2d - lon) ** 2)
    i, j = np.unravel_index(distance.argmin(), distance.shape)
    return int(i), int(j)


def complete_end_year(index: pd.DatetimeIndex, start: int, end: int, dataset: str) -> Optional[int]:
    """The last end year <= end for which the data covers complete years from start, or None."""
    for candidate in range(end, start - 1, -1):
        try:
            check_date_coverage(index[index.year <= candidate], start, candidate, dataset)
            return candidate
        except ForcingCoverageError:
            continue
    return None


def verify_t2m(plan: Plan, df: pd.DataFrame) -> str:
    """Compare the CSV T2m of the first year with the RCM tas at the model point and at the old cell."""
    if plan.rcm not in ("RACMO", "RACMO23") or "T2m" not in df.columns:
        return "verify: not available for this RCM"
    year = df.index[0].year
    data_path = rcm_data_path(plan.rcm)
    if plan.rcm == "RACMO":
        files = sorted(glob.glob(f"{data_path}/tas_*_day_{year}0101-*.nc"))
    else:
        files = sorted(glob.glob(f"{data_path}/tas.KNMI-{year}.*.nc"))
    if not files:
        return f"verify: no tas file for {year}"
    csv_t2m = df.loc[df.index.year == year, "T2m"].to_numpy()
    results = []
    with xr.open_dataset(files[0]) as ds:
        tas = ds["tas"].squeeze(drop=True)
        for label, (i, j) in (("model point", (plan.model_point.i, plan.model_point.j)), ("old cell", plan.old_cell)):
            series = tas.isel(rlat=i, rlon=j)
            series = series.sel(time=series["time"].dt.year == year).to_numpy()
            n = min(len(series), len(csv_t2m))
            match = n > 0 and np.allclose(series[:n], csv_t2m[:n], rtol=0, atol=1e-4)
            results.append(f"{label} {'matches' if match else 'differs'}")
    return "verify T2m " + str(year) + ": " + ", ".join(results)


def plan_csv(path: str, site_by_coords: dict, verify: bool) -> Optional[Plan]:
    match = OLD_NAME.match(os.path.basename(path))
    if not match:
        return None
    rcm, lat, lon = match["rcm"], float(match["lat"]), float(match["lon"])
    start, end = int(match["start"]), int(match["end"])
    plan = Plan(source=path, rcm=rcm, site=site_by_coords.get((lat, lon), "(no site)"))

    plan.model_point = model_point_at(lat, lon, rcm)
    plan.old_cell = old_cell(rcm, lat, lon)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    plan.first_date, plan.last_date = str(df.index.min().date()), str(df.index.max().date())
    plan.sha256 = sha256_file(path)
    notes = []
    if verify:
        notes.append(verify_t2m(plan, df))

    if plan.old_cell != (plan.model_point.i, plan.model_point.j):
        plan.action = "SKIP"
        notes.insert(
            0,
            f"data is from the old cell {plan.old_cell}, not the nearest cell "
            f"({plan.model_point.i}, {plan.model_point.j}) (BUG-002); extract again",
        )
        plan.note = "; ".join(notes)
        return plan

    dataset = f"{rcm} ({plan.model_point.release})"
    cache_end = complete_end_year(df.index, start, end, dataset)
    if cache_end is None:
        plan.action = "SKIP"
        notes.insert(0, f"the data does not cover complete years from {start}")
    else:
        plan.target = forcing_csv_path(plan.model_point, start, cache_end)
        plan.action = "COPY"
        if cache_end != end:
            notes.insert(0, f"data stops at {plan.last_date}: cache end year {cache_end}, not {end} (BUG-003)")
    plan.note = "; ".join(notes)
    return plan


def group_plans(plans: List[Plan]) -> None:
    """Group the copies by target: identical files give one copy, different files are a conflict."""
    by_target = defaultdict(list)
    for plan in plans:
        if plan.target:
            by_target[plan.target].append(plan)
    for target, group in by_target.items():
        hashes = {plan.sha256 for plan in group}
        if len(hashes) > 1:
            for plan in group:
                plan.action = "CONFLICT"
                plan.note = (plan.note + "; " if plan.note else "") + "different CSVs go to the same cache file"
        else:
            for plan in group[1:]:
                plan.action = "DUPLICATE"
                same = f"same data as {os.path.basename(group[0].source)}"
                plan.note = (plan.note + "; " if plan.note else "") + same
            if os.path.exists(target):
                same = sha256_file(target) in hashes
                for plan in group:
                    plan.action = "EXISTS" if same else "CONFLICT"
                    plan.note = (plan.note + "; " if plan.note else "") + (
                        "already in the cache" if same else "a different file is already in the cache"
                    )


def build_plans(folders: List[str], verify: bool) -> List[Plan]:
    site_by_coords = {(site.lat, site.lon): name for name, site in sites(active_only=False).items()}
    plans = []
    for folder in folders:
        for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
            plan = plan_csv(path, site_by_coords, verify)
            if plan is None:
                print(f"IGNORE   {path} (name does not match <rcm>_<lat>_<lon>_<start>_<end>.csv)")
            else:
                plans.append(plan)
    group_plans(plans)
    return plans


def cache_years(target: str) -> tuple:
    """(start_year, end_year) from the name <i>_<j>_<start>_<end>.csv."""
    _, _, start, end = os.path.splitext(os.path.basename(target))[0].split("_")
    return int(start), int(end)


def _write_atomic(path: str, write) -> None:
    """Call write(temporary_path), then rename the temporary file to path."""
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def copy_plan(plan: Plan) -> None:
    """Copy one CSV into the cache and write the JSON file next to it. The source file is not changed."""
    mp, target = plan.model_point, plan.target
    if os.path.exists(target) and sha256_file(target) != plan.sha256:
        raise RuntimeError(f"{target}: a different file is already in the cache; not overwritten")
    start, end = cache_years(target)
    df = pd.read_csv(plan.source, index_col=0, parse_dates=True)
    site_lat, site_lon = (float(x) for x in OLD_NAME.match(os.path.basename(plan.source)).group("lat", "lon"))
    record = {
        "model_point": {
            "rcm_name": mp.rcm_name,
            "release": mp.release,
            "i": mp.i,
            "j": mp.j,
            "cell_lat": mp.cell_lat,
            "cell_lon": mp.cell_lon,
            "method": mp.method,
            "tag": mp.tag,
        },
        "requested_location": {"lat": site_lat, "lon": site_lon, "distance_km": mp.distance_km},
        "start_year": start,
        "end_year": end,
        "first_date": str(pd.Timestamp(df.index.min())),
        "last_date": str(pd.Timestamp(df.index.max())),
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "data_path": rcm_data_path(mp.rcm_name),
        "sha256": plan.sha256,
        "csv_sha256": plan.sha256,
        "migrated_from": {"file": plan.source, "site": plan.site, "old_cell": list(plan.old_cell)},
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
    }

    def write_json(tmp):
        with open(tmp, "w") as file:
            json.dump(record, file, indent=2)
            file.write("\n")

    os.makedirs(os.path.dirname(target), exist_ok=True)
    _write_atomic(target, lambda tmp: shutil.copyfile(plan.source, tmp))
    if sha256_file(target) != plan.sha256:
        os.remove(target)
        raise RuntimeError(f"{target}: the copy has a different SHA-256 from the source; the copy is removed")
    _write_atomic(forcing_json_path(target), write_json)


def copy_all(plans: List[Plan]) -> int:
    """Copy the plans with the action COPY. Stop before any copy if the plan has a CONFLICT."""
    conflicts = [plan for plan in plans if plan.action == "CONFLICT"]
    if conflicts:
        for plan in conflicts:
            print(f"ERROR: CONFLICT {plan.source}: {plan.note}", file=sys.stderr)
        print("Nothing is copied.", file=sys.stderr)
        return 1
    copied = 0
    for plan in plans:
        if plan.action == "COPY":
            copy_plan(plan)
            copied += 1
            print(f"COPIED   {plan.source} -> {plan.target}")
    print(f"Copied {copied} CSVs. No source file is moved or deleted.")
    return 0


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+", help="Folders with old forcing CSVs (cfm_input/)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Only print the plan (the default)")
    mode.add_argument("--copy", action="store_true", help="Copy the CSVs with the action COPY into the cache")
    parser.add_argument("--verify", action="store_true", help="Compare T2m with the RCM data (reads the share)")
    args = parser.parse_args(argv)

    plans = build_plans(args.folders, args.verify)
    print("COPY MODE\n" if args.copy else "DRY RUN: no file is copied, moved or written.\n")
    for plan in plans:
        mp = plan.model_point
        print(f"{plan.action:9s} {plan.source}")
        print(
            f"          site {plan.site}, {mp.tag} ({mp.distance_km:.1f} km), old cell {plan.old_cell}, "
            f"dates {plan.first_date} to {plan.last_date}"
        )
        if plan.target:
            print(f"          -> {plan.target} (+ .json)")
        if plan.note:
            print(f"          {plan.note}")
    counts = defaultdict(int)
    for plan in plans:
        counts[plan.action] += 1
    print("\nSummary: " + ", ".join(f"{action} {n}" for action, n in sorted(counts.items())))
    return copy_all(plans) if args.copy else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
