"""Plan the copy of old CFM output folders into the run layout of FEAT-001, and copy them with --copy.

Old folders: <cfm_output>/CFMoutput_<lat>_<lon>_<start>_<end>_<physRho>_<liquid>_<rcm>/, with the CFM config JSON
<cfm_output>/CFMconfig_<same>.json (if it exists) and the forcing CSV <cfm_output>/../cfm_input/<rcm>_<lat>_<lon>_<start>_<end>.csv.
New folders: {CFM_data_path}/runs/<model point>/<physRho>_<liquid>_<start>-<end>/<cfm label>/<hash8>/ (see sim.manifest).

The default is a dry run: the script reads the folder names, the CFM config JSON files, the forcing CSVs and the RCM
coordinate arrays, and prints what it would do. It does not write any file.

With --copy the script copies the .hdf5 files of each folder with the action COPY, writes CFMconfig.json (a copy of the
old JSON), run.log and, last, manifest.json. It never moves, deletes or changes a source file. It stops before any copy
if the plan has a CONFLICT, or if the destination disk does not have the space. It writes each file to a temporary name
and then renames it. A copy that stops leaves a folder with no manifest; sim.run does not use such a folder.

An old folder has no run hash, and the CFM identity, the Python environment and the wrapper code of the old run are not
known. The script gives it a migrated run hash: the SHA-256 of a document with hash_version "migrated-2" and the items
that are known, and the sizes of the .hdf5 files (see migrated_hash_inputs). A new run never has the same hash, so
sim.run never skips a new run because of a migrated run. Two old folders with the same known items and file sizes (for
example the same run in speersm/ and in ~/CFM_data/, or two sites in one cell) get the same hash: the script copies
one, and --copy compares the files of the other byte for byte (DUPLICATE). It stops if they are different.

The old folders do not record spin_end_year, and the speersm/ runs used two values (1995 and 1989, MIG-001). The script
measures it from the repeated blocks of the spin-up in CFMforcing.hdf5 (measure_spin, FEAT-004). The migrated hash
contains it, the manifest records it in run.spin_end_year (the key of a new run) and the other measured values in
migrated.spin. If it cannot be measured, the value is None and the plan has the note "the spin-up is not measured".
The dry run shows spin_end_year for each folder, and the summary gives the COPY folders of each label by spin_end_year.
The migrated-1 hash (FEAT-001) had no spin_end_year, so a migrated-2 hash never agrees with a migrated-1 hash.

The manifest records the site of each old folder (the site in sites.yaml with the same coordinates to 3 decimals) and,
in run.site_coords, the coordinates of the old folder name, not those of sites.yaml. Each site of a DUPLICATE group
has the coordinates of its own folder. find_runs compares them with sites.yaml to 3 decimals (see sim.runs).

The model point of an old run is the cell whose data is in its forcing CSV: the cell of the search in degrees that the
code used before the BUG-002 fix. For CI-0 and WI-70 on RACMO (ANT-12) and CI-22 on MAR, this is not the model point of
the site now. find_runs(site=...) does not give these runs; use find_runs(model_point=...).

Each source has a CFM label, which the user gives, because the old folders do not record the CFM version.

Usage:
    python scripts/migrate_runs.py --source 3e3bf7e-speersm ~/luna/CPOM/speersm/CFM_data/cfm_output \\
        --source ab871c8 ~/CFM_data/ab871c8/cfm_output                     # dry run (the default)
    python scripts/migrate_runs.py ... --match _GSFC2020_bucket_RACMO      # only the folders with one of these texts in the name
    python scripts/migrate_runs.py ... --copy                              # copy into {CFM_data_path}/runs
"""

import argparse
import datetime
import functools
import glob
import importlib.util
import json
import os
import re
import shutil
import socket
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

import h5py
import numpy as np
import pandas as pd

from cfm_config import config, format_coord, sites
from force.cache import sha256_file
from force.model_point import ModelPoint, _grid, rcm_data_path
from sim.identity import wrapper_identity
from sim.manifest import COMPLETE, WRAPPER_PARAMS, read_manifest, run_folder, run_hash, runs_root, write_manifest

_spec = importlib.util.spec_from_file_location(
    "migrate_forcing_csvs", os.path.join(os.path.dirname(os.path.abspath(__file__)), "migrate_forcing_csvs.py")
)
_forcing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_forcing)
old_cell = _forcing.old_cell
complete_end_year = _forcing.complete_end_year

# each forcing CSV is used by about 5 runs (one for each scheme)
_sha256 = functools.lru_cache(maxsize=None)(sha256_file)


@functools.lru_cache(maxsize=None)
def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


OLD_FOLDER = re.compile(
    r"^CFMoutput_(?P<lat>-?[\d.]+)_(?P<lon>-?[\d.]+)_(?P<start>\d{4})_(?P<end>\d{4})"
    r"_(?P<physRho>[^_]+)_(?P<liquid>[^_]+)_(?P<rcm>MAR|RACMO|RACMO23)$"
)
OLD_CSV = re.compile(r"^(?P<rcm>MAR|RACMO|RACMO23)_(?P<lat>-?[\d.]+)_(?P<lon>-?[\d.]+)_(?P<start>\d{4})_(?P<end>\d{4})\.csv$")

MIGRATED_HASH_VERSION = "migrated-2"

# cfm_config keys that do not describe the run: the paths
_UNHASHED_CFM_CONFIG_KEYS = ("resultsFolder",)


@dataclass
class RunPlan:
    source: str
    label: str
    rcm: str
    physRho: str
    liquid: str
    lat: float
    lon: float
    start: int
    end: int
    site: str = "(no site)"
    config_json: Optional[str] = None
    cfm_config: Optional[dict] = None
    forcing_csv: Optional[str] = None
    forcing_sha256: Optional[str] = None
    model_point: Optional[ModelPoint] = None
    spin: Optional[dict] = None  # measure_spin() of CFMforcing.hdf5, None if it is not measured
    files: List[str] = field(default_factory=list)
    size: int = 0
    finished: str = ""
    hash_inputs: dict = field(default_factory=dict)
    run_hash: str = ""
    target: str = ""
    action: str = ""
    notes: List[str] = field(default_factory=list)
    sites: List[str] = field(default_factory=list)
    site_coords: dict = field(default_factory=dict)  # {site: [lat, lon]}, the coordinates of the old folder name

    @property
    def spin_end_year(self) -> Optional[int]:
        return self.spin["spin_end_year"] if self.spin is not None else None


def cell_model_point(rcm: str, i: int, j: int, lat: float, lon: float) -> ModelPoint:
    """A ModelPoint for the cell (i, j), with the distance from (lat, lon)."""
    from force.grid import great_circle_km

    release, _, lat2d, lon2d = _grid(rcm, rcm_data_path(rcm))
    cell_lat, cell_lon = float(lat2d[i, j]), float(lon2d[i, j])
    return ModelPoint(rcm, release, i, j, cell_lat, cell_lon, float(great_circle_km(lat, lon, cell_lat, cell_lon)),
                      method="degrees (before BUG-002)")


def find_forcing_csv(cfm_output: str, rcm: str, lat: float, lon: float, start: int, end: int) -> Optional[str]:
    """The old forcing CSV of a run in <cfm_output>/../cfm_input, by the coordinates (the names have 3 or 4 decimals)."""
    candidates = []
    for path in glob.glob(os.path.join(os.path.dirname(cfm_output.rstrip("/")), "cfm_input", f"{rcm}_*.csv")):
        match = OLD_CSV.match(os.path.basename(path))
        if not match or match["rcm"] != rcm or (int(match["start"]), int(match["end"])) != (start, end):
            continue
        csv_lat, csv_lon = float(match["lat"]), float(match["lon"])
        if (csv_lat, csv_lon) == (lat, lon):
            return path
        if (format_coord(csv_lat), format_coord(csv_lon)) == (format_coord(lat), format_coord(lon)):
            candidates.append(path)
    return candidates[0] if len(candidates) == 1 else None


def measure_spin(path: str, start_year: int) -> Optional[dict]:
    """The spin-up of an old run, measured from its CFMforcing.hdf5 (FEAT-004).

    The CFM (makeSpinFiles) puts the forcing from spin_date_st to spin_date_end, repeated a whole number of times,
    before the forcing of the main run. The spin-up is the steps of main/time before the first time that is not before
    start_year. The block length L is the smallest length such that the number of spin-up steps is a multiple of L and
    each forcing dataset of main/, in each block of L spin-up steps, is identical to the first L steps of the main run.
    spin_end_year is the year of the main run at step L - 1. Do not use the length of the spin-up: the CFM finds the
    number of blocks from the firn depth and the accumulation, so it changes from site to site.

    Returns {"spin_end_year", "spin_start" (the first time of the spin-up), "block_steps" (L), "blocks"}, or None if
    the file is not a readable HDF5 file, it has no spin-up, or no L agrees.
    """
    try:
        with h5py.File(path, "r") as file:
            time = file["main/time"][()]
            # the forcing datasets: the 1-D datasets of main/ with one value for each step, except the time
            forcing = {
                name: dataset[()]
                for name, dataset in file["main"].items()
                if name != "time" and isinstance(dataset, h5py.Dataset) and dataset.shape == time.shape
            }
    except (OSError, KeyError):
        return None
    if time.ndim != 1 or not forcing:
        return None
    start = int(np.searchsorted(time, start_year, side="left"))  # the first time that is not before start_year
    spin_steps, main_steps = start, len(time) - start
    if spin_steps == 0 or main_steps == 0:
        return None

    years = np.floor(time[start:]).astype(int)
    # a block ends at the end of a year of the main run: the lengths L with a new year at step L, or the whole main run
    year_ends = list(np.flatnonzero(np.diff(years)) + 1) + [main_steps]
    candidates = [int(n) for n in year_ends if spin_steps % n == 0]

    def same(values, n):
        blocks = values[:spin_steps].reshape(-1, n)
        first = values[start : start + n]
        return all(np.array_equal(block, first, equal_nan=values.dtype.kind == "f") for block in blocks)

    # a quick check of one dataset before all of them: T2m changes at each step
    quick = forcing.get("T2m", next(iter(forcing.values())))
    for n in candidates:
        if same(quick, n) and all(same(values, n) for values in forcing.values()):
            return {
                "spin_end_year": int(years[n - 1]),
                "spin_start": float(time[0]),
                "block_steps": n,
                "blocks": spin_steps // n,
            }
    return None


def migrated_hash_inputs(plan: RunPlan) -> dict:
    """The document of the migrated run hash: the items of an old run that are known. See the module docstring."""
    cfm_config = None
    if plan.cfm_config is not None:
        cfm_config = {key: value for key, value in plan.cfm_config.items() if key not in _UNHASHED_CFM_CONFIG_KEYS}
    return {
        "hash_version": MIGRATED_HASH_VERSION,
        "cfm_config": cfm_config,
        "physRho": plan.physRho,
        "liquid": plan.liquid,
        "start_year": plan.start,
        "end_year": plan.end,
        "spin_end_year": plan.spin_end_year,
        "model_point": {"tag": plan.model_point.tag, "release": plan.model_point.release},
        "forcing_csv_sha256": plan.forcing_sha256,
        "cfm_label": plan.label,
        # a fingerprint of the results: two old runs with the same known items but different results get different
        # hashes. Runs with the same sizes are compared byte for byte in --copy (DUPLICATE)
        "hdf5_sizes": {name: os.path.getsize(os.path.join(plan.source, name)) for name in plan.files},
    }


def plan_folder(path: str, label: str, site_by_coords: dict) -> Optional[RunPlan]:
    match = OLD_FOLDER.match(os.path.basename(path.rstrip("/")))
    if not match:
        return None
    cfm_output = os.path.dirname(path.rstrip("/"))
    plan = RunPlan(
        source=path,
        label=label,
        rcm=match["rcm"],
        physRho=match["physRho"],
        liquid=match["liquid"],
        lat=float(match["lat"]),
        lon=float(match["lon"]),
        start=int(match["start"]),
        end=int(match["end"]),
    )
    plan.site = site_by_coords.get((format_coord(plan.lat), format_coord(plan.lon)), "(no site)")
    plan.sites = [plan.site] if plan.site != "(no site)" else []
    plan.site_coords = {site: [plan.lat, plan.lon] for site in plan.sites}

    results = os.path.join(path, "CFMresults.hdf5")
    if not os.path.exists(results):
        plan.action = "SKIP"
        plan.notes.append("no CFMresults.hdf5: the run did not complete")
        return plan
    plan.files = sorted(name for name in os.listdir(path) if name.endswith(".hdf5"))
    plan.size = sum(os.path.getsize(os.path.join(path, name)) for name in plan.files)
    plan.finished = datetime.datetime.fromtimestamp(os.path.getmtime(results)).astimezone().isoformat(timespec="seconds")

    config_json = os.path.join(cfm_output, os.path.basename(path).replace("CFMoutput_", "CFMconfig_", 1) + ".json")
    if os.path.exists(config_json):
        plan.config_json = config_json
        with open(config_json) as file:
            plan.cfm_config = json.load(file)
    else:
        plan.notes.append("no CFMconfig JSON: cfm_config is not known")

    i, j = old_cell(plan.rcm, plan.lat, plan.lon)
    plan.model_point = cell_model_point(plan.rcm, i, j, plan.lat, plan.lon)

    plan.forcing_csv = find_forcing_csv(cfm_output, plan.rcm, plan.lat, plan.lon, plan.start, plan.end)
    if plan.forcing_csv is None:
        plan.notes.append("no forcing CSV in ../cfm_input: the forcing is not known")
    else:
        plan.forcing_sha256 = _sha256(plan.forcing_csv)
        df = _read_csv(plan.forcing_csv)
        if os.path.getmtime(plan.forcing_csv) > os.path.getmtime(results):
            plan.notes.append("the forcing CSV is newer than the results: it is possibly not the forcing of this run")
        end = complete_end_year(df.index, plan.start, plan.end, plan.forcing_csv)
        if end is not None and end != plan.end:
            plan.notes.append(f"the forcing stops at {df.index.max().date()}: end year {end}, not {plan.end} (BUG-003)")
            plan.end = end

    plan.spin = measure_spin(os.path.join(path, "CFMforcing.hdf5"), plan.start)
    if plan.spin is None:
        plan.notes.append("the spin-up is not measured: CFMforcing.hdf5 has no spin-up of whole blocks, or cannot be read")

    plan.hash_inputs = migrated_hash_inputs(plan)
    plan.run_hash = run_hash(plan.hash_inputs)
    plan.target = run_folder(plan.model_point.tag, plan.physRho, plan.liquid, plan.start, plan.end, label, plan.run_hash)
    plan.action = "COPY"
    return plan


def group_plans(plans: List[RunPlan]) -> None:
    """Plans with one target: the first is a COPY, the others DUPLICATEs. A target that exists is EXISTS or CONFLICT."""
    by_target = defaultdict(list)
    for plan in plans:
        if plan.action == "COPY":
            by_target[plan.target].append(plan)
    for target, group in by_target.items():
        first = group[0]
        for plan in group[1:]:
            plan.action = "DUPLICATE"
            plan.notes.append(f"same run hash as {first.source}; the results are compared in --copy")
            for site in plan.sites:
                if site not in first.sites:
                    first.sites.append(site)
                    first.site_coords[site] = plan.site_coords[site]
        manifest = read_manifest(target) if os.path.isdir(target) else None
        if os.path.isdir(target):
            if manifest is not None and manifest.get("run_hash") == first.run_hash and manifest.get("status") == COMPLETE:
                for plan in group:
                    if plan.action != "CONFLICT":
                        plan.action = "EXISTS"
            else:
                for plan in group:
                    plan.action = "CONFLICT"
                    plan.notes.append(f"{target} exists with no complete manifest of this hash")


def build_plans(sources: List[tuple], match: Optional[List[str]] = None) -> List[RunPlan]:
    site_by_coords = {(format_coord(s.lat), format_coord(s.lon)): name for name, s in sites(active_only=False).items()}
    plans = []
    for label, cfm_output in sources:
        for path in sorted(glob.glob(os.path.join(cfm_output, "CFMoutput_*"))):
            if not os.path.isdir(path) or (match and not any(text in os.path.basename(path) for text in match)):
                continue
            plan = plan_folder(path, label, site_by_coords)
            if plan is None:
                print(f"IGNORE    {path} (name does not match CFMoutput_<lat>_<lon>_<start>_<end>_<physRho>_<liquid>_<rcm>)")
            else:
                plans.append(plan)
    group_plans(plans)
    return plans


def migrated_manifest(plan: RunPlan) -> dict:
    mp = plan.model_point
    # the wrapper parameters that the old folder gives: the years of its name, and spin_end_year if it is measured
    known = ("start_year", "end_year") + (("spin_end_year",) if plan.spin is not None else ())
    return {
        "status": COMPLETE,
        "run_hash": plan.run_hash,
        "hash_inputs": plan.hash_inputs,
        "migrated": {
            "source": plan.source,
            "source_config_json": plan.config_json,
            "source_forcing_csv": plan.forcing_csv,
            "notes": plan.notes,
            "unknown": ["cfm identity (commit, patch, environment)", "wrapper code"]
            + [key for key in WRAPPER_PARAMS if key not in known]
            + ([] if plan.cfm_config is not None else ["cfm_config"]),
            # the values of measure_spin() other than spin_end_year, which is in run.spin_end_year
            "spin": None
            if plan.spin is None
            else {
                **{key: value for key, value in plan.spin.items() if key != "spin_end_year"},
                "method": "measured from CFMforcing.hdf5",
            },
            "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "script": "scripts/migrate_runs.py",
            "wrapper": wrapper_identity(),
        },
        "run": {
            "sites": plan.sites,
            "site_coords": plan.site_coords,
            "model_point": {
                "tag": mp.tag,
                "rcm_name": mp.rcm_name,
                "release": mp.release,
                "i": mp.i,
                "j": mp.j,
                "cell_lat": mp.cell_lat,
                "cell_lon": mp.cell_lon,
                "distance_km": mp.distance_km,
                "method": mp.method,
            },
            "borehole_lat": plan.lat,
            "borehole_lon": plan.lon,
            "physRho": plan.physRho,
            "liquid": plan.liquid,
            "rcm_name": plan.rcm,
            "start_year": plan.start,
            "end_year": plan.end,
            "spin_end_year": plan.spin_end_year,
            "forcing_csv": plan.forcing_csv,
        },
        "cfm": {"label": plan.label},
        "cfm_config": plan.cfm_config,
        "finished": plan.finished,
    }


def _copy_atomic(source: str, target: str) -> None:
    tmp = f"{target}.tmp{os.getpid()}"
    try:
        shutil.copyfile(source, tmp)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def copy_plan(plan: RunPlan) -> None:
    """Copy one old folder into its run folder. The source is not changed. The manifest is written last."""
    os.makedirs(plan.target, exist_ok=True)
    with open(os.path.join(plan.target, "run.log"), "a") as log:
        log.write(f"{datetime.datetime.now().astimezone().isoformat(timespec='seconds')} migrated from {plan.source}\n")
    for name in plan.files:
        source, target = os.path.join(plan.source, name), os.path.join(plan.target, name)
        _copy_atomic(source, target)
        if sha256_file(target) != sha256_file(source):
            raise RuntimeError(f"{target}: the copy has a different SHA-256 from {source}")
    if plan.config_json:
        _copy_atomic(plan.config_json, os.path.join(plan.target, "CFMconfig.json"))
    write_manifest(plan.target, migrated_manifest(plan))


def check_duplicate(plan: RunPlan, first: RunPlan) -> None:
    """A DUPLICATE must have the same .hdf5 files as the run that is copied."""
    for name in plan.files:
        if sha256_file(os.path.join(plan.source, name)) != sha256_file(os.path.join(first.target, name)):
            raise RuntimeError(f"{plan.source}/{name} is not the same as {first.target}/{name}")


def copy_all(plans: List[RunPlan]) -> int:
    conflicts = [plan for plan in plans if plan.action == "CONFLICT"]
    if conflicts:
        for plan in conflicts:
            print(f"ERROR: CONFLICT {plan.source}: {'; '.join(plan.notes)}", file=sys.stderr)
        print("Nothing is copied.", file=sys.stderr)
        return 1
    size = sum(plan.size for plan in plans if plan.action == "COPY")
    os.makedirs(runs_root(), exist_ok=True)
    free = shutil.disk_usage(runs_root()).free
    if size > free:
        print(f"ERROR: the copy needs {size / 1e9:.1f} GB, but {runs_root()} has {free / 1e9:.1f} GB free.", file=sys.stderr)
        return 1
    first_by_target = {}
    for plan in plans:
        if plan.action == "COPY":
            copy_plan(plan)
            first_by_target[plan.target] = plan
            print(f"COPIED    {plan.source} -> {plan.target}")
    for plan in plans:
        if plan.action == "DUPLICATE":
            check_duplicate(plan, first_by_target[plan.target])
            print(f"SAME      {plan.source} = {plan.target}")
    print("No source file is moved, deleted or changed.")
    return 0


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source", nargs=2, action="append", required=True, metavar=("LABEL", "CFM_OUTPUT"),
        help="A CFM label and an old cfm_output folder. Give --source again for each folder.",
    )
    parser.add_argument("--match", nargs="+", help="Only the folders with one of these texts in the name")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Only print the plan (the default)")
    mode.add_argument("--copy", action="store_true", help="Copy the folders with the action COPY into the runs folder")
    args = parser.parse_args(argv)

    sources = [(label, os.path.expanduser(path)) for label, path in args.source]
    plans = build_plans(sources, args.match)
    print(f"COPY MODE into {runs_root()}\n" if args.copy else f"DRY RUN: no file is copied or written. Target {runs_root()}\n")
    for plan in plans:
        print(f"{plan.action:9s} {plan.source}")
        if plan.model_point is not None:
            mp = plan.model_point
            print(f"          site {plan.site}, {mp.tag} ({mp.distance_km:.1f} km), {plan.size / 1e9:.2f} GB, finished {plan.finished}")
            spin = plan.spin
            if spin is not None:
                print(f"          spin_end_year {spin['spin_end_year']} ({spin['blocks']} blocks of {spin['block_steps']} steps from {spin['spin_start']:g})")
            else:
                print("          spin_end_year None")
        if plan.target:
            print(f"          -> {plan.target}")
        for note in plan.notes:
            print(f"          {note}")
    counts = defaultdict(int)
    for plan in plans:
        counts[plan.action] += 1
    size = sum(plan.size for plan in plans if plan.action == "COPY")
    print("\nSummary: " + ", ".join(f"{action} {n}" for action, n in sorted(counts.items())) + f"; {size / 1e9:.1f} GB to copy")
    by_spin = defaultdict(lambda: defaultdict(int))  # {label: {spin_end_year: COPY folders}}
    for plan in plans:
        if plan.action == "COPY":
            by_spin[plan.label][plan.spin_end_year] += 1
    for label, counts_of_label in sorted(by_spin.items()):
        print(f"COPY of {label} by spin_end_year: " + ", ".join(
            f"{year} {n}" for year, n in sorted(counts_of_label.items(), key=lambda item: (item[0] is None, item[0] or 0))
        ))
    return copy_all(plans) if args.copy else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
