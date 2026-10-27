"""Find CFM runs with a query, not with a folder name (FEAT-001).

find_runs() reads the manifest.json of each run folder under runs/ (see sim.manifest for the layout) and returns the
runs that agree with the filters. find_run() returns one run. The site filter uses the site table (sites.yaml) and
the model point of the site, as two sites in one grid cell share a run (FEAT-002).

find_runs() also compares the coordinates of each site in the manifests of the runs that it returns (run.site_coords;
in a manifest from before that key, run.borehole_lat and run.borehole_lon for the first site) with sites.yaml, to 3
decimals (format_coord). The name and the coordinates of a site do not change (FEAT-002), so a difference, or a site
that is not in sites.yaml, gives a warning: once for each run folder and site in each process. The run is still
returned. The run hash does not contain the site, so the results are correct for the model point; the warning is about
the provenance of the site name (DISC-002).

With CFM_share_data_path (FEAT-008), find_runs() reads the runs in each root of sim.manifest.run_roots(): the local
runs, and the copy of the runs on the share. A run that agrees with the filters in the two roots is returned once, as
the local copy with location "both". Run.folder, and each path made from it, is the folder of the copy returned.

index.csv is only an export of runs_table() for the user, made on request (scripts/find_runs.py --index). The code
never reads it, so it cannot be out of date for a query.

The plot and analysis scripts find their runs with the spin_end_year of config.yaml by default, as a different spin-up
gives different results (FEAT-004). add_spin_end_year_argument() gives them --spin-end-year <year> or any, and
spin_end_year_filter() the value for find_run.
"""

import argparse
import glob
import logging
import os
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from cfm_config import config, format_coord, sites as site_table
from force.grid import great_circle_km
from sim.manifest import COMPLETE, MANIFEST_NAME, read_manifest, run_roots, runs_root, share_runs_root

# the (run folder, site) pairs that had a site warning in this process, so that each pair gives one warning only
_warned_sites: Set[Tuple[str, str]] = set()

# the values of Run.location (FEAT-008)
LOCAL = "local"
SHARE = "share"
BOTH = "both"

# the meltwater solver of a run whose cfm_config has no meltwater_solver key: the default of the CFM
DEFAULT_MELTWATER_SOLVER = "enthalpy"


@dataclass(frozen=True)
class Run:
    """One run folder and its manifest.

    location is where the run is (FEAT-008): LOCAL (in runs_root()), SHARE (in share_runs_root() only) or BOTH (in
    the two roots). folder is the local copy if there is one, else the share copy. Each path of the run is made from
    folder.
    """

    folder: str
    manifest: dict
    location: str = LOCAL

    @property
    def run_hash(self) -> str:
        return self.manifest.get("run_hash", "")

    @property
    def status(self) -> str:
        return self.manifest.get("status", "")

    @property
    def model_point(self) -> str:
        """ModelPoint.tag of the run, for example RACMO_261_126."""
        return self.manifest["run"]["model_point"]["tag"]

    @property
    def rcm_name(self) -> str:
        return self.manifest["run"]["rcm_name"]

    @property
    def physRho(self) -> str:
        return self.manifest["run"]["physRho"]

    @property
    def liquid(self) -> str:
        return self.manifest["run"]["liquid"]

    @property
    def meltwater_solver(self) -> str:
        """The meltwater_solver of the cfm_config of the run. A run without the key used the default of the CFM,
        DEFAULT_MELTWATER_SOLVER: a run that did not set it, a run of a CFM before 3.2.0 (3e3bf7e has no such key) and
        a migrated run with no cfm_config in its manifest."""
        return (self.manifest.get("cfm_config") or {}).get("meltwater_solver", DEFAULT_MELTWATER_SOLVER)

    @property
    def start_year(self) -> int:
        return self.manifest["run"]["start_year"]

    @property
    def end_year(self) -> int:
        return self.manifest["run"]["end_year"]

    @property
    def spin_end_year(self) -> Optional[int]:
        """The last year of the spin-up block. A migrated manifest from before FEAT-004 (migrated-1) does not have it."""
        return self.manifest["run"].get("spin_end_year")

    @property
    def cfm(self) -> str:
        """The label of the CFM version."""
        return self.manifest["cfm"]["label"]

    @property
    def cfm_commit(self) -> str:
        return self.manifest["cfm"].get("commit", "")

    @property
    def sites(self) -> List[str]:
        return list(self.manifest["run"].get("sites", []))

    @property
    def site_coords(self) -> Dict[str, Tuple[float, float]]:
        """The (lat, lon) of each site in the manifest. A manifest from before run.site_coords has the coordinates of
        its first site only, in run.borehole_lat and run.borehole_lon."""
        run = self.manifest["run"]
        coords = {site: (lat, lon) for site, (lat, lon) in (run.get("site_coords") or {}).items()}
        first = self.sites[0] if self.sites else None
        if first is not None and first not in coords and None not in (run.get("borehole_lat"), run.get("borehole_lon")):
            coords[first] = (run["borehole_lat"], run["borehole_lon"])
        return coords

    def distance_km(self, site: str) -> Optional[float]:
        """The distance in km from a site of the run to the centre of its cell, from run.site_coords and
        run.model_point. The manifest records distance_km for the first site only; this gives it for each site
        (FEAT-006, FEAT-007). None if the manifest has no coordinates for the site or no cell centre."""
        cell = self.manifest["run"]["model_point"]
        coords = self.site_coords.get(site)
        if coords is None or cell.get("cell_lat") is None or cell.get("cell_lon") is None:
            return None
        return float(great_circle_km(coords[0], coords[1], cell["cell_lat"], cell["cell_lon"]))

    @property
    def finished(self) -> str:
        return self.manifest.get("finished") or ""

    @property
    def results_file(self) -> str:
        """The CFMresults.hdf5 file of the run."""
        return os.path.join(self.folder, self.manifest.get("cfm_config", {}).get("resultsFileName", "CFMresults.hdf5"))

    def row(self) -> dict:
        """One row of runs_table()."""
        return {
            "model_point": self.model_point,
            "rcm_name": self.rcm_name,
            "physRho": self.physRho,
            "liquid": self.liquid,
            "meltwater_solver": self.meltwater_solver,
            "start_year": self.start_year,
            "end_year": self.end_year,
            "spin_end_year": self.spin_end_year,
            "cfm": self.cfm,
            "cfm_commit": self.cfm_commit,
            "run_hash": self.run_hash,
            "status": self.status,
            "sites": ",".join(self.sites),
            "started": self.manifest.get("started", ""),
            "finished": self.finished,
            "folder": self.folder,
        }


def _site_model_point(site: str, rcm_name: str) -> str:
    from force.model_point import model_point  # reads the RCM grid; imported here, as most queries do not need it

    return model_point(site, rcm_name).tag


def find_runs(
    site: Optional[str] = None,
    model_point: Optional[str] = None,
    rcm_name: Optional[str] = None,
    physRho: Optional[str] = None,
    liquid: Optional[str] = None,
    meltwater_solver: Optional[str] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    spin_end_year: Optional[int] = None,
    cfm: Optional[str] = None,
    run_hash: Optional[str] = None,
    status: Optional[str] = COMPLETE,
    root: Optional[str] = None,
) -> List[Run]:
    """The runs that agree with all the filters that are not None, sorted by folder.

    site is a name in sites.yaml: the runs of its model point. With site and no rcm_name, the model point of the site is
    found for each RCM that has runs. model_point is a ModelPoint.tag. meltwater_solver is run.meltwater_solver: a run
    without the key agrees with DEFAULT_MELTWATER_SOLVER. spin_end_year is run.spin_end_year of the
    manifest: a run with no spin_end_year (a migrated-1 run, FEAT-004) never agrees with it. cfm is a label in
    cfm_versions. run_hash can be the first characters of the hash. status is "complete" by default; None gives all the
    runs.

    root is one run root. If None, the runs in each root of run_roots() (FEAT-008), sorted by folder: the local runs,
    and the share runs. A run hash that agrees with the filters in the two roots gives the local run only, with
    location BOTH. A run that agrees in the share root only (for example, its local copy is not complete) gives the
    share run.
    """
    filters = dict(
        site=site,
        model_point=model_point,
        rcm_name=rcm_name,
        physRho=physRho,
        liquid=liquid,
        meltwater_solver=meltwater_solver,
        start_year=start_year,
        end_year=end_year,
        spin_end_year=spin_end_year,
        cfm=cfm,
        run_hash=run_hash,
        status=status,
    )
    site_tags: dict = {}  # model point of the site, for each RCM, for all the roots
    if root:
        runs = _find_in_root(root, SHARE if root == share_runs_root() else LOCAL, site_tags, **filters)
    else:
        runs = []
        local = {}  # the index in runs of each local run hash
        for one_root in run_roots():
            if one_root == runs_root():
                found = _find_in_root(one_root, LOCAL, site_tags, **filters)
                local.update((run.run_hash, len(runs) + i) for i, run in enumerate(found))
                runs.extend(found)
                continue
            for run in _find_in_root(one_root, SHARE, site_tags, **filters):
                if run.run_hash in local:
                    runs[local[run.run_hash]] = replace(runs[local[run.run_hash]], location=BOTH)
                else:
                    runs.append(run)
        runs.sort(key=lambda run: run.folder)
    if runs:
        _check_sites(runs)
    return runs


def _find_in_root(
    root: str,
    location: str,
    site_tags: dict,
    site: Optional[str],
    model_point: Optional[str],
    rcm_name: Optional[str],
    physRho: Optional[str],
    liquid: Optional[str],
    meltwater_solver: Optional[str],
    start_year: Optional[int],
    end_year: Optional[int],
    spin_end_year: Optional[int],
    cfm: Optional[str],
    run_hash: Optional[str],
    status: Optional[str],
) -> List[Run]:
    """The runs in one run root that agree with the filters of find_runs, sorted by folder, with this location.
    site_tags is the model point of the site for each RCM, filled in as the RCMs are found."""
    if model_point is not None:
        pattern = os.path.join(root, model_point, "*", "*", "*", MANIFEST_NAME)
    elif rcm_name is not None:
        pattern = os.path.join(root, f"{rcm_name}_*_*", "*", "*", "*", MANIFEST_NAME)
    else:
        pattern = os.path.join(root, "*", "*", "*", "*", MANIFEST_NAME)

    runs = []
    for path in sorted(glob.glob(pattern)):
        folder = os.path.dirname(path)
        manifest = read_manifest(folder)
        if not manifest or "run_hash" not in manifest:
            logging.warning(f"{folder} has no run hash in its manifest. It is not a run of sim.run, so it is ignored.")
            continue
        run = Run(folder, manifest, location)
        if status is not None and run.status != status:
            continue
        if model_point is not None and run.model_point != model_point:
            continue
        if rcm_name is not None and run.rcm_name != rcm_name:
            continue
        if site is not None:
            if run.rcm_name not in site_tags:
                site_tags[run.rcm_name] = _site_model_point(site, run.rcm_name)
            if run.model_point != site_tags[run.rcm_name]:
                continue
        wanted = {
            "physRho": physRho,
            "liquid": liquid,
            "meltwater_solver": meltwater_solver,
            "start_year": start_year,
            "end_year": end_year,
            "spin_end_year": spin_end_year,
            "cfm": cfm,
        }
        if any(value is not None and getattr(run, key) != value for key, value in wanted.items()):
            continue
        if run_hash is not None and not run.run_hash.startswith(run_hash):
            continue
        runs.append(run)
    return runs


def _check_sites(runs: List[Run]) -> None:
    """Warn if a site of a run has different coordinates in sites.yaml, or is not in sites.yaml (see the docstring)."""
    table = site_table(active_only=False)
    for run in runs:
        coords = run.site_coords
        for name in run.sites:
            if (run.folder, name) in _warned_sites:
                continue
            site = table.get(name)
            if site is None:
                message = (
                    f"{run.folder}: site {name} is not in sites.yaml. The run is possibly from a different checkout. "
                    "The run hash does not depend on the site, so the results are correct for the model point of the "
                    "run, but the location of this site name is not known (FEAT-002)."
                )
            elif name in coords and _differ(coords[name], (site.lat, site.lon)):
                lat, lon = coords[name]
                message = (
                    f"{run.folder}: the manifest has site {name} at lat {lat}, lon {lon}, but sites.yaml has lat "
                    f"{site.lat}, lon {site.lon}. The run hash does not depend on the site coordinates, so the results "
                    "are correct for the model point of the run, but the site name had different coordinates when the "
                    "run was made. The name and the coordinates of a site do not change (FEAT-002)."
                )
            else:
                continue
            _warned_sites.add((run.folder, name))
            logging.warning(message)


def _differ(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """Whether two (lat, lon) differ to 3 decimals, the precision of sites.yaml, sites.lock and the site matching of
    scripts/migrate_runs.py."""
    return tuple(map(format_coord, a)) != tuple(map(format_coord, b))


def find_run(**filters) -> Run:
    """The one run that agrees with the filters of find_runs().

    If more than one run agrees (for example, runs of one CFM version before and after a change to src/), this gives
    the run that finished last, with a warning that lists the others. Give run_hash to choose a different run. Stops
    with LookupError if no run agrees.
    """
    runs = find_runs(**filters)
    given = ", ".join(f"{key}={value}" for key, value in filters.items() if value is not None)
    if not runs:
        spin = filters.get("spin_end_year")
        hint = (
            f" Only the runs with spin_end_year {spin} agree; the plot and analysis scripts use spin_end_year of "
            "config.yaml if --spin-end-year is not given (FEAT-004)."
            if spin is not None
            else ""
        )
        roots = filters.get("root") or " or ".join(run_roots()) or runs_root()
        raise LookupError(f"No run with {given} in {roots}.{hint} Run scripts/run_cfm.py first.")
    runs.sort(key=lambda run: run.finished)
    if len(runs) > 1:
        def describe(run):
            return (
                f"{run.run_hash[:8]} ({run.cfm}, {run.meltwater_solver}, spin_end_year {run.spin_end_year}, "
                f"finished {run.finished})"
            )

        others = ", ".join(describe(run) for run in runs[:-1])
        logging.warning(
            f"{len(runs)} runs with {given}. Using {describe(runs[-1])}, the last to finish. The others: {others}. "
            "Give run_hash, meltwater_solver or spin_end_year to choose one."
        )
    return runs[-1]


ANY = "any"  # --spin-end-year any: no spin_end_year filter


def _spin_end_year_value(text: str) -> Union[int, str]:
    if text.lower() == ANY:
        return ANY
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"give a year or {ANY}, not {text!r}")


def add_spin_end_year_argument(parser: argparse.ArgumentParser) -> None:
    """Add --spin-end-year: the spin_end_year of the runs to find (see spin_end_year_filter)."""
    parser.add_argument(
        "--spin-end-year",
        type=_spin_end_year_value,
        help=f"only the runs with this spin_end_year, or {ANY} for all the runs (default: spin_end_year in config.yaml)",
    )


def spin_end_year_filter(value: Union[int, str, None] = None) -> Optional[int]:
    """The spin_end_year filter of find_run for the value of --spin-end-year: spin_end_year of config.yaml if None,
    no filter (None) if "any", else the year."""
    if value is None:
        return int(config["spin_end_year"])
    if isinstance(value, str) and value.lower() == ANY:
        return None
    return int(value)


ROW_KEYS = (
    "model_point",
    "rcm_name",
    "physRho",
    "liquid",
    "meltwater_solver",
    "start_year",
    "end_year",
    "spin_end_year",
    "cfm",
    "cfm_commit",
    "run_hash",
    "status",
    "sites",
    "started",
    "finished",
    "folder",
)


def runs_table(runs: List[Run]) -> pd.DataFrame:
    """A table of runs, one row for each run, with the columns ROW_KEYS."""
    return pd.DataFrame([run.row() for run in runs], columns=list(ROW_KEYS))
