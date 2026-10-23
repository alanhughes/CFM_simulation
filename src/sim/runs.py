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

index.csv is only an export of runs_table() for the user, made on request (scripts/find_runs.py --index). The code
never reads it, so it cannot be out of date for a query.
"""

import glob
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from cfm_config import format_coord, sites as site_table
from sim.manifest import COMPLETE, MANIFEST_NAME, read_manifest, runs_root

# the (run folder, site) pairs that had a site warning in this process, so that each pair gives one warning only
_warned_sites: Set[Tuple[str, str]] = set()


@dataclass(frozen=True)
class Run:
    """One run folder and its manifest."""

    folder: str
    manifest: dict

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
    def start_year(self) -> int:
        return self.manifest["run"]["start_year"]

    @property
    def end_year(self) -> int:
        return self.manifest["run"]["end_year"]

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
            "start_year": self.start_year,
            "end_year": self.end_year,
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
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    cfm: Optional[str] = None,
    run_hash: Optional[str] = None,
    status: Optional[str] = COMPLETE,
    root: Optional[str] = None,
) -> List[Run]:
    """The runs that agree with all the filters that are not None, sorted by folder.

    site is a name in sites.yaml: the runs of its model point. With site and no rcm_name, the model point of the site is
    found for each RCM that has runs. model_point is a ModelPoint.tag. cfm is a label in cfm_versions. run_hash can be
    the first characters of the hash. status is "complete" by default; None gives all the runs.
    """
    root = root or runs_root()
    if model_point is not None:
        pattern = os.path.join(root, model_point, "*", "*", "*", MANIFEST_NAME)
    elif rcm_name is not None:
        pattern = os.path.join(root, f"{rcm_name}_*_*", "*", "*", "*", MANIFEST_NAME)
    else:
        pattern = os.path.join(root, "*", "*", "*", "*", MANIFEST_NAME)

    site_tags: dict = {}  # model point of the site, for each RCM
    runs = []
    for path in sorted(glob.glob(pattern)):
        folder = os.path.dirname(path)
        manifest = read_manifest(folder)
        if not manifest or "run_hash" not in manifest:
            logging.warning(f"{folder} has no run hash in its manifest. It is not a run of sim.run, so it is ignored.")
            continue
        run = Run(folder, manifest)
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
        wanted = {"physRho": physRho, "liquid": liquid, "start_year": start_year, "end_year": end_year, "cfm": cfm}
        if any(value is not None and getattr(run, key) != value for key, value in wanted.items()):
            continue
        if run_hash is not None and not run.run_hash.startswith(run_hash):
            continue
        runs.append(run)
    if runs:
        _check_sites(runs)
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
        raise LookupError(f"No run with {given} in {filters.get('root') or runs_root()}. Run scripts/run_cfm.py first.")
    runs.sort(key=lambda run: run.finished)
    if len(runs) > 1:
        others = ", ".join(f"{run.run_hash[:8]} ({run.cfm}, finished {run.finished})" for run in runs[:-1])
        logging.warning(
            f"{len(runs)} runs with {given}. Using {runs[-1].run_hash[:8]} ({runs[-1].cfm}, finished "
            f"{runs[-1].finished}), the last to finish. The others: {others}. Give run_hash to choose one."
        )
    return runs[-1]


ROW_KEYS = (
    "model_point",
    "rcm_name",
    "physRho",
    "liquid",
    "start_year",
    "end_year",
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
