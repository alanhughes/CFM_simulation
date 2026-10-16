"""Model points: the RCM grid cell that gives the forcing data for a site (FEAT-002).

A site is a location with in situ measurements (see sites.yaml). A model point is the RCM grid cell whose data forces
the CFM. Two sites in one grid cell share a model point, so they share the forcing data and the CFM run.

The only method is "nearest": the cell whose centre has the smallest great circle distance to the site (BUG-002).
"""

import functools
import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr

from cfm_config import Site, config, get_site

from .grid import nearest_cell

# For each RCM: the config key of its data path, the names of its 2D latitude/longitude arrays, and a regex that
# splits a file name into its variable/year parts and the release. The release is the part of the file name that
# names the dataset, so a new data release in a new folder gives a new forcing cache folder.
_RCM_GRIDS = {
    "MAR": {
        "path_key": "MAR_data_path",
        "lat": "LAT",
        "lon": "LON",
        # MARv3.14.3-27.5km-daily-ERA5-1979.nc
        "file_regex": r"^(?P<release>.+)-(?P<year>\d{4})\.nc$",
    },
    "RACMO": {
        "path_key": "RACMO_data_path",
        "lat": "lat",
        "lon": "lon",
        # ts_ANT-12_ERA5_evaluation_r1i1p1f1_UU-IMAU_RACMO24P-NN_v1-r1_day_19790101-19801231.nc
        "file_regex": r"^(?P<var>[a-z]+)_(?P<release>.+)_(?P<year>\d{4})\d{4}-\d{8}\.nc$",
    },
    "RACMO23": {
        "path_key": "RACMO23_data_path",
        "lat": "lat",
        "lon": "lon",
        # ts.KNMI-1979.XPEN055.RACMO2.4p1_v5_historical.DD.nc
        "file_regex": r"^(?P<var>[a-z]+)\.KNMI-(?P<year>\d{4})\.(?P<release>.+)\.nc$",
    },
}


@dataclass(frozen=True)
class ModelPoint:
    """The RCM grid cell that gives the forcing data for a location.

    i is the row index (rlat / Y) and j is the column index (rlon / X) of the 2D latitude/longitude arrays.
    """

    rcm_name: str
    release: str
    i: int
    j: int
    cell_lat: float
    cell_lon: float
    distance_km: float
    method: str = "nearest"

    @property
    def tag(self) -> str:
        """`<rcm>_<i>_<j>`: names the model point in run folders (FEAT-001) and in plots."""
        return model_point_tag(self.rcm_name, self.i, self.j)


def model_point_tag(rcm_name: str, i: int, j: int) -> str:
    """`<rcm>_<i>_<j>` for one model point. The release is not in the tag; it is a folder above it in the cache."""
    return f"{rcm_name}_{i}_{j}"


def rcm_data_path(rcm_name: str) -> str:
    try:
        return config[_RCM_GRIDS[rcm_name]["path_key"]]
    except KeyError:
        raise ValueError(f"Unknown rcm_name {rcm_name!r}. Choose one of {sorted(_RCM_GRIDS)}.") from None


@functools.lru_cache(maxsize=None)
def _release_and_files(rcm_name: str, data_path: str) -> Tuple[str, Tuple[str, ...]]:
    """Release name and the sorted RCM data file names in data_path. Lists the folder only; opens no file."""
    spec = _RCM_GRIDS[rcm_name]
    pattern = re.compile(spec["file_regex"])
    matches = sorted((m for m in map(pattern.match, os.listdir(data_path)) if m), key=lambda m: m.string)
    if not matches:
        raise FileNotFoundError(f"No {rcm_name} files that match {spec['file_regex']} in {data_path}")
    releases = {m["release"] for m in matches}
    if len(releases) != 1:
        raise ValueError(f"More than one {rcm_name} release in {data_path}: {sorted(releases)}")
    return releases.pop(), tuple(m.string for m in matches)


def rcm_release(rcm_name: str, data_path: Optional[str] = None) -> str:
    """The release name of an RCM dataset, from the names of its files. Lists the folder only; reads no data.

    data_path defaults to the configured data path of rcm_name. The forcing cache uses the release as a folder name
    (forcing/<rcm_name>/<release>/), so a new data release gives new forcing data.
    """
    return _release_and_files(rcm_name, data_path or rcm_data_path(rcm_name))[0]


@functools.lru_cache(maxsize=None)
def _grid(rcm_name: str, data_path: str) -> Tuple[str, str, np.ndarray, np.ndarray]:
    """Release name, coordinate file, and the 2D latitude/longitude arrays of one RCM dataset.

    Opens one file only. The result is cached for the life of the process.
    """
    spec = _RCM_GRIDS[rcm_name]
    release, files = _release_and_files(rcm_name, data_path)
    coord_file = os.path.join(data_path, files[0])
    with xr.open_dataset(coord_file) as ds:
        lat2d = np.asarray(ds[spec["lat"]].values, dtype=float)
        lon2d = np.asarray(ds[spec["lon"]].values, dtype=float)
    if lat2d.ndim != 2:
        raise ValueError(f"Expected 2D {spec['lat']} in {coord_file}, got shape {lat2d.shape}")
    return release, coord_file, lat2d, lon2d


def clear_cache() -> None:
    """Forget the cached file listings and coordinate arrays (for tests, or after the data path changes)."""
    _release_and_files.cache_clear()
    _grid.cache_clear()


def model_point_at(lat: float, lon: float, rcm_name: str) -> ModelPoint:
    """The model point for a location. Reads only the coordinate arrays of one RCM file."""
    release, _, lat2d, lon2d = _grid(rcm_name, rcm_data_path(rcm_name))
    i, j, distance = nearest_cell(lat2d, lon2d, lat, lon)
    return ModelPoint(
        rcm_name=rcm_name,
        release=release,
        i=i,
        j=j,
        cell_lat=float(lat2d[i, j]),
        cell_lon=float(lon2d[i, j]),
        distance_km=distance,
    )


def model_point(site: Union[str, Site], rcm_name: str) -> ModelPoint:
    """The model point of a site in sites.yaml (by name, or a Site)."""
    if isinstance(site, str):
        site = get_site(site)
    return model_point_at(site.lat, site.lon, rcm_name)


def shared_model_points(sites, rcm_name: str) -> dict:
    """Group sites by model point. Returns {tag: [(site name, ModelPoint), ...]} for every model point."""
    groups: dict = {}
    for site in sites.values() if isinstance(sites, dict) else sites:
        mp = model_point(site, rcm_name)
        groups.setdefault(mp.tag, []).append((site.name, mp))
    return groups


def model_point_label(site_name: str, groups: dict) -> str:
    """A short label for plots, for example "J409: RACMO23_101_222, 3.8 km from the cell centre, same cell as J109"."""
    for members in groups.values():
        names = [name for name, _ in members]
        if site_name in names:
            mp = dict(members)[site_name]
            label = f"{site_name}: {mp.tag}, {mp.distance_km:.1f} km from the cell centre"
            others = [name for name in names if name != site_name]
            if others:
                label += f", same cell as {', '.join(others)}"
            return label
    raise KeyError(f"{site_name} is not in the model point groups")
