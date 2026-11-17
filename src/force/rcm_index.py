"""RCM indexes: a small stand-in for an RCM data folder, for a machine with no RCM data (FEAT-012).

A run needs the RCM data folder (RACMO_data_path, ...) for metadata only, before it reads the forcing CSV:

- the release, from the file names (model_point._release_and_files);
- the 2D latitude/longitude of the grid, from one file, to find the model point (model_point._grid);
- the first and last days of the data, from the file names, for the years of the run (years.data_days).

An RCM index holds these three things, so a machine without the RCM data (the HEC) can find the same model point,
release and years as a machine with the data, and then read the forcing CSV from the forcing cache:

    <index folder>/rcm_index.json    the release, the names of the files in the data folder, the coordinate file
    <index folder>/grid.npz          lat and lon, the 2D arrays that model_point._grid read from the coordinate file

rcm_index.json also holds the SHA-256 of the grid values (grid_sha256). The grid is checked against it when it is read,
so a grid file that is damaged, or from a different index, stops the run.

scripts/make_rcm_index.py writes an index from a data folder. To use it, set the data path key of the RCM (for example
RACMO_data_path) to the index folder. The code then reads the index in place of the folder. An index has no data, so
the forcing cannot be made from it: force.extract.process_fast stops with an error if the forcing CSV is not in the
cache.
"""

import datetime
import fnmatch
import functools
import hashlib
import json
import os
import socket
from typing import List, Optional, Sequence, Tuple

import numpy as np

INDEX_FILE = "rcm_index.json"
GRID_FILE = "grid.npz"
INDEX_VERSION = 1


def is_rcm_index(data_path: str) -> bool:
    """Whether data_path is an RCM index folder (it has rcm_index.json), not an RCM data folder."""
    return os.path.isfile(os.path.join(data_path, INDEX_FILE))


@functools.lru_cache(maxsize=None)
def read_index(data_path: str) -> dict:
    """The rcm_index.json record of an index folder. Cached for the life of the process."""
    path = os.path.join(data_path, INDEX_FILE)
    with open(path) as file:
        record = json.load(file)
    if record.get("index_version") != INDEX_VERSION:
        raise ValueError(f"{path} has index_version {record.get('index_version')!r}, not {INDEX_VERSION}")
    return record


def data_file_names(data_path: str) -> List[str]:
    """The names of the files in an RCM data folder (os.listdir), or the names that an index folder recorded."""
    if is_rcm_index(data_path):
        return list(read_index(data_path)["files"])
    return os.listdir(data_path)


def match_data_files(data_path: str, pattern: str) -> List[str]:
    """The paths in data_path whose names match a glob pattern (no folder part), as glob.glob would give them.

    For an index folder, the names are the recorded names, and the paths are in the index folder (they do not exist).
    As glob.glob, a name that starts with "." matches only a pattern that starts with ".", and a folder that cannot be
    listed gives no paths.
    """
    try:
        names = data_file_names(data_path)
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return []
    if not pattern.startswith("."):
        names = [name for name in names if not name.startswith(".")]
    return [os.path.join(data_path, name) for name in fnmatch.filter(names, pattern)]


def grid_sha256(lat2d: np.ndarray, lon2d: np.ndarray) -> str:
    """The SHA-256 of the float64 values of the grid, to check grid.npz against rcm_index.json."""
    digest = hashlib.sha256()
    for array in (lat2d, lon2d):
        digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return digest.hexdigest()


def read_index_grid(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """The 2D latitude and longitude arrays of an index folder. Stops if the grid file is missing, or if its values do
    not agree with grid_sha256 in rcm_index.json."""
    path = os.path.join(data_path, GRID_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"The RCM index {data_path} has no {GRID_FILE}. Copy it with {INDEX_FILE}.")
    with np.load(path) as grid:
        lat2d, lon2d = grid["lat"], grid["lon"]
    if grid_sha256(lat2d, lon2d) != read_index(data_path)["grid_sha256"]:
        raise ValueError(f"{path} does not agree with grid_sha256 in {INDEX_FILE}. Copy the index folder again.")
    return lat2d, lon2d


def write_rcm_index(
    out: str,
    rcm_name: str,
    release: str,
    data_path: str,
    files: Sequence[str],
    coord_file: str,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    extra: Optional[dict] = None,
) -> str:
    """Write an index folder. files are all the names in data_path; coord_file is the name that gave lat2d and lon2d.

    Stops if out exists. The files go to <out>.tmp first, and the folder is renamed to out at the end, so a stopped
    write leaves no folder that looks complete. Returns out.
    """
    if os.path.exists(out):
        raise FileExistsError(f"{out} exists. Remove it to write a new index.")
    tmp = f"{out}.tmp"
    if os.path.exists(tmp):
        raise FileExistsError(f"{tmp} exists, from a write that stopped. Remove it.")
    os.makedirs(tmp)
    np.savez_compressed(os.path.join(tmp, GRID_FILE), lat=lat2d, lon=lon2d)
    record = {
        "index_version": INDEX_VERSION,
        "rcm_name": rcm_name,
        "release": release,
        "data_path": data_path,
        "coord_file": coord_file,
        "grid_shape": list(lat2d.shape),
        "grid_sha256": grid_sha256(lat2d, lon2d),
        "files": sorted(files),
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
    }
    if extra:
        record.update(extra)
    with open(os.path.join(tmp, INDEX_FILE), "w") as file:
        json.dump(record, file, indent=2)
        file.write("\n")
    os.rename(tmp, out)
    return out
