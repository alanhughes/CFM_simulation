"""The forcing cache: one forcing CSV for each model point (FEAT-002).

    <forcing_data_path>/<rcm_name>/<release>/<i>_<j>_<start>_<end>.csv
    <forcing_data_path>/<rcm_name>/<release>/<i>_<j>_<start>_<end>.json

forcing_data_path defaults to {CFM_data_path}/forcing. The forcing data does not depend on the CFM version, so all the
CFM versions use this one folder. The JSON file records where the CSV came from (see write_forcing_json).

forcing_csv_path() is the only place that builds the local forcing CSV path. It is also the path for a write.
find_forcing_csv() is for a read: it gives the share copy, {CFM_share_data_path}/forcing/..., when the CSV is not
local and the share has it (FEAT-008).
"""

import datetime
import hashlib
import json
import os
import socket
from typing import Dict, Iterable, List, Optional

import pandas as pd

from cfm_config import config, find_on_share, share_data_path

from .model_point import ModelPoint, rcm_data_path


class ForcingCoverageError(ValueError):
    """The forcing data does not include all the years from start_year to end_year (BUG-003)."""


def forcing_root() -> str:
    """The root folder of the forcing cache."""
    root = config.get("forcing_data_path") or "{CFM_data_path}/forcing"
    return root.replace("{CFM_data_path}", config["CFM_data_path"])


def forcing_csv_path(mp: ModelPoint, start_year: Optional[int] = None, end_year: Optional[int] = None) -> str:
    """Path of the forcing CSV of one model point and year range (default: start_year/end_year of config.yaml)."""
    start_year = config["start_year"] if start_year is None else start_year
    end_year = config["end_year"] if end_year is None else end_year
    return os.path.join(forcing_root(), mp.rcm_name, mp.release, f"{mp.i}_{mp.j}_{start_year}_{end_year}.csv")


def find_forcing_csv(mp: ModelPoint, start_year: Optional[int] = None, end_year: Optional[int] = None) -> str:
    """Path of the forcing CSV for a read: the local file if it exists, else the share copy if it exists, else the
    local path (the caller then gives its usual error). Do not use for a write: use forcing_csv_path."""
    local = forcing_csv_path(mp, start_year, end_year)
    share = share_forcing_path(local)
    return local if share is None else find_on_share(local, share)


def share_forcing_path(local: str) -> Optional[str]:
    """The path on the share of a file in the forcing cache: {CFM_share_data_path}/forcing/<path relative to
    forcing_root()>. None if CFM_share_data_path is not set."""
    share = share_data_path()
    return None if share is None else os.path.join(share, "forcing", os.path.relpath(local, forcing_root()))


def forcing_json_path(csv_path: str) -> str:
    """Path of the JSON file next to a forcing CSV."""
    return os.path.splitext(csv_path)[0] + ".json"


def check_date_coverage(dates: Iterable, start_year: int, end_year: int, dataset: str) -> None:
    """Stop with ForcingCoverageError if the dates do not include all the years from start_year to end_year.

    The data must start in January of start_year and stop in December of end_year, with data in each year between.
    """
    index = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    if len(index) == 0:
        raise ForcingCoverageError(f"{dataset}: no data for {start_year}-{end_year}")
    years = set(index.year)
    missing = [year for year in range(start_year, end_year + 1) if year not in years]
    first, last = index.min(), index.max()
    problems = []
    if missing:
        problems.append(f"no data for the years {_year_ranges(missing)}")
    if (first.year, first.month) > (start_year, 1):
        problems.append(f"the data starts at {first.date()}, after January {start_year}")
    if (last.year, last.month) < (end_year, 12):
        problems.append(f"the data stops at {last.date()}, before December {end_year}")
    if problems:
        raise ForcingCoverageError(
            f"{dataset}: {'; '.join(problems)}. start_year-end_year in config.yaml is {start_year}-{end_year}. "
            "Change start_year/end_year, or use a dataset that includes these years."
        )


def _year_ranges(years: List[int]) -> str:
    """[1990, 2023, 2024] -> "1990, 2023-2024"."""
    groups: List[List[int]] = []
    for year in years:
        if groups and year == groups[-1][-1] + 1:
            groups[-1].append(year)
        else:
            groups.append([year])
    return ", ".join(str(g[0]) if len(g) == 1 else f"{g[0]}-{g[-1]}" for g in groups)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_forcing_json(
    csv_path: str,
    mp: ModelPoint,
    df: pd.DataFrame,
    source_files: Iterable[str],
    column_map: Dict[str, str],
    unit_factors: Dict[str, str],
    start_year: int,
    end_year: int,
    site_lat: Optional[float] = None,
    site_lon: Optional[float] = None,
    extra: Optional[dict] = None,
) -> str:
    """Write the JSON file that records where a forcing CSV came from. Returns its path."""
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
        # the location that was used to find the model point; other sites in the same cell share this file
        "requested_location": {"lat": site_lat, "lon": site_lon, "distance_km": mp.distance_km},
        "start_year": start_year,
        "end_year": end_year,
        "first_date": str(pd.Timestamp(df.index.min())),
        "last_date": str(pd.Timestamp(df.index.max())),
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "column_map": dict(column_map),
        "unit_factors": dict(unit_factors),
        "data_path": rcm_data_path(mp.rcm_name),
        "source_files": sorted({os.path.basename(path) for path in source_files}),
        "csv_sha256": sha256_file(csv_path),
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
    }
    if extra:
        record.update(extra)
    json_path = forcing_json_path(csv_path)
    with open(json_path, "w") as file:
        json.dump(record, file, indent=2)
        file.write("\n")
    return json_path
