"""Fast extraction of RACMO daily forcing at many grid cells in one pass (FEAT-003).

The RACMO files store each day as one compressed HDF5 chunk of the full map, so reading one cell costs as much as
reading the whole file. This module therefore reads each (variable, file) pair once with h5py, in parallel worker
processes, and keeps every requested cell from it. Extra cells cost no extra reads from the share.

h5py does not decode the data as xarray does, so this module does it itself: CF time units and calendar, _FillValue
and missing_value (to NaN), scale_factor and add_offset. The derived columns (rain and albedo), the unit factors and
the column maps follow ProcessRACMO._xr_to_input_dataframe (src/force/process.py) step by step, so the output agrees
with the old CSVs.

A local cache (cache.h5) keeps all cells in a box around the sites, so a new site near the others needs no new pass
over the share. See ensure_cache().
"""

import glob
import json
import logging
import multiprocessing
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd

Cell = Tuple[int, int]  # (y, x) = (rlat index, rlon index)
Box = Tuple[int, int, int, int]  # (y0, y1, x0, x1), end exclusive

DEFAULT_PROCESSES = 8
# kg/m2/s -> mmWE/day. A list is applied one factor at a time, in float32: [60, 60, 24] gives the same values as the
# old code (x * 60 * 60 * 24), which differ from x * 86400 by up to 1 unit in the last place.
DEFAULT_UNIT_FACTORS = {"SUBLIM": [60, 60, 24], "RAIN": [60, 60, 24], "BDOT": [60, 60, 24], "SMELT": [60, 60, 24]}
DEFAULT_CACHE_MARGIN = {"lat": 0.5, "lon": 1.0}  # degrees
DEFAULT_CACHE_DIR = "{CFM_data_path}/forcing/{rcm_name}"
CACHE_FILE_NAME = "cache.h5"
# Cells per side of a cache chunk; each chunk holds the full time series. Measured 2026-10-16, XPEN055 cache of 1980
# cells, one cell not in the page cache: 16 -> 2.0-2.6 s, 8 -> 0.4 s, 4 -> 0.3 s, 2 -> 0.08 s.
CACHE_CHUNK = 4
CACHE_VERSION = 1
TIME_BLOCK = 64  # time steps read in one h5py call; 64 full ANT-12 maps are approximately 90 MB


@dataclass(frozen=True)
class DatasetSpec:
    """How the files of one RACMO dataset are named and which variable is the snowfall."""

    rcm_name: str
    data_path_key: str
    column_map_key: str
    file_glob: str  # "{var}" is replaced by the variable name
    year_regex: str  # groups: first year and, if present, last year in the file name
    snowfall: str


DATASETS: Dict[str, DatasetSpec] = {
    # ANT-12, e.g. tas_ANT-12_ERA5_evaluation_r1i1p1f1_UU-IMAU_RACMO24P-NN_v1-r1_day_19790101-19801231.nc
    "RACMO": DatasetSpec(
        "RACMO", "RACMO_data_path", "RACMO_to_CFM_column_map", "{var}_*.nc", r"_(\d{4})\d{4}-(\d{4})\d{4}\.nc$", "prsn"
    ),
    # XPEN055, e.g. tas.KNMI-1979.XPEN055.RACMO2.4p1_v5_historical.DD.nc
    "RACMO23": DatasetSpec(
        "RACMO23", "RACMO23_data_path", "RACMO23_to_CFM_column_map", "{var}.KNMI-*.nc", r"\.KNMI-(\d{4})\.", "sf"
    ),
}

# Derived source names and their inputs. "snowfall" stands for DatasetSpec.snowfall.
DERIVED = {"rf": ("pr", "snowfall"), "alb": ("rsusgl", "rsds")}
# Source variables clipped to >= 0 before the derived columns are made, as in ProcessRACMO.
CLIP_SOURCE = ("pr", "snowfall", "mltgl")


@dataclass
class ExtractSettings:
    """Everything one extraction needs. Use from_config() to fill it from config.yaml."""

    dataset: str
    data_path: str
    column_map: Dict[str, str]  # source or derived variable -> CFM column, in CSV column order
    start_year: int
    end_year: int
    unit_factors: Dict[str, Union[float, List[float]]] = field(default_factory=lambda: dict(DEFAULT_UNIT_FACTORS))
    processes: int = DEFAULT_PROCESSES
    time_block: int = TIME_BLOCK

    @property
    def spec(self) -> DatasetSpec:
        return DATASETS[self.dataset]

    @property
    def source_vars(self) -> List[str]:
        """Variables to read from the files: the mapped ones that are not derived, plus the inputs of the derived."""
        names = []
        for var in self.column_map:
            inputs = DERIVED.get(var, (var,))
            for name in inputs:
                name = self.spec.snowfall if name == "snowfall" else name
                if name not in names:
                    names.append(name)
        return names

    @classmethod
    def from_config(cls, dataset: str, cfg: Optional[Mapping] = None, **overrides) -> "ExtractSettings":
        if cfg is None:
            from cfm_config import config as cfg
        if dataset not in DATASETS:
            raise ValueError(f"Fast extraction supports {list(DATASETS)}, not {dataset!r}")
        spec = DATASETS[dataset]
        values = dict(
            dataset=dataset,
            data_path=cfg[spec.data_path_key],
            column_map=dict(cfg[spec.column_map_key]),
            start_year=int(cfg["start_year"]),
            end_year=int(cfg["end_year"]),
            unit_factors=dict(cfg.get("RACMO_to_CFM_unit_factors") or DEFAULT_UNIT_FACTORS),
            processes=int(cfg.get("forcing_processes") or DEFAULT_PROCESSES),
        )
        values.update(overrides)
        return cls(**values)


# ----------------------------------------------------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------------------------------------------------

_STANDARD_CALENDARS = {"standard", "gregorian", "proleptic_gregorian"}
_UNIT_NS = {
    "days": 86400 * 10**9,
    "day": 86400 * 10**9,
    "d": 86400 * 10**9,
    "hours": 3600 * 10**9,
    "hour": 3600 * 10**9,
    "h": 3600 * 10**9,
    "minutes": 60 * 10**9,
    "minute": 60 * 10**9,
    "min": 60 * 10**9,
    "seconds": 10**9,
    "second": 10**9,
    "s": 10**9,
}


def _attr_str(attrs: Mapping, name: str, default: Optional[str] = None) -> Optional[str]:
    """A string attribute from h5py, which gives bytes or numpy bytes for netCDF text attributes."""
    if name not in attrs:
        return default
    value = attrs[name]
    if isinstance(value, np.ndarray):
        value = value.item() if value.size == 1 else value.tobytes()
    if isinstance(value, bytes):
        value = value.decode()
    return str(value)


def decode_time(values: np.ndarray, units: str, calendar: str = "standard") -> pd.DatetimeIndex:
    """Decode CF time values ("<unit> since <reference>") to a DatetimeIndex, as xarray does for real calendars.

    Only the standard (Gregorian) calendars are supported. Others (noleap, 360_day, ...) raise ValueError rather than
    give silently wrong dates.
    """
    calendar = (calendar or "standard").lower()
    if calendar not in _STANDARD_CALENDARS:
        raise ValueError(f"Calendar {calendar!r} is not supported, only {sorted(_STANDARD_CALENDARS)}")
    match = re.fullmatch(r"\s*(\w+)\s+since\s+(.+?)\s*", units)
    if match is None or match.group(1).lower() not in _UNIT_NS:
        raise ValueError(f"Cannot decode time units {units!r}")
    unit_ns = _UNIT_NS[match.group(1).lower()]
    reference = pd.Timestamp(match.group(2))
    if reference.tzinfo is not None:
        reference = reference.tz_convert(None)
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.integer):
        offsets = values.astype(np.int64) * unit_ns
    else:
        offsets = np.round(values.astype(np.float64) * unit_ns).astype(np.int64)
    return pd.DatetimeIndex(reference.as_unit("ns").value + offsets, name="time")


def decode_values(data: np.ndarray, attrs: Mapping) -> np.ndarray:
    """Mask _FillValue and missing_value to NaN, then apply scale_factor and add_offset, as xarray's CF decoding does."""
    fills = []
    for name in ("_FillValue", "missing_value"):
        if name in attrs:
            fills.extend(np.atleast_1d(attrs[name]).tolist())
    scale = attrs.get("scale_factor")
    offset = attrs.get("add_offset")
    if scale is not None or offset is not None:
        scale_arr = np.asarray(scale if scale is not None else 1).reshape(())
        offset_arr = np.asarray(offset if offset is not None else 0).reshape(())
        dtype = np.result_type(data.dtype, scale_arr.dtype, offset_arr.dtype)
        if not np.issubdtype(dtype, np.floating):
            dtype = np.float64
    elif fills and not np.issubdtype(data.dtype, np.floating):
        dtype = np.float64
    else:
        dtype = data.dtype
    out = data.astype(dtype, copy=True)
    if fills:
        mask = np.isin(data, np.asarray(fills, dtype=data.dtype))
        out[mask] = np.nan
    if scale is not None:
        out *= scale_arr.astype(dtype)
    if offset is not None:
        out += offset_arr.astype(dtype)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Files and grid
# ----------------------------------------------------------------------------------------------------------------


def _file_years(spec: DatasetSpec, path: str) -> Tuple[int, int]:
    match = re.search(spec.year_regex, os.path.basename(path))
    if match is None:
        raise ValueError(f"Cannot read the years from file name {path}")
    first = int(match.group(1))
    last = int(match.group(2)) if match.lastindex and match.lastindex >= 2 else first
    return first, last


def list_files(settings: ExtractSettings) -> Dict[str, List[str]]:
    """Source files for each variable, in time order, that overlap start_year..end_year.

    Logs a warning for years in the range that no file covers (see BUG-003); the old code skips them silently too.
    """
    spec = settings.spec
    files: Dict[str, List[str]] = {}
    for var in settings.source_vars:
        paths = sorted(glob.glob(os.path.join(settings.data_path, spec.file_glob.format(var=var))))
        keep, years = [], set()
        for path in paths:
            first, last = _file_years(spec, path)
            if last >= settings.start_year and first <= settings.end_year:
                keep.append(path)
                years.update(range(first, last + 1))
        if not keep:
            raise FileNotFoundError(
                f"No {settings.dataset} files for {var!r} in {settings.data_path} for "
                f"{settings.start_year}-{settings.end_year}"
            )
        missing = sorted(set(range(settings.start_year, settings.end_year + 1)) - years)
        if missing:
            logging.warning(f"{settings.dataset} {var}: no files for years {missing} (BUG-003)")
        files[var] = keep
    return files


def read_grid(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """2D latitude and longitude (rlat, rlon) of a RACMO file."""
    with h5py.File(path, "r") as h5:
        return h5["lat"][:], h5["lon"][:]


def nearest_cell(lat2d: np.ndarray, lon2d: np.ndarray, lat: float, lon: float) -> Tuple[int, int, float]:
    """Grid cell (i, j, km) selected for (lat, lon) by the OLD rule, distance in degrees.

    i is the row (rlat), j the column (rlon), km the great circle distance from the site to the cell centre.

    TEMPORARY: this copies the cell selection of ProcessRACMO._xr_to_input_dataframe (src/force/process.py), including
    BUG-002 (no cos(lat) factor on the longitude difference), so the fast path agrees with the old CSVs. FEAT-002
    fixes BUG-002 in force.grid.nearest_cell, which has the same signature; use that function instead at the merge.
    """
    distance = np.sqrt(np.abs(lat2d - lat) ** 2 + np.abs(lon2d - lon) ** 2)
    i, j = np.unravel_index(distance.argmin(), distance.shape)
    phi1, phi2 = np.radians(lat), np.radians(lat2d[i, j])
    dphi, dlmb = phi2 - phi1, np.radians(lon2d[i, j] - lon)
    hav = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
    km = 2 * 6371.0 * np.arcsin(np.sqrt(hav))
    return int(i), int(j), float(km)


# ----------------------------------------------------------------------------------------------------------------
# Parallel read
# ----------------------------------------------------------------------------------------------------------------


def _read_file(job) -> Tuple[str, str, pd.DatetimeIndex, np.ndarray]:
    """Worker: read one variable from one file at the given cells. Returns decoded times and (time, cell) values."""
    path, var, rows, cols, block = job
    with h5py.File(path, "r") as h5:
        dataset = h5[var]
        if dataset.ndim == 4 and dataset.shape[1] != 1:
            raise ValueError(f"{path}:{var} has {dataset.shape[1]} levels; only one level is supported")
        if dataset.ndim not in (3, 4):
            raise ValueError(f"{path}:{var} has shape {dataset.shape}; expected (time, [level,] y, x)")
        y0, y1 = int(rows.min()), int(rows.max()) + 1
        x0, x1 = int(cols.min()), int(cols.max()) + 1
        n_time = dataset.shape[0]
        out = np.empty((n_time, len(rows)), dtype=dataset.dtype)
        for start in range(0, n_time, block):
            # every chunk is one full map, so this decompresses each chunk exactly once
            data = dataset[start : start + block, ..., y0:y1, x0:x1]
            data = data.reshape(data.shape[0], y1 - y0, x1 - x0)
            out[start : start + block] = data[:, rows - y0, cols - x0]
        values = decode_values(out, dataset.attrs)
        time_var = h5["time"]
        times = decode_time(
            time_var[:], _attr_str(time_var.attrs, "units"), _attr_str(time_var.attrs, "calendar", "standard")
        )
    return path, var, times, values


def _map_jobs(jobs: Sequence, processes: int) -> Iterable:
    if processes <= 1 or len(jobs) <= 1:
        return map(_read_file, jobs)
    # forkserver: fork() from a parent with threads (BLAS, dask) can deadlock. The workers import force.extract,
    # which reads config.yaml from the working directory, as every entry point of this repo does.
    method = "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
    executor = ProcessPoolExecutor(
        max_workers=min(processes, len(jobs)), mp_context=multiprocessing.get_context(method)
    )

    def results():
        with executor:
            yield from executor.map(_read_file, jobs)

    return results()


def read_raw(
    settings: ExtractSettings, cells: Sequence[Cell], files: Optional[Dict[str, List[str]]] = None
) -> Tuple[pd.DatetimeIndex, Dict[str, np.ndarray]]:
    """Read every source variable at the cells in one pass over the files.

    Returns the times and, for each variable, an array of shape (time, len(cells)). The times are not filtered yet.
    """
    if files is None:
        files = list_files(settings)
    rows = np.array([c[0] for c in cells], dtype=np.intp)
    cols = np.array([c[1] for c in cells], dtype=np.intp)
    jobs = [(path, var, rows, cols, settings.time_block) for var, paths in files.items() for path in paths]
    n_bytes = sum(os.path.getsize(job[0]) for job in jobs)
    logging.info(
        f"Reading {len(jobs)} {settings.dataset} files ({n_bytes / 1e9:.1f} GB) for {len(cells)} cells "
        f"with {settings.processes} processes..."
    )
    start = time.perf_counter()
    parts: Dict[str, List[Tuple[pd.DatetimeIndex, np.ndarray]]] = {var: [] for var in files}
    for done, (_, var, times, values) in enumerate(_map_jobs(jobs, settings.processes), 1):
        parts[var].append((times, values))
        if done % 25 == 0 or done == len(jobs):
            elapsed = time.perf_counter() - start
            logging.info(f"...{done}/{len(jobs)} files read, {elapsed:.0f} s")
    elapsed = time.perf_counter() - start
    logging.info(f"...read {n_bytes / 1e9:.1f} GB in {elapsed:.1f} s ({n_bytes / 1e6 / max(elapsed, 1e-9):.0f} MB/s)")

    times = None
    raw = {}
    for var, var_parts in parts.items():  # parts keep the file order, which is the time order
        var_times = pd.DatetimeIndex(np.concatenate([t.values for t, _ in var_parts]), name="time")
        if times is None:
            times = var_times
        elif not var_times.equals(times):
            raise ValueError(f"{settings.dataset}: the times of {var!r} differ from the times of {list(raw)[0]!r}")
        raw[var] = np.concatenate([v for _, v in var_parts], axis=0)
    return times, raw


# ----------------------------------------------------------------------------------------------------------------
# Derived columns, units, time filter
# ----------------------------------------------------------------------------------------------------------------


def derive_columns(raw: Mapping[str, np.ndarray], settings: ExtractSettings) -> Dict[str, np.ndarray]:
    """CFM columns from the source variables. All operations are per element, so this works for any array shape.

    The steps and their order are the same as in ProcessRACMO._xr_to_input_dataframe:
    clip pr, snowfall and mltgl to >= 0; rf = pr - snowfall; alb = rsusgl / rsds; rename with the column map;
    multiply by the unit factors; clip RAIN to >= 0.
    """
    snowfall = settings.spec.snowfall
    source = dict(raw)
    for name in CLIP_SOURCE:
        name = snowfall if name == "snowfall" else name
        if name in source:
            source[name] = np.clip(source[name], a_min=0, a_max=None)
    with np.errstate(divide="ignore", invalid="ignore"):
        for derived, inputs in DERIVED.items():
            if derived not in settings.column_map:
                continue
            a, b = (snowfall if name == "snowfall" else name for name in inputs)
            source[derived] = source[a] - source[b] if derived == "rf" else source[a] / source[b]
    columns = {cfm: source[var] for var, cfm in settings.column_map.items()}
    for cfm, factors in settings.unit_factors.items():
        if cfm in columns:
            for factor in np.atleast_1d(factors).tolist():
                columns[cfm] = columns[cfm] * factor  # a Python number keeps the float32 dtype (NEP 50)
    if "RAIN" in columns:
        columns["RAIN"] = np.clip(columns["RAIN"], a_min=0, a_max=None)
    return columns


def time_selection(times: pd.DatetimeIndex, start_year: int, end_year: int) -> np.ndarray:
    """Boolean mask: times in start_year..end_year, first occurrence of each time only."""
    in_range = (times.year >= start_year) & (times.year <= end_year)
    return np.asarray(in_range & ~times.duplicated(keep="first"))


def _extract(
    settings: ExtractSettings, cells: Sequence[Cell], files: Optional[Dict[str, List[str]]] = None
) -> Dict[Cell, pd.DataFrame]:
    unique_cells = list(dict.fromkeys((int(y), int(x)) for y, x in cells))
    times, raw = read_raw(settings, unique_cells, files)
    columns = derive_columns(raw, settings)
    keep = time_selection(times, settings.start_year, settings.end_year)
    times = times[keep]
    columns = {cfm: values[keep] for cfm, values in columns.items()}
    return {
        cell: pd.DataFrame({cfm: values[:, k] for cfm, values in columns.items()}, index=times)
        for k, cell in enumerate(unique_cells)
    }


# ----------------------------------------------------------------------------------------------------------------
# Public: extract_points
# ----------------------------------------------------------------------------------------------------------------


def extract_points(
    dataset: str, cells: Sequence[Cell], settings: Optional[ExtractSettings] = None, **overrides
) -> Dict[Cell, pd.DataFrame]:
    """Forcing DataFrames (CFM columns, daily time index) for grid cells, from one pass over the RACMO files.

    Args:
        dataset: "RACMO" (ANT-12) or "RACMO23" (XPEN055).
        cells: (y, x) grid indices, (rlat, rlon). Duplicates are read once.
        settings: optional settings; by default they come from config.yaml, with overrides as keyword arguments
            (e.g. processes=4, data_path=...).
    """
    if settings is None:
        settings = ExtractSettings.from_config(dataset, **overrides)
    elif settings.dataset != dataset:
        raise ValueError(f"settings are for {settings.dataset!r}, not {dataset!r}")
    return _extract(settings, cells)


# ----------------------------------------------------------------------------------------------------------------
# Local cache around the sites
# ----------------------------------------------------------------------------------------------------------------


def cache_path(dataset: str, cfg: Optional[Mapping] = None) -> str:
    """Path of cache.h5 for a dataset. The folder comes from forcing_cache_dir (placeholders: CFM_data_path,
    rcm_name)."""
    if cfg is None:
        from cfm_config import config as cfg
    folder = (cfg.get("forcing_cache_dir") or DEFAULT_CACHE_DIR).format(
        CFM_data_path=cfg["CFM_data_path"], rcm_name=dataset
    )
    return os.path.join(folder, CACHE_FILE_NAME)


def source_manifest(files: Mapping[str, Sequence[str]]) -> List[list]:
    """[variable, file name, size, mtime_ns] for each source file. A change in any of them makes the cache stale."""
    manifest = []
    for var, paths in files.items():
        for path in paths:
            stat = os.stat(path)
            manifest.append([var, os.path.basename(path), int(stat.st_size), int(stat.st_mtime_ns)])
    return manifest


def box_for_sites(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    sites: Sequence[Tuple[float, float]],
    margin: Mapping[str, float],
    cells: Sequence[Cell] = (),
) -> Box:
    """Index box (y0, y1, x0, x1) around the grid cells within the lat/lon range of the sites plus the margin.

    The box also contains the given cells (e.g. the selected cell of each site), so it is never empty. The grid is
    rotated, so the box in index space is the bounding box of the cells in the lat/lon range. Longitudes are not
    wrapped at +-180 degrees, which is correct for the Antarctic Peninsula.
    """
    lats = [s[0] for s in sites]
    lons = [s[1] for s in sites]
    ys, xs = [], []
    if sites:
        in_range = (
            (lat2d >= min(lats) - margin["lat"])
            & (lat2d <= max(lats) + margin["lat"])
            & (lon2d >= min(lons) - margin["lon"])
            & (lon2d <= max(lons) + margin["lon"])
        )
        ys, xs = np.nonzero(in_range)
        ys, xs = ys.tolist(), xs.tolist()
    ys += [c[0] for c in cells]
    xs += [c[1] for c in cells]
    if not ys:
        raise ValueError("box_for_sites needs at least one site or cell")
    return min(ys), max(ys) + 1, min(xs), max(xs) + 1


def box_contains(box: Box, cells: Iterable[Cell]) -> bool:
    y0, y1, x0, x1 = box
    return all(y0 <= y < y1 and x0 <= x < x1 for y, x in cells)


def union_box(a: Box, b: Box) -> Box:
    return min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])


def _identity(settings: ExtractSettings) -> dict:
    """The settings that the cached values depend on. A change in any of them makes the cache stale."""
    return {
        "version": CACHE_VERSION,
        "dataset": settings.dataset,
        "data_path": os.path.abspath(settings.data_path),
        "column_map": settings.column_map,
        "unit_factors": settings.unit_factors,
        "start_year": settings.start_year,
        "end_year": settings.end_year,
    }


def read_cache_meta(path: str) -> Optional[dict]:
    """The metadata of a cache file, or None if there is no readable cache."""
    if not os.path.exists(path):
        return None
    try:
        with h5py.File(path, "r") as h5:
            meta = json.loads(_attr_str(h5.attrs, "meta"))
    except (OSError, KeyError, TypeError, ValueError) as error:
        logging.warning(f"Cannot read forcing cache {path} ({error}); it will be made again")
        return None
    meta["box"] = tuple(meta["box"])
    return meta


def cache_stale_reason(meta: Optional[dict], settings: ExtractSettings, manifest: List[list]) -> Optional[str]:
    """Why the cache cannot be used for these settings and source files, or None if it can."""
    if meta is None:
        return "no cache"
    identity = _identity(settings)
    for key, value in identity.items():
        if json.loads(json.dumps(value)) != meta.get(key):
            return f"{key} changed"
    if meta.get("files") != manifest:
        return "source files changed (list, sizes or modification times)"
    return None


def build_cache(
    settings: ExtractSettings,
    box: Box,
    path: str,
    files: Optional[Dict[str, List[str]]] = None,
    margin: Optional[Mapping[str, float]] = None,
) -> None:
    """Read all cells of the box in one pass over the share and write them to a cache file.

    Each CFM column is stored as float32 (time, y, x) with chunks (time, CACHE_CHUNK, CACHE_CHUNK), so one cell
    reads one small chunk. The file is written under a temporary name and then renamed, so a stopped build leaves
    no partial cache.
    """
    if files is None:
        files = list_files(settings)
    manifest = source_manifest(files)
    y0, y1, x0, x1 = box
    yy, xx = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
    cells = list(zip(yy.ravel().tolist(), xx.ravel().tolist()))
    times, raw = read_raw(settings, cells, files)
    columns = derive_columns(raw, settings)
    del raw
    keep = time_selection(times, settings.start_year, settings.end_year)
    times = times[keep]
    lat2d, lon2d = read_grid(next(iter(files.values()))[0])

    meta = dict(
        _identity(settings),
        box=list(box),
        margin=dict(margin or {}),
        files=manifest,
        created=pd.Timestamp.now().isoformat(timespec="seconds"),
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp{os.getpid()}"
    shape = (len(times), y1 - y0, x1 - x0)
    chunks = (len(times), min(CACHE_CHUNK, shape[1]), min(CACHE_CHUNK, shape[2])) if len(times) else None
    try:
        with h5py.File(tmp_path, "w") as h5:
            h5.attrs["meta"] = json.dumps(meta)
            h5.create_dataset("time", data=times.values.astype("datetime64[ns]").astype(np.int64))
            h5["time"].attrs["units"] = "nanoseconds since 1970-01-01"
            h5.create_dataset("lat", data=lat2d[y0:y1, x0:x1])
            h5.create_dataset("lon", data=lon2d[y0:y1, x0:x1])
            columns_group = h5.create_group("columns")
            columns_group.attrs["order"] = json.dumps(list(columns))
            for cfm, values in columns.items():
                data = values[keep].reshape(shape).astype(np.float32, copy=False)
                columns_group.create_dataset(cfm, data=data, chunks=chunks)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    size = os.path.getsize(path)
    logging.info(f"Forcing cache written to {path}: box {box}, {shape[1] * shape[2]} cells, {size / 1e9:.2f} GB")


def read_cache_points(path: str, cells: Sequence[Cell]) -> Dict[Cell, pd.DataFrame]:
    """Forcing DataFrames for cells from a cache file. All the cells must be inside the cache box."""
    with h5py.File(path, "r") as h5:
        meta = json.loads(_attr_str(h5.attrs, "meta"))
        y0, y1, x0, x1 = meta["box"]
        if not box_contains((y0, y1, x0, x1), cells):
            raise ValueError(f"Cells {cells} are not all inside the cache box {meta['box']}")
        times = pd.DatetimeIndex(h5["time"][:].astype("datetime64[ns]"), name="time")
        order = json.loads(_attr_str(h5["columns"].attrs, "order"))
        frames = {}
        for y, x in dict.fromkeys((int(y), int(x)) for y, x in cells):
            frames[(y, x)] = pd.DataFrame({cfm: h5["columns"][cfm][:, y - y0, x - x0] for cfm in order}, index=times)
    return frames


def ensure_cache(
    settings: ExtractSettings,
    path: str,
    cells: Sequence[Cell],
    sites: Sequence[Tuple[float, float]],
    margin: Mapping[str, float],
    files: Optional[Dict[str, List[str]]] = None,
    grid: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> str:
    """Make sure that the cache at path is current and contains the cells. Build it again if not.

    A new box contains all the sites plus the margin, the given cells and, if the old cache is for the same data,
    the old box too, so no site drops out of the cache.
    """
    if files is None:
        files = list_files(settings)
    manifest = source_manifest(files)
    meta = read_cache_meta(path)
    reason = cache_stale_reason(meta, settings, manifest)
    if reason is None and box_contains(meta["box"], cells):
        return path
    if reason is None:
        reason = "a cell is outside the cache box"
    if grid is None:
        grid = read_grid(next(iter(files.values()))[0])
    box = box_for_sites(grid[0], grid[1], sites, margin, cells)
    if (
        meta is not None
        and meta.get("dataset") == settings.dataset
        and meta.get("data_path") == os.path.abspath(settings.data_path)
    ):
        box = union_box(box, meta["box"])
    logging.info(f"Making the {settings.dataset} forcing cache again ({reason}); new box {box}")
    build_cache(settings, box, path, files, margin)
    return path


# ----------------------------------------------------------------------------------------------------------------
# Public: forcing for sites, CSV output
# ----------------------------------------------------------------------------------------------------------------


def forcing_for_sites(
    dataset: str,
    sites: Mapping[str, Tuple[float, float]],
    use_cache: Optional[bool] = None,
    cache_sites: Sequence[Tuple[float, float]] = (),
    cfg: Optional[Mapping] = None,
    **overrides,
) -> Dict[str, pd.DataFrame]:
    """Forcing DataFrame for each site (name -> (lat, lon)), from one pass over the share or from the cache.

    Args:
        dataset: "RACMO" or "RACMO23".
        sites: the sites to return.
        use_cache: use cache.h5; default from the forcing_cache key (true if not set).
        cache_sites: more (lat, lon) points for the cache box, e.g. all borehole_sites, so later sites near them
            need no pass over the share.
        cfg: config dict; default config.yaml.
    """
    if cfg is None:
        from cfm_config import config as cfg
    settings = ExtractSettings.from_config(dataset, cfg, **overrides)
    if use_cache is None:
        use_cache = bool(cfg.get("forcing_cache", True))
    files = list_files(settings)
    grid = read_grid(next(iter(files.values()))[0])
    site_cells = {}
    for name, (lat, lon) in sites.items():
        i, j, km = nearest_cell(grid[0], grid[1], lat, lon)
        site_cells[name] = (i, j)
        logging.info(f"{dataset} {name} ({lat}, {lon}): grid cell (rlat, rlon) = ({i}, {j}), {km:.1f} km away")
    cells = list(site_cells.values())

    if use_cache:
        path = cache_path(dataset, cfg)
        margin = dict(cfg.get("forcing_cache_margin") or DEFAULT_CACHE_MARGIN)
        box_sites = list(sites.values()) + [tuple(s) for s in cache_sites]
        ensure_cache(settings, path, cells, box_sites, margin, files, grid)
        frames = read_cache_points(path, cells)
    else:
        frames = _extract(settings, cells, files)
    return {name: frames[cell] for name, cell in site_cells.items()}


def write_forcing_csvs(
    dataset: str, targets: Mapping[str, Tuple[float, float]], overwrite: bool = False, **kwargs
) -> List[str]:
    """Write one forcing CSV for each target (CSV path -> (lat, lon)) in one pass. Returns the paths written.

    Existing CSVs are skipped unless overwrite is true, as in ProcessBase.process. kwargs go to forcing_for_sites.
    """
    todo = {path: site for path, site in targets.items() if overwrite or not os.path.exists(path)}
    for path in targets:
        if path not in todo:
            logging.info(f"Forcing CSV already exists at {path}. Skipping.")
    if not todo:
        return []
    frames = forcing_for_sites(dataset, todo, **kwargs)
    for path, frame in frames.items():
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        frame.to_csv(path)
        logging.info(f"Forcing CSV saved to {path}")
    return list(frames)
