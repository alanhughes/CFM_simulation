"""Fast extraction of RACMO and MAR daily forcing at many grid cells in one pass (FEAT-003).

The RACMO and MAR files store each day as one compressed HDF5 chunk of the full map, so reading one cell costs as much
as reading the whole file. This module therefore reads each (variable, file) pair once with h5py, in parallel worker
processes, and keeps every requested cell from it. Extra cells cost no extra reads from the share.

h5py does not decode the data as xarray does, so this module does it itself: CF time units and calendar, _FillValue
and missing_value (to NaN), scale_factor and add_offset. The derived columns (rain and albedo), the unit factors and
the column maps follow the old xarray code (ProcessRACMO._xr_to_input_dataframe and ProcessMAR._xr_to_input_dataframe)
step by step, so the output agrees with the old CSVs. MAR variables have extra axes (SECTOR, OUTLAY, a height level)
inside each chunk; the worker keeps one index on each, by coordinate value (DatasetSpec.level_keys, fixed_levels).

A local cache (cache.h5, next to the forcing CSVs of FEAT-002) keeps all cells in a box around the sites, so a new site
near the others needs no new pass over the share. See ensure_cache(). process_fast() writes the forcing CSV of each
model point through ProcessBase.save_forcing, so the CSVs and JSON files are the same as from the old path.
"""

import glob
import json
import logging
import multiprocessing
import os
import re
import sys
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
DEFAULT_CACHE_DIR = "{forcing_data_path}/{rcm_name}/{release}"  # the folder of the FEAT-002 forcing CSVs
CACHE_FILE_NAME = "cache.h5"
# Cells per side of a cache chunk; each chunk holds the full time series. Measured 2026-10-16, XPEN055 cache of 1980
# cells, one cell not in the page cache: 16 -> 2.0-2.6 s, 8 -> 0.4 s, 4 -> 0.3 s, 2 -> 0.08 s.
CACHE_CHUNK = 4
CACHE_VERSION = 1
TIME_BLOCK = 64  # time steps read in one h5py call; 64 full ANT-12 maps are approximately 90 MB


@dataclass(frozen=True)
class DatasetSpec:
    """How the files of one dataset are named, the names of its coordinates, and its unit changes."""

    rcm_name: str
    data_path_key: str
    column_map_key: str
    file_glob: str  # "{var}" is replaced by the variable name; without it, every variable is in every file
    year_regex: str  # groups: first year and, if present, last year in the file name
    snowfall: Optional[str]  # the snowfall variable for the derived rain (RACMO only)
    time_name: str = "time"  # also the index name of the output DataFrame, as in the old CSVs
    lat_name: str = "lat"
    lon_name: str = "lon"
    # config key of the unit factors (None: no key) and the factors if the key is not set
    unit_factors_key: Optional[str] = "RACMO_to_CFM_unit_factors"
    default_unit_factors: Mapping = field(default_factory=lambda: dict(DEFAULT_UNIT_FACTORS))
    unit_offsets: Mapping = field(default_factory=dict)  # CFM column -> value added after the unit factors
    # Axes between time and (y, x) with more than one value: keep the index whose coordinate value is the config value
    # (level_keys, axis -> config key) or the fixed value (fixed_levels). An axis of size 1 needs no entry.
    level_keys: Mapping = field(default_factory=dict)
    fixed_levels: Mapping = field(default_factory=dict)


DATASETS: Dict[str, DatasetSpec] = {
    # ANT-12, e.g. tas_ANT-12_ERA5_evaluation_r1i1p1f1_UU-IMAU_RACMO24P-NN_v1-r1_day_19790101-19801231.nc
    "RACMO": DatasetSpec(
        "RACMO", "RACMO_data_path", "RACMO_to_CFM_column_map", "{var}_*.nc", r"_(\d{4})\d{4}-(\d{4})\d{4}\.nc$", "prsn"
    ),
    # XPEN055, e.g. tas.KNMI-1979.XPEN055.RACMO2.4p1_v5_historical.DD.nc
    "RACMO23": DatasetSpec(
        "RACMO23", "RACMO23_data_path", "RACMO23_to_CFM_column_map", "{var}.KNMI-*.nc", r"\.KNMI-(\d{4})\.", "sf"
    ),
    # MAR 27.5 km, e.g. MARv3.14.3-27.5km-daily-ERA5-1979.nc: one file per year with all the variables. The old code
    # (ProcessMAR) kept SECTOR = config sector and OUTLAY = 0.0 (surface layer), and changed degC to K.
    "MAR": DatasetSpec(
        "MAR",
        "MAR_data_path",
        "MAR_to_CFM_column_map",
        "*-[0-9][0-9][0-9][0-9].nc",
        r"-(\d{4})\.nc$",
        None,
        time_name="TIME",
        lat_name="LAT",
        lon_name="LON",
        unit_factors_key=None,
        default_unit_factors={},
        unit_offsets={"T2m": 273.15, "TSKIN": 273.15},
        level_keys={"SECTOR": "sector"},
        fixed_levels={"OUTLAY": 0.0},
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
    unit_offsets: Dict[str, float] = field(default_factory=dict)  # CFM column -> value added after the factors
    levels: Dict[str, float] = field(default_factory=dict)  # axis name -> coordinate value to keep

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
        unit_factors = cfg.get(spec.unit_factors_key) if spec.unit_factors_key else None
        values = dict(
            dataset=dataset,
            data_path=cfg[spec.data_path_key],
            column_map=dict(cfg[spec.column_map_key]),
            start_year=int(cfg["start_year"]),
            end_year=int(cfg["end_year"]),
            unit_factors=dict(unit_factors or spec.default_unit_factors),
            processes=int(cfg.get("forcing_processes") or DEFAULT_PROCESSES),
            unit_offsets=dict(spec.unit_offsets),
            levels={**spec.fixed_levels, **{axis: float(cfg[key]) for axis, key in spec.level_keys.items()}},
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

    It does not check that the files cover all the years; see check_file_years().
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
        files[var] = keep
    return files


def check_file_years(settings: ExtractSettings, files: Mapping[str, Sequence[str]]) -> None:
    """Stop with ForcingCoverageError before the long pass if the file names do not cover start_year..end_year for
    every variable (BUG-003)."""
    from .cache import check_date_coverage

    for var, paths in files.items():
        years = set()
        for path in paths:
            first, last = _file_years(settings.spec, path)
            years.update(range(first, last + 1))
        dates = [pd.Timestamp(year, month, 1) for year in sorted(years) for month in (1, 12)]
        check_date_coverage(dates, settings.start_year, settings.end_year, f"{settings.dataset} {var}")


def read_grid(path: str, spec: DatasetSpec) -> Tuple[np.ndarray, np.ndarray]:
    """2D latitude and longitude (y, x) of a file of the dataset."""
    with h5py.File(path, "r") as h5:
        return h5[spec.lat_name][:], h5[spec.lon_name][:]


# ----------------------------------------------------------------------------------------------------------------
# Parallel read
# ----------------------------------------------------------------------------------------------------------------


def _axis_name(dataset: h5py.Dataset, axis: int) -> Optional[str]:
    """The netCDF dimension name of an axis, from its HDF5 dimension scale, or None if it has no scale."""
    try:
        return dataset.dims[axis][0].name.rsplit("/", 1)[-1]
    except (IndexError, RuntimeError):
        return None


def _level_index(h5: h5py.File, dataset: h5py.Dataset, axis: int, levels: Mapping[str, float]) -> int:
    """The index to keep on an axis between time and (y, x): the given coordinate value, or 0 if the axis has size 1."""
    name = _axis_name(dataset, axis)
    if name in levels:
        found = np.flatnonzero(h5[name][:] == levels[name])
        if len(found) != 1:
            raise ValueError(f"{h5.filename}:{dataset.name} has no single {name} = {levels[name]}")
        return int(found[0])
    if dataset.shape[axis] == 1:
        return 0
    raise ValueError(
        f"{h5.filename}:{dataset.name} has {dataset.shape[axis]} values on axis {name or axis}; "
        f"give the value to keep in DatasetSpec.level_keys or fixed_levels"
    )


def _read_file(job) -> Tuple[str, str, pd.DatetimeIndex, np.ndarray, int]:
    """Worker: read one variable from one file at the given cells.

    Returns the decoded times, the (time, cell) values and the stored (compressed) bytes of the variable.
    """
    path, var, rows, cols, block, time_name, levels = job
    with h5py.File(path, "r") as h5:
        dataset = h5[var]
        if dataset.ndim < 3:
            raise ValueError(f"{path}:{var} has shape {dataset.shape}; expected (time, [levels,] y, x)")
        index = tuple(_level_index(h5, dataset, axis, levels) for axis in range(1, dataset.ndim - 2))
        y0, y1 = int(rows.min()), int(rows.max()) + 1
        x0, x1 = int(cols.min()), int(cols.max()) + 1
        n_time = dataset.shape[0]
        out = np.empty((n_time, len(rows)), dtype=dataset.dtype)
        for start in range(0, n_time, block):
            # every chunk is one full map (with all its levels), so this decompresses each chunk exactly once
            data = dataset[(slice(start, start + block), *index, slice(y0, y1), slice(x0, x1))]
            out[start : start + block] = data[:, rows - y0, cols - x0]
        values = decode_values(out, dataset.attrs)
        n_bytes = dataset.id.get_storage_size()
        time_var = h5[time_name]
        times = decode_time(
            time_var[:], _attr_str(time_var.attrs, "units"), _attr_str(time_var.attrs, "calendar", "standard")
        )
    return path, var, times, values, n_bytes


def _map_jobs(jobs: Sequence, processes: int) -> Iterable:
    if processes <= 1 or len(jobs) <= 1:
        return map(_read_file, jobs)
    # fork on Linux: the workers need no new imports, and a caller needs no `if __name__ == "__main__":` guard
    # (spawn and forkserver import the main script again in each worker, so a script without the guard runs again).
    # The parent holds no open HDF5 file at this point. Elsewhere (macOS, Windows) fork is not safe, so use spawn;
    # there the calling script must have the guard.
    method = "fork" if sys.platform.startswith("linux") else "spawn"
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
    time_name = settings.spec.time_name
    jobs = [
        (path, var, rows, cols, settings.time_block, time_name, settings.levels)
        for var, paths in files.items()
        for path in paths
    ]
    n_files = len({job[0] for job in jobs})
    logging.info(
        f"Reading {len(jobs)} {settings.dataset} variables from {n_files} files for {len(cells)} cells "
        f"with {settings.processes} processes..."
    )
    start = time.perf_counter()
    n_bytes = 0
    parts: Dict[str, List[Tuple[pd.DatetimeIndex, np.ndarray]]] = {var: [] for var in files}
    for done, (_, var, times, values, var_bytes) in enumerate(_map_jobs(jobs, settings.processes), 1):
        parts[var].append((times, values))
        n_bytes += var_bytes
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
    multiply by the unit factors; clip the rain column (from rf) to >= 0. Then add the unit offsets, as
    ProcessMAR._xr_to_input_dataframe did (+273.15 for the MAR temperatures). MAR has no derived columns or factors.
    """
    snowfall = settings.spec.snowfall
    source = dict(raw)
    for name in CLIP_SOURCE:
        name = snowfall if name == "snowfall" else name
        if name is not None and name in source:
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
    rain = settings.column_map.get("rf")
    if rain in columns:
        columns[rain] = np.clip(columns[rain], a_min=0, a_max=None)
    for cfm, offset in settings.unit_offsets.items():
        if cfm in columns:
            columns[cfm] = columns[cfm] + offset  # a Python number keeps the float32 dtype (NEP 50)
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
    times = times[keep].rename(settings.spec.time_name)
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
    """Forcing DataFrames (CFM columns, daily time index) for grid cells, from one pass over the RCM files.

    Args:
        dataset: "RACMO" (ANT-12), "RACMO23" (XPEN055) or "MAR".
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


def cache_path(dataset: str) -> str:
    """Path of cache.h5 for a dataset: forcing_cache_dir/cache.h5, by default in the folder of the forcing CSVs,
    {forcing_data_path}/<rcm_name>/<release>/ (FEAT-002). Placeholders: CFM_data_path, forcing_data_path, rcm_name,
    release."""
    from cfm_config import config

    from .cache import forcing_root
    from .model_point import rcm_release

    template = config.get("forcing_cache_dir") or DEFAULT_CACHE_DIR
    folder = template.format(
        CFM_data_path=config["CFM_data_path"],
        forcing_data_path=forcing_root(),
        rcm_name=dataset,
        release=rcm_release(dataset) if "{release}" in template else "",
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
    """The settings that the cached values depend on. A change in any of them makes the cache stale.

    unit_offsets and levels are only in the identity if they are set (MAR), so the RACMO caches stay valid.
    """
    identity = {
        "version": CACHE_VERSION,
        "dataset": settings.dataset,
        "data_path": os.path.abspath(settings.data_path),
        "column_map": settings.column_map,
        "unit_factors": settings.unit_factors,
        "start_year": settings.start_year,
        "end_year": settings.end_year,
    }
    for key in ("unit_offsets", "levels"):
        if getattr(settings, key):
            identity[key] = getattr(settings, key)
    return identity


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
    lat2d, lon2d = read_grid(next(iter(files.values()))[0], settings.spec)

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
        time_name = DATASETS[meta["dataset"]].time_name  # the index name of the old CSVs
        times = pd.DatetimeIndex(h5["time"][:].astype("datetime64[ns]"), name=time_name)
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
        grid = read_grid(next(iter(files.values()))[0], settings.spec)
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


def forcing_for_cells(
    dataset: str,
    cells: Sequence[Cell],
    use_cache: Optional[bool] = None,
    cache_sites: Sequence[Tuple[float, float]] = (),
    files: Optional[Dict[str, List[str]]] = None,
    **overrides,
) -> Tuple[Dict[Cell, pd.DataFrame], List[str]]:
    """Forcing DataFrames for grid cells (i, j), from one pass over the share or from the cache.

    Args:
        dataset: "RACMO", "RACMO23" or "MAR".
        cells: (i, j) grid indices, for example from force.model_point.
        use_cache: use cache.h5; default from the forcing_cache key (true if not set).
        cache_sites: (lat, lon) points for the cache box, for example all the sites, so later sites near them need no
            pass over the share. The cells are always in the box.
        files: the source files from list_files(); default: list them.
        overrides: ExtractSettings fields, for example processes=4 or end_year=2023.

    Returns:
        The DataFrame of each cell, and the list of the source files.
    """
    from cfm_config import config

    settings = ExtractSettings.from_config(dataset, **overrides)
    if use_cache is None:
        use_cache = bool(config.get("forcing_cache", True))
    if files is None:
        files = list_files(settings)
    check_file_years(settings, files)
    cells = [(int(i), int(j)) for i, j in cells]
    if use_cache:
        path = cache_path(dataset)
        margin = dict(config.get("forcing_cache_margin") or DEFAULT_CACHE_MARGIN)
        ensure_cache(settings, path, cells, list(cache_sites), margin, files)
        frames = read_cache_points(path, cells)
    else:
        frames = _extract(settings, cells, files)
    # MAR has all the variables in each file, so a file can occur for more than one variable
    return frames, list(dict.fromkeys(path for paths in files.values() for path in paths))


def process_fast(
    processors: Sequence,
    use_cache: Optional[bool] = None,
    overwrite: bool = False,
    cache_sites: Optional[Sequence[Tuple[float, float]]] = None,
    **overrides,
) -> List[str]:
    """Write the forcing CSV and JSON of each ProcessRACMO/ProcessRACMO23/ProcessMAR in one pass. Returns the CSVs
    written.

    Existing CSVs are checked and skipped, as in ProcessBase.process, unless overwrite is true. Processors that share
    a model point share one CSV. The cache box contains cache_sites (default: all active sites in sites.yaml) and the
    locations of the processors.
    """
    from cfm_config import config, sites

    datasets = {p._rcm_name for p in processors}
    if len(datasets) != 1:
        raise ValueError(f"process_fast needs processors of one dataset, not {sorted(datasets)}")
    dataset = datasets.pop()
    start_year, end_year = config["start_year"], config["end_year"]
    todo, seen = [], set()
    for processor in processors:
        save_path = processor._save_path
        if save_path in seen:
            continue
        seen.add(save_path)
        if os.path.exists(save_path) and not overwrite:
            logging.info(f"Processed RCM data already exists at {save_path}. Skipping processing.")
            processor._check_existing(save_path, start_year, end_year, f"{dataset} ({processor.model_point.release})")
        else:
            todo.append(processor)
    if not todo:
        return []

    settings = ExtractSettings.from_config(dataset, **overrides)
    files = list_files(settings)
    lat2d, lon2d = read_grid(next(iter(files.values()))[0], settings.spec)
    for processor in todo:
        mp = processor.model_point
        logging.info(
            f"{processor._borehole_lat}, {processor._borehole_lon}: model point {mp.tag}, "
            f"{mp.distance_km:.1f} km from the cell centre"
        )
        if not (np.isclose(lat2d[mp.i, mp.j], mp.cell_lat) and np.isclose(lon2d[mp.i, mp.j], mp.cell_lon)):
            raise ValueError(f"{dataset}: the grid of {settings.data_path} does not agree with model point {mp}")
    if cache_sites is None:
        cache_sites = [(site.lat, site.lon) for site in sites().values()]
    cache_sites = list(cache_sites) + [(p._borehole_lat, p._borehole_lon) for p in todo]
    cells = [(p.model_point.i, p.model_point.j) for p in todo]
    use_cache = bool(config.get("forcing_cache", True)) if use_cache is None else use_cache
    frames, source_files = forcing_for_cells(dataset, cells, use_cache, cache_sites, files, **overrides)

    extra = {
        "extraction": "h5py (FEAT-003, src/force/extract.py)",
        "extraction_cache": cache_path(dataset) if use_cache else None,
        "unit_factors_applied": settings.unit_factors,
    }
    if settings.unit_offsets:
        extra["unit_offsets_applied"] = settings.unit_offsets
    if settings.levels:
        extra["levels"] = settings.levels
    written = []
    for processor, cell in zip(todo, cells):
        processor.save_forcing(processor._save_path, frames[cell], source_files, extra=extra)
        written.append(processor._save_path)
    return written
