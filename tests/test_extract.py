"""Tests for the fast RACMO extraction and the forcing cache (FEAT-003), on small synthetic files.

The synthetic files copy the layout of the real data: ANT-12 (RACMO) files of several years, (time, rlat, rlon),
_FillValue and missing_value 1e20, times at 12:00; XPEN055 (RACMO23) files of one year, (time, height, rlat, rlon),
_FillValue -9999, times at 00:00. The main check is that extract_points() gives the same DataFrame as the old xarray
code in src/force/process.py. Run from the repo root.
"""

import json
import os

import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cfm_config import config
from force import extract
from force.cache import ForcingCoverageError, forcing_json_path
from force.grid import nearest_cell
from force.model_point import clear_cache
from force.process import ProcessMAR, ProcessRACMO, ProcessRACMO23

SITE_A = (-67.6, -65.1)
SITE_B = (-67.78, -65.0)  # the cell of this site has a fill value in each file

NY, NX = 9, 11
RACMO_MAP = {
    "tas": "T2m",
    "ts": "TSKIN",
    "sublgl": "SUBLIM",
    "hfssgl": "QH",
    "hflsgl": "QL",
    "rsds": "SW_d",
    "rlds": "LW_d",
    "rf": "RAIN",
    "prsn": "BDOT",
    "alb": "ALBEDO",
    "mltgl": "SMELT",
}
RACMO23_MAP = {
    "tas": "T2m",
    "ts": "TSKIN",
    "subltot": "SUBLIM",
    "hfssgl": "QH",
    "hflsgl": "QL",
    "rsds": "SW_d",
    "rlds": "LW_d",
    "rf": "RAIN",
    "sf": "BDOT",
    "alb": "ALBEDO",
    "mltgl": "SMELT",
}
SOURCE = {
    "RACMO": [v for v in RACMO_MAP if v not in ("rf", "alb")] + ["pr", "rsusgl"],
    "RACMO23": [v for v in RACMO23_MAP if v not in ("rf", "alb")] + ["pr", "rsusgl"],
}
# source files: RACMO 2 groups of years and RACMO23 3 years for each variable; MAR 3 years with all the variables
N_SOURCE_FILES = {"RACMO": 2 * len(SOURCE["RACMO"]), "RACMO23": 3 * len(SOURCE["RACMO23"]), "MAR": 3}


def _grid():
    """A small rotated-looking grid around the Antarctic Peninsula."""
    yy, xx = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    lat = -68.0 + 0.11 * yy + 0.02 * xx
    lon = -66.0 + 0.27 * xx - 0.05 * yy
    return lat, lon


def _values(rng, var, shape):
    data = rng.normal(0, 1, shape).astype(np.float32)
    if var in ("tas", "ts"):
        data = (260 + 5 * data).astype(np.float32)
    elif var in ("pr", "prsn", "sf", "mltgl", "sublgl", "subltot"):
        data = (1e-5 * data).astype(np.float32)  # kg/m2/s, some negative to test the clipping
    elif var == "rsds":
        data = np.abs(200 * data).astype(np.float32)
        data[::5] = 0  # polar night: alb = 0/0
    elif var == "rsusgl":
        data = np.abs(150 * data).astype(np.float32)
        data[::5] = 0
    else:
        data = (50 * data).astype(np.float32)
    data[3, 2, 4] = np.nan  # written as the fill value
    return data


def _write_racmo(folder, variables, file_years):
    """ANT-12 style files: one file per variable and group of years."""
    rng = np.random.default_rng(1)
    lat, lon = _grid()
    for var in variables:
        for first, last in file_years:
            time = pd.date_range(f"{first}-01-01", f"{last}-12-31", freq="D")
            days = ((time - pd.Timestamp("1950-01-01")) / pd.Timedelta(days=1)).to_numpy() + 0.5
            ds = xr.Dataset(
                {var: (("time", "rlat", "rlon"), _values(rng, var, (len(time), NY, NX)))},
                coords={
                    "time": ("time", days, {"units": "days since 1950-01-01", "calendar": "standard"}),
                    "lat": (("rlat", "rlon"), lat),
                    "lon": (("rlat", "rlon"), lon),
                    "rlat": np.arange(NY, dtype=float),
                    "rlon": np.arange(NX, dtype=float),
                },
            )
            name = f"{var}_ANT-12_TEST_day_{first}0101-{last}1231.nc"
            ds.to_netcdf(
                folder / name,
                engine="netcdf4",
                encoding={
                    var: {
                        "zlib": True,
                        "complevel": 1,
                        "shuffle": True,
                        "chunksizes": (1, NY, NX),
                        "_FillValue": np.float32(1e20),
                        "missing_value": np.float32(1e20),
                    }
                },
            )


def _write_racmo23(folder, variables, years):
    """XPEN055 style files: one file per variable and year, with a height dimension of size 1."""
    rng = np.random.default_rng(2)
    lat, lon = _grid()
    for var in variables:
        for year in years:
            time = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
            days = ((time - pd.Timestamp("1950-01-01")) / pd.Timedelta(days=1)).to_numpy()
            data = _values(rng, var, (len(time), NY, NX))[:, None]
            ds = xr.Dataset(
                {var: (("time", "height", "rlat", "rlon"), data)},
                coords={
                    "time": ("time", days, {"units": "days since 1950-01-01 00:00:00.0", "calendar": "standard"}),
                    "height": [0.0],
                    "lat": (("rlat", "rlon"), lat),
                    "lon": (("rlat", "rlon"), lon),
                    "rlat": np.arange(NY, dtype=float),
                    "rlon": np.arange(NX, dtype=float),
                },
            )
            name = f"{var}.KNMI-{year}.XPEN055.TEST.DD.nc"
            ds.to_netcdf(
                folder / name,
                engine="netcdf4",
                encoding={
                    var: {
                        "zlib": True,
                        "complevel": 1,
                        "shuffle": True,
                        "chunksizes": (1, 1, NY, NX),
                        "_FillValue": np.float32(-9999.0),
                    }
                },
            )


MAR_MAP = {
    "TTZ": "T2m",
    "TI1": "TSKIN",
    "SU": "SUBLIM",
    "SHF": "QH",
    "LHF": "QL",
    "SWD": "SW_d",
    "LWD": "LW_d",
    "RF": "RAIN",
    "SF": "BDOT",
    "AL2": "ALBEDO",
    "ME": "SMELT",
}
# the dimensions of each MAR variable between TIME and (Y15_176, X18_215), as in MARv3.14.3-27.5km-daily-ERA5
MAR_LEVELS = {"TTZ": "ZTQLEV1_1", "TI1": "OUTLAY", "SU": "SECTOR", "AL2": "SECTOR", "ME": "SECTOR1_1", "SMB": "SECTOR"}
MAR_FILLED = ("SU", "RF", "SF", "ME", "SMB")  # variables with _FillValue and missing_value -1e34
MAR_RELEASE = "MARvTEST-27.5km-daily-ERA5"


def _write_mar(folder, years):
    """MAR style files: one file per year with all the variables, some with a SECTOR, OUTLAY or height axis.

    SECTOR is [2, 1] and OUTLAY starts at 0.1, so a wrong index (for example always 0) gives other values.
    """
    rng = np.random.default_rng(3)
    lat, lon = _grid()
    levels = {"SECTOR": [2.0, 1.0], "OUTLAY": [0.1, 0.0, 0.2], "SECTOR1_1": [1.0], "ZTQLEV1_1": [2.0]}
    for year in years:
        time = pd.date_range(f"{year}-01-01 12:00", f"{year}-12-31 12:00", freq="D")
        minutes = ((time - pd.Timestamp("1900-01-01")) / pd.Timedelta(minutes=1)).to_numpy()
        data_vars, encoding = {}, {}
        for var in list(MAR_MAP) + ["SMB"]:
            level = MAR_LEVELS.get(var)
            shape = (len(time), len(levels[level]), NY, NX) if level else (len(time), NY, NX)
            data = rng.normal(0, 1, shape).astype(np.float32)
            if var in ("TTZ", "TI1"):
                data = (-20 + 5 * data).astype(np.float32)  # degC
            data[(3,) + (0,) * (data.ndim - 3) + (2, 4)] = np.nan
            if level:
                data[(3, -1, 2, 4)] = np.nan  # also at the kept level, for SECTOR (index 1)
            dims = ("TIME",) + ((level,) if level else ()) + ("Y15_176", "X18_215")
            data_vars[var] = (dims, data)
            encoding[var] = {"zlib": True, "complevel": 4, "shuffle": True, "chunksizes": (1,) + shape[1:]}
            if var in MAR_FILLED:
                encoding[var].update(_FillValue=np.float32(-1e34), missing_value=np.float32(-1e34))
            else:
                encoding[var]["_FillValue"] = np.float32(np.nan)
        ds = xr.Dataset(
            data_vars,
            coords={
                "TIME": ("TIME", minutes, {"units": "minutes since 1900-01-01 00:00:00", "calendar": "gregorian"}),
                "LAT": (("Y15_176", "X18_215"), lat),
                "LON": (("Y15_176", "X18_215"), lon),
                "X18_215": np.arange(NX, dtype=np.float32) * 27.5,
                "Y15_176": np.arange(NY, dtype=np.float32) * 27.5,
                **{name: np.array(values, dtype=np.float32) for name, values in levels.items()},
            },
        )
        ds.to_netcdf(folder / f"{MAR_RELEASE}-{year}.nc", engine="netcdf4", encoding=encoding)
    (folder / "AIS_files_for_download.txt").write_text("not data\n")  # the real folder has other files too
    (folder / "OLD").mkdir(exist_ok=True)


@pytest.fixture(scope="module")
def racmo_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("racmo")
    # tsgl checks that the file pattern ts_* does not also match tsgl_*
    _write_racmo(folder, SOURCE["RACMO"] + ["tsgl"], [(1979, 1980), (1981, 1983)])
    return folder


@pytest.fixture(scope="module")
def racmo23_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("racmo23")
    _write_racmo23(folder, SOURCE["RACMO23"], [1979, 1980, 1981])
    return folder


@pytest.fixture(scope="module")
def mar_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("mar")
    _write_mar(folder, [1979, 1980, 1981])
    return folder


@pytest.fixture
def cfg(monkeypatch, racmo_dir, racmo23_dir, mar_dir, tmp_path):
    """The shared config dict, with the synthetic data paths and a temporary CFM_data_path."""
    for key, value in {
        "RACMO_data_path": str(racmo_dir),
        "RACMO23_data_path": str(racmo23_dir),
        "MAR_data_path": str(mar_dir),
        "RACMO_to_CFM_column_map": RACMO_MAP,
        "RACMO23_to_CFM_column_map": RACMO23_MAP,
        "MAR_to_CFM_column_map": MAR_MAP,
        "sector": 1.0,
        "RACMO_additional_vars": ["pr", "rsusgl"],
        "start_year": 1979,
        "end_year": 1981,
        "CFM_data_path": str(tmp_path / "CFM_data"),
        "forcing_processes": 1,
    }.items():
        monkeypatch.setitem(config, key, value)
    for key in (
        "RACMO_to_CFM_unit_factors",
        "forcing_data_path",
        "forcing_cache_dir",
        "forcing_cache_margin",
        "forcing_cache",
    ):
        monkeypatch.delitem(config, key, raising=False)
    clear_cache()
    yield config
    clear_cache()


def _cell(site):
    """The model point cell (i, j) of a site on the synthetic grid (FEAT-002 rule)."""
    lat, lon = _grid()
    return nearest_cell(lat, lon, *site)[:2]


def _old_dataframe(cls, lat, lon):
    """The forcing DataFrame of the xarray code that FEAT-003 removed (ProcessRACMO._xr_to_input_dataframe).

    This is a copy of that code. It is the reference for the tests of the h5py extraction.
    """
    processor = cls(lat, lon)
    frames = []
    for xr_data in processor._read_data():
        y_idx, x_idx, _ = nearest_cell(xr_data["lat"], xr_data["lon"], lat, lon)
        data = xr_data.isel(rlat=y_idx, rlon=x_idx)
        data = data.drop_vars([var for var in data.data_vars if var not in processor._var_to_read])
        df = data.to_dataframe().reset_index()
        snow = processor._snowfall_name
        for name in ("pr", snow, "mltgl"):
            df[name] = np.clip(df[name], a_min=0, a_max=None)
        df["rf"] = df["pr"] - df[snow]
        df["alb"] = df["rsusgl"] / df["rsds"]
        mapping = config[processor._map_name]
        df.rename(columns=mapping, inplace=True)
        df.set_index("time", inplace=True)
        df = df[list(mapping.values())]
        df = df[(df.index.year >= config["start_year"]) & (df.index.year <= config["end_year"])]
        for col in ["SUBLIM", "RAIN", "BDOT", "SMELT"]:
            if col in df.columns:
                df[col] = df[col] * 60 * 60 * 24
        df["RAIN"] = np.clip(df["RAIN"], a_min=0, a_max=None)
        frames.append(df[~df.index.duplicated(keep="first")])
    return pd.concat(frames)


def _old_mar_dataframe(lat, lon):
    """The forcing DataFrame of the xarray MAR code that FEAT-003 removed (ProcessMAR._xr_to_input_dataframe).

    This is a copy of that code, for the years start_year..end_year. It is the reference for the MAR tests.
    """
    folder = config["MAR_data_path"]
    frames = []
    for year in range(config["start_year"], config["end_year"] + 1):
        xr_data = xr.open_dataset(f"{folder}/{MAR_RELEASE}-{year}.nc", engine="netcdf4", chunks="auto")
        y_idx, x_idx, _ = nearest_cell(xr_data["LAT"], xr_data["LON"], lat, lon)
        borehole_data = xr_data.isel(Y15_176=y_idx, X18_215=x_idx)
        borehole_data = borehole_data.sel(SECTOR=config["sector"])
        mapping = config["MAR_to_CFM_column_map"]
        borehole_data = borehole_data.drop_vars([var for var in borehole_data.data_vars if var not in mapping])
        borehole_df = borehole_data.to_dataframe().reset_index()
        borehole_df = borehole_df[borehole_df["OUTLAY"] == 0.0]
        borehole_df.rename(columns=mapping, inplace=True)
        borehole_df.set_index("TIME", inplace=True)
        borehole_df = borehole_df[list(mapping.values())]
        borehole_df["TSKIN"] = borehole_df["TSKIN"] + 273.15
        borehole_df["T2m"] = borehole_df["T2m"] + 273.15
        frames.append(borehole_df[~borehole_df.index.duplicated(keep="first")])
    return pd.concat(frames)


# ---------------------------------------------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------------------------------------------


def test_decode_time_days_with_half_day():
    times = extract.decode_time(np.array([10592.5, 10593.5]), "days since 1950-01-01", "standard")
    assert list(times) == [pd.Timestamp("1979-01-01 12:00"), pd.Timestamp("1979-01-02 12:00")]
    assert times.name == "time"


@pytest.mark.parametrize(
    "units, values, expected",
    [
        ("days since 1950-01-01 00:00:00.0", [10592.0], "1979-01-01"),
        ("hours since 1979-01-01", [36], "1979-01-02 12:00"),
        ("seconds since 1970-01-01 00:00:00", [86400.0], "1970-01-02"),
        ("days since 1950-1-1", np.array([0], dtype=np.int32), "1950-01-01"),
    ],
)
def test_decode_time_units(units, values, expected):
    assert extract.decode_time(np.array(values), units, "gregorian")[0] == pd.Timestamp(expected)


@pytest.mark.parametrize(
    "units, calendar",
    [
        ("days since 1950-01-01", "noleap"),
        ("days since 1950-01-01", "360_day"),
        ("days after 1950-01-01", "standard"),
        ("fortnights since 1950", None),
    ],
)
def test_decode_time_rejects(units, calendar):
    with pytest.raises(ValueError):
        extract.decode_time(np.array([1.0]), units, calendar)


def test_decode_values_fill_and_missing():
    data = np.array([1.0, -9999.0, 1e20, 2.5], dtype=np.float32)
    out = extract.decode_values(
        data, {"_FillValue": np.array([-9999.0], np.float32), "missing_value": np.array([1e20], np.float32)}
    )
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, [1.0, np.nan, np.nan, 2.5])
    assert data[1] == -9999.0  # the input is not changed


def test_decode_values_scale_offset_like_xarray():
    data = np.array([0, 10, -1], dtype=np.int16)
    attrs = {"_FillValue": np.int16(-1), "scale_factor": np.float64(0.5), "add_offset": np.float64(100.0)}
    out = extract.decode_values(data, attrs)
    np.testing.assert_array_equal(out, [100.0, 105.0, np.nan])
    decoded = xr.decode_cf(xr.Dataset({"v": ("x", data, attrs)}))["v"].values
    np.testing.assert_array_equal(out, decoded)


def test_attr_str_bytes():
    assert extract._attr_str({"units": np.bytes_(b"K")}, "units") == "K"
    assert extract._attr_str({}, "calendar", "standard") == "standard"


# ---------------------------------------------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------------------------------------------


def test_derive_columns():
    settings = extract.ExtractSettings(
        "RACMO23", "/x", {"rf": "RAIN", "sf": "BDOT", "alb": "ALBEDO", "mltgl": "SMELT", "tas": "T2m"}, 1979, 1979
    )
    assert settings.source_vars == ["pr", "sf", "rsusgl", "rsds", "mltgl", "tas"]
    raw = {
        "pr": np.float32([2e-5, 1e-5, -1e-5, np.nan]),
        "sf": np.float32([1e-5, 2e-5, -2e-5, 0]),
        "rsusgl": np.float32([80, 0, 1, 1]),
        "rsds": np.float32([100, 0, 0, 2]),
        "mltgl": np.float32([-1e-6, 1e-5, 0, 0]),
        "tas": np.float32([250, 260, 270, 280]),
    }
    cols = extract.derive_columns(raw, settings)
    assert list(cols) == ["RAIN", "BDOT", "ALBEDO", "SMELT", "T2m"]
    assert all(v.dtype == np.float32 for v in cols.values())
    rain = (np.float32(2e-5) - np.float32(1e-5)) * 60 * 60 * 24  # float32 steps, as in the old code
    np.testing.assert_array_equal(cols["RAIN"], np.float32([rain, 0, 0, np.nan]))  # clipped after rf
    np.testing.assert_array_equal(cols["BDOT"], np.float32([1e-5, 2e-5, 0, 0]) * 60 * 60 * 24)
    np.testing.assert_array_equal(cols["ALBEDO"], np.float32([0.8, np.nan, np.inf, 0.5]))
    np.testing.assert_array_equal(cols["SMELT"], np.float32([0, 1e-5, 0, 0]) * 60 * 60 * 24)
    np.testing.assert_array_equal(cols["T2m"], raw["tas"])  # no unit factor


def test_unit_factor_single_number():
    settings = extract.ExtractSettings("RACMO", "/x", {"sublgl": "SUBLIM"}, 1979, 1979, unit_factors={"SUBLIM": 86400})
    x = np.float32([0.37644228e-5, -1.1e-5])
    out = extract.derive_columns({"sublgl": x}, settings)["SUBLIM"]
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, x * np.float32(86400))


def test_time_selection_drops_out_of_range_and_duplicates():
    times = pd.DatetimeIndex(["1978-12-31", "1979-01-01", "1979-01-01", "1980-06-01", "1981-01-01"])
    np.testing.assert_array_equal(extract.time_selection(times, 1979, 1980), [False, True, False, True, False])


# ---------------------------------------------------------------------------------------------------------------
# Agreement with the old xarray code
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dataset, cls", [("RACMO", ProcessRACMO), ("RACMO23", ProcessRACMO23)])
@pytest.mark.parametrize("site", [SITE_A, SITE_B])
def test_extract_points_matches_old_code(cfg, dataset, cls, site):
    old = _old_dataframe(cls, *site)
    cell = _cell(site)
    new = extract.extract_points(dataset, [cell])[cell]
    pd.testing.assert_frame_equal(new, old, check_exact=True, check_freq=False)


@pytest.mark.parametrize("site", [SITE_A, SITE_B])
def test_extract_points_mar_matches_old_code(cfg, site):
    old = _old_mar_dataframe(*site)
    cell = _cell(site)
    new = extract.extract_points("MAR", [cell])[cell]
    pd.testing.assert_frame_equal(new, old, check_exact=True, check_freq=False)
    assert new.index.name == "TIME" and new.index[0] == pd.Timestamp("1979-01-01 12:00")


def test_mar_files_and_levels(cfg):
    settings = extract.ExtractSettings.from_config("MAR")
    assert settings.levels == {"OUTLAY": 0.0, "SECTOR": 1.0}
    assert settings.unit_factors == {} and settings.unit_offsets == {"T2m": 273.15, "TSKIN": 273.15}
    files = extract.list_files(settings)
    assert list(files) == list(MAR_MAP)
    assert [os.path.basename(p) for p in files["TI1"]] == [f"{MAR_RELEASE}-{y}.nc" for y in (1979, 1980, 1981)]
    frames, source_files = extract.forcing_for_cells("MAR", [_cell(SITE_A)], use_cache=False)
    assert len(source_files) == 3  # one file per year for all the variables
    # SECTOR = 2 is index 0 in the synthetic files, so another config value gives other values
    other, _ = extract.forcing_for_cells("MAR", [_cell(SITE_A)], use_cache=False, levels={"SECTOR": 2.0, "OUTLAY": 0.0})
    assert not other[_cell(SITE_A)]["SUBLIM"].equals(frames[_cell(SITE_A)]["SUBLIM"])
    assert other[_cell(SITE_A)]["T2m"].equals(frames[_cell(SITE_A)]["T2m"])  # TTZ has no SECTOR axis
    with pytest.raises(ValueError, match="no single SECTOR"):
        extract.forcing_for_cells("MAR", [_cell(SITE_A)], use_cache=False, levels={"SECTOR": 3.0, "OUTLAY": 0.0})
    with pytest.raises(ValueError, match="OUTLAY"):
        extract.forcing_for_cells("MAR", [_cell(SITE_A)], use_cache=False, levels={"SECTOR": 1.0})


def test_extract_points_many_cells_and_processes(cfg):
    cells = [(2, 4), (0, 0), (8, 10), (2, 4)]
    one = extract.extract_points("RACMO23", cells, processes=1)
    many = extract.extract_points("RACMO23", cells, processes=3)
    assert list(one) == [(2, 4), (0, 0), (8, 10)]
    for cell in one:
        pd.testing.assert_frame_equal(one[cell], many[cell], check_exact=True)
    assert one[(2, 4)]["T2m"].isna().sum() == 3  # one fill value in each of the 3 files
    assert one[(0, 0)].index[0] == pd.Timestamp("1979-01-01") and one[(0, 0)].index[-1] == pd.Timestamp("1981-12-31")


def test_list_files_filters_years_and_checks_coverage(cfg):
    config["end_year"] = 1984
    settings = extract.ExtractSettings.from_config("RACMO")
    files = extract.list_files(settings)
    assert [os.path.basename(p) for p in files["ts"]] == [
        "ts_ANT-12_TEST_day_19790101-19801231.nc",
        "ts_ANT-12_TEST_day_19810101-19831231.nc",
    ]
    with pytest.raises(ForcingCoverageError, match="1984"):  # BUG-003, before the long pass
        extract.check_file_years(settings, files)
    with pytest.raises(ForcingCoverageError):
        extract.forcing_for_cells("RACMO", [_cell(SITE_A)], use_cache=False)
    config["end_year"] = 1983
    extract.check_file_years(extract.ExtractSettings.from_config("RACMO"), files)
    config["start_year"] = 1981
    assert len(extract.list_files(extract.ExtractSettings.from_config("RACMO"))["ts"]) == 1


def test_list_files_missing_variable(cfg):
    settings = extract.ExtractSettings.from_config("RACMO", column_map={"nosuchvar": "X"})
    with pytest.raises(FileNotFoundError):
        extract.list_files(settings)


# ---------------------------------------------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------------------------------------------


def test_box_for_sites():
    lat, lon = _grid()
    box = extract.box_for_sites(lat, lon, [(-67.6, -65.1)], {"lat": 0.0, "lon": 0.0}, cells=[(4, 3)])
    assert box == (4, 5, 3, 4)
    box = extract.box_for_sites(lat, lon, [(-67.6, -65.1)], {"lat": 0.2, "lon": 0.3})
    y0, y1, x0, x1 = box
    inside = (lat >= -67.8) & (lat <= -67.4) & (lon >= -65.4) & (lon <= -64.8)
    ys, xs = np.nonzero(inside)
    assert (y0, y1, x0, x1) == (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
    with pytest.raises(ValueError):
        extract.box_for_sites(lat, lon, [], {"lat": 0, "lon": 0})


def test_box_contains_and_union():
    assert extract.box_contains((1, 3, 2, 5), [(1, 2), (2, 4)])
    assert not extract.box_contains((1, 3, 2, 5), [(3, 2)])
    assert extract.union_box((1, 3, 2, 5), (0, 2, 4, 9)) == (0, 3, 2, 9)


def test_cache_path(cfg):
    # next to the FEAT-002 forcing CSVs: {forcing_data_path}/<rcm_name>/<release>/
    assert extract.cache_path("RACMO") == f"{cfg['CFM_data_path']}/forcing/RACMO/ANT-12_TEST_day/cache.h5"
    config["forcing_data_path"] = "/shared/forcing"
    assert extract.cache_path("RACMO23") == "/shared/forcing/RACMO23/XPEN055.TEST.DD/cache.h5"
    config["forcing_cache_dir"] = "/elsewhere/{rcm_name}/x"
    assert extract.cache_path("RACMO23") == "/elsewhere/RACMO23/x/cache.h5"


@pytest.mark.parametrize("dataset", ["RACMO", "RACMO23", "MAR"])
def test_cache_matches_direct_extraction(cfg, dataset):
    config["forcing_cache_margin"] = {"lat": 0.1, "lon": 0.2}
    cells = [_cell(SITE_A), _cell(SITE_B)]
    direct, files = extract.forcing_for_cells(dataset, cells, use_cache=False)
    cached, _ = extract.forcing_for_cells(dataset, cells, use_cache=True, cache_sites=[SITE_A])
    for cell in cells:
        pd.testing.assert_frame_equal(cached[cell], direct[cell], check_exact=True, check_freq=False)
    assert len(files) == N_SOURCE_FILES[dataset]
    with h5py.File(extract.cache_path(dataset)) as h5:
        y0, y1, x0, x1 = extract.read_cache_meta(extract.cache_path(dataset))["box"]
        assert h5["columns/T2m"].chunks == (
            h5["time"].shape[0],
            min(extract.CACHE_CHUNK, y1 - y0),
            min(extract.CACHE_CHUNK, x1 - x0),
        )
        assert h5["columns/T2m"].dtype == np.float32


def test_cache_reuse_and_rebuild(cfg, monkeypatch):
    config["forcing_cache_margin"] = {"lat": 0.0, "lon": 0.0}
    calls = []
    real_build = extract.build_cache
    monkeypatch.setattr(extract, "build_cache", lambda *a, **k: (calls.append(a[1]), real_build(*a, **k)))
    path = extract.cache_path("RACMO23")

    a = _cell(SITE_A)
    extract.forcing_for_cells("RACMO23", [a], cache_sites=[SITE_A])
    assert len(calls) == 1
    first_box = calls[0]

    extract.forcing_for_cells("RACMO23", [a], cache_sites=[SITE_A])  # same site: no new pass
    assert len(calls) == 1

    far = (-67.2, -63.6)  # outside the box: new box contains the old box and the new site
    frames, _ = extract.forcing_for_cells("RACMO23", [_cell(far)], cache_sites=[far])
    assert len(calls) == 2
    assert extract.box_contains(calls[1], [first_box[::2], (first_box[1] - 1, first_box[3] - 1), _cell(far)])
    direct, _ = extract.forcing_for_cells("RACMO23", [_cell(far)], use_cache=False)
    pd.testing.assert_frame_equal(frames[_cell(far)], direct[_cell(far)], check_exact=True)

    source = sorted((cfg["RACMO23_data_path"] + "/" + f) for f in os.listdir(cfg["RACMO23_data_path"]))[0]
    stat = os.stat(source)
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))  # a source file changed
    try:
        extract.forcing_for_cells("RACMO23", [a], cache_sites=[SITE_A])
        assert len(calls) == 3
        assert calls[2] == calls[1]  # the rebuild keeps the old box
        config["end_year"] = 1980  # a different period
        extract.forcing_for_cells("RACMO23", [a], cache_sites=[SITE_A])
        assert len(calls) == 4
    finally:
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert not [f for f in os.listdir(os.path.dirname(path)) if ".tmp" in f]


def test_corrupt_cache_is_rebuilt(cfg):
    path = extract.cache_path("RACMO")
    os.makedirs(os.path.dirname(path))
    with open(path, "w") as f:
        f.write("not hdf5")
    frames, _ = extract.forcing_for_cells("RACMO", [_cell(SITE_A)])
    assert extract.read_cache_meta(path) is not None
    assert len(frames[_cell(SITE_A)]) == 365 + 366 + 365


# ---------------------------------------------------------------------------------------------------------------
# CSV output through ProcessRACMO
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dataset, cls", [("RACMO", ProcessRACMO), ("RACMO23", ProcessRACMO23), ("MAR", ProcessMAR)])
@pytest.mark.parametrize("use_cache", [True, False])
def test_process_writes_same_csv_as_old_code(cfg, dataset, cls, use_cache):
    old = _old_mar_dataframe(*SITE_B) if dataset == "MAR" else _old_dataframe(cls, *SITE_B)
    config["forcing_cache"] = use_cache
    processor = cls(*SITE_B)
    processor.process()
    with open(processor._save_path) as file:
        assert file.read() == old.to_csv()
    with open(forcing_json_path(processor._save_path)) as file:
        record = json.load(file)
    assert record["extraction"].startswith("h5py") and record["model_point"]["i"] == _cell(SITE_B)[0]
    assert len(record["source_files"]) == N_SOURCE_FILES[dataset]
    assert (record["extraction_cache"] is not None) == use_cache


def test_process_fast_shared_cell_and_skip(cfg):
    near_a = (SITE_A[0] + 0.001, SITE_A[1])  # the same model point as SITE_A
    processors = [ProcessRACMO23(*site) for site in (SITE_A, near_a, SITE_B)]
    assert processors[0]._save_path == processors[1]._save_path
    written = extract.process_fast(processors, use_cache=False)
    assert written == [processors[0]._save_path, processors[2]._save_path]
    assert extract.process_fast(processors, use_cache=False) == []  # existing CSVs are checked and skipped
    frame = pd.read_csv(processors[0]._save_path, index_col=0)
    assert list(frame.columns) == list(RACMO23_MAP.values()) and frame.index.name == "time"
    with pytest.raises(ValueError, match="one dataset"):
        extract.process_fast([ProcessRACMO(*SITE_A), ProcessRACMO23(*SITE_A)])


def _load_extract_forcing_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "extract_forcing.py"
    spec = importlib.util.spec_from_file_location("extract_forcing_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_forcing_script_years(cfg):
    """--end-year replaces end_year of config.yaml: the CSV name and the date check use it (FEAT-002 step 8)."""
    script = _load_extract_forcing_script()
    script.main(["--rcm", "RACMO23", "--sites", "CI-0", "--no-cache", "--processes", "1", "--end-year", "1980"])
    assert config["end_year"] == 1980
    processor = ProcessRACMO23(*SITE_A)
    folder = os.path.dirname(processor._save_path)
    names = sorted(name for name in os.listdir(folder) if name.endswith(".csv"))
    assert len(names) == 1 and names[0].endswith("_1979_1980.csv"), names
    frame = pd.read_csv(os.path.join(folder, names[0]), index_col=0, parse_dates=True)
    assert frame.index[0].year == 1979 and frame.index[-1].year == 1980
    record = json.loads(open(os.path.join(folder, names[0][:-4] + ".json")).read())
    assert record["start_year"] == 1979 and record["end_year"] == 1980


def test_extract_forcing_script_years_out_of_order(cfg):
    script = _load_extract_forcing_script()
    with pytest.raises(SystemExit):
        script.main(["--rcm", "RACMO23", "--sites", "CI-0", "--start-year", "1981", "--end-year", "1980"])
