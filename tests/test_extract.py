"""Tests for the fast RACMO extraction and the forcing cache (FEAT-003), on small synthetic files.

The synthetic files copy the layout of the real data: ANT-12 (RACMO) files of several years, (time, rlat, rlon),
_FillValue and missing_value 1e20, times at 12:00; XPEN055 (RACMO23) files of one year, (time, height, rlat, rlon),
_FillValue -9999, times at 00:00. The main check is that extract_points() gives the same DataFrame as the old xarray
code in src/force/process.py. Run from the repo root.
"""

import os

import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cfm_config import config
from force import extract
from force.process import ProcessRACMO, ProcessRACMO23

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


@pytest.fixture
def cfg(monkeypatch, racmo_dir, racmo23_dir, tmp_path):
    """The shared config dict, with the synthetic data paths and a temporary CFM_data_path."""
    for key, value in {
        "RACMO_data_path": str(racmo_dir),
        "RACMO23_data_path": str(racmo23_dir),
        "RACMO_to_CFM_column_map": RACMO_MAP,
        "RACMO23_to_CFM_column_map": RACMO23_MAP,
        "RACMO_additional_vars": ["pr", "rsusgl"],
        "start_year": 1979,
        "end_year": 1981,
        "CFM_data_path": str(tmp_path / "CFM_data"),
        "forcing_processes": 1,
    }.items():
        monkeypatch.setitem(config, key, value)
    for key in ("RACMO_to_CFM_unit_factors", "forcing_cache_dir", "forcing_cache_margin", "forcing_cache"):
        monkeypatch.delitem(config, key, raising=False)
    return config


def _old_dataframe(cls, lat, lon):
    """The forcing DataFrame from the old xarray code, as ProcessBase.process builds it."""
    processor = cls(lat, lon)
    return pd.concat([processor._xr_to_input_dataframe(ds) for ds in processor._read_data()])


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
@pytest.mark.parametrize("site", [(-67.6, -65.1), (-67.78, -65.0)])  # the second site is at the cell with NaN
def test_extract_points_matches_old_code(cfg, dataset, cls, site):
    old = _old_dataframe(cls, *site)
    lat, lon = _grid()
    cell = extract.nearest_cell(lat, lon, *site)[:2]
    new = extract.extract_points(dataset, [cell])[cell]
    pd.testing.assert_frame_equal(new, old, check_exact=True, check_freq=False)


def test_extract_points_many_cells_and_processes(cfg):
    cells = [(2, 4), (0, 0), (8, 10), (2, 4)]
    one = extract.extract_points("RACMO23", cells, processes=1)
    many = extract.extract_points("RACMO23", cells, processes=3)
    assert list(one) == [(2, 4), (0, 0), (8, 10)]
    for cell in one:
        pd.testing.assert_frame_equal(one[cell], many[cell], check_exact=True)
    assert one[(2, 4)]["T2m"].isna().sum() == 3  # one fill value in each of the 3 files
    assert one[(0, 0)].index[0] == pd.Timestamp("1979-01-01") and one[(0, 0)].index[-1] == pd.Timestamp("1981-12-31")


def test_list_files_filters_years_and_warns(cfg, caplog):
    config["end_year"] = 1984
    settings = extract.ExtractSettings.from_config("RACMO")
    files = extract.list_files(settings)
    assert [os.path.basename(p) for p in files["ts"]] == [
        "ts_ANT-12_TEST_day_19790101-19801231.nc",
        "ts_ANT-12_TEST_day_19810101-19831231.nc",
    ]
    assert "1984" in caplog.text
    config["start_year"] = 1981
    assert len(extract.list_files(extract.ExtractSettings.from_config("RACMO"))["ts"]) == 1


def test_list_files_missing_variable(cfg):
    settings = extract.ExtractSettings.from_config("RACMO", column_map={"nosuchvar": "X"})
    with pytest.raises(FileNotFoundError):
        extract.list_files(settings)


# ---------------------------------------------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------------------------------------------


def test_nearest_cell_old_rule_and_distance():
    lat, lon = _grid()
    i, j, km = extract.nearest_cell(lat, lon, lat[4, 3], lon[4, 3])
    assert (i, j) == (4, 3) and km == pytest.approx(0, abs=1e-9)
    i, j, km = extract.nearest_cell(lat, lon, lat[4, 3] + 0.01, lon[4, 3])
    assert (i, j) == (4, 3) and km == pytest.approx(1.112, abs=1e-3)


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
    assert extract.cache_path("RACMO") == f"{cfg['CFM_data_path']}/forcing/RACMO/cache.h5"
    config["forcing_cache_dir"] = "/elsewhere/{rcm_name}/x"
    assert extract.cache_path("RACMO23") == "/elsewhere/RACMO23/x/cache.h5"


@pytest.mark.parametrize("dataset", ["RACMO", "RACMO23"])
def test_cache_matches_direct_extraction(cfg, dataset):
    config["forcing_cache_margin"] = {"lat": 0.1, "lon": 0.2}
    sites = {"a": (-67.6, -65.1), "b": (-67.78, -65.0)}
    direct = extract.forcing_for_sites(dataset, sites, use_cache=False)
    cached = extract.forcing_for_sites(dataset, sites, use_cache=True)
    for name in sites:
        pd.testing.assert_frame_equal(cached[name], direct[name], check_exact=True, check_freq=False)
    with h5py.File(extract.cache_path(dataset)) as h5:
        y0, y1, x0, x1 = extract.read_cache_meta(extract.cache_path(dataset))["box"]
        assert h5["columns/T2m"].chunks == (h5["time"].shape[0], min(16, y1 - y0), min(16, x1 - x0))
        assert h5["columns/T2m"].dtype == np.float32


def test_cache_reuse_and_rebuild(cfg, monkeypatch):
    config["forcing_cache_margin"] = {"lat": 0.0, "lon": 0.0}
    calls = []
    real_build = extract.build_cache
    monkeypatch.setattr(extract, "build_cache", lambda *a, **k: (calls.append(a[1]), real_build(*a, **k)))
    path = extract.cache_path("RACMO23")

    extract.forcing_for_sites("RACMO23", {"a": (-67.6, -65.1)})
    assert len(calls) == 1
    first_box = calls[0]

    extract.forcing_for_sites("RACMO23", {"a": (-67.6, -65.1)})  # same site: no new pass
    assert len(calls) == 1

    far = (-67.2, -63.6)  # outside the box: new box contains the old box and the new site
    frames = extract.forcing_for_sites("RACMO23", {"far": far})
    assert len(calls) == 2
    assert extract.box_contains(calls[1], [first_box[::2], (first_box[1] - 1, first_box[3] - 1)])
    lat, lon = _grid()
    direct = extract.forcing_for_sites("RACMO23", {"far": far}, use_cache=False)
    pd.testing.assert_frame_equal(frames["far"], direct["far"], check_exact=True)

    source = sorted((cfg["RACMO23_data_path"] + "/" + f) for f in os.listdir(cfg["RACMO23_data_path"]))[0]
    stat = os.stat(source)
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))  # a source file changed
    try:
        extract.forcing_for_sites("RACMO23", {"a": (-67.6, -65.1)})
        assert len(calls) == 3
        assert calls[2] == calls[1]  # the rebuild keeps the old box
        config["end_year"] = 1980  # a different period
        extract.forcing_for_sites("RACMO23", {"a": (-67.6, -65.1)})
        assert len(calls) == 4
    finally:
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert not [f for f in os.listdir(os.path.dirname(path)) if ".tmp" in f]


def test_corrupt_cache_is_rebuilt(cfg):
    path = extract.cache_path("RACMO")
    os.makedirs(os.path.dirname(path))
    with open(path, "w") as f:
        f.write("not hdf5")
    frames = extract.forcing_for_sites("RACMO", {"a": (-67.6, -65.1)})
    assert extract.read_cache_meta(path) is not None
    assert len(frames["a"]) == 365 + 366 + 365


# ---------------------------------------------------------------------------------------------------------------
# CSV output and the opt-in switch in ProcessRACMO
# ---------------------------------------------------------------------------------------------------------------


def test_write_forcing_csvs_skips_existing(cfg, tmp_path):
    targets = {str(tmp_path / "a.csv"): (-67.6, -65.1), str(tmp_path / "b.csv"): (-67.78, -65.0)}
    assert sorted(extract.write_forcing_csvs("RACMO", targets, use_cache=False)) == sorted(targets)
    assert extract.write_forcing_csvs("RACMO", targets, use_cache=False) == []
    frame = pd.read_csv(tmp_path / "a.csv", index_col=0)
    assert list(frame.columns) == list(RACMO_MAP.values()) and frame.index.name == "time"
