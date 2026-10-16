"""Tests for model points on synthetic RCM grids (FEAT-002, BUG-002). No luna data is read."""

import numpy as np
import pytest
import xarray as xr

from cfm_config import config, get_site
from force.grid import great_circle_km
from force.model_point import (
    clear_cache,
    model_point,
    model_point_at,
    model_point_label,
    model_point_tag,
    rcm_release,
    shared_model_points,
)


def _write_grid(folder, file_name, lat_name="lat", lon_name="lon", dims=("rlat", "rlon")):
    """A 0.1 x 0.25 degree regular grid around the Antarctic Peninsula sites."""
    lat1d = np.arange(-69.0, -66.0, 0.1)
    lon1d = np.arange(-66.0, -60.0, 0.25)
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)
    ds = xr.Dataset(
        {"ts": (dims, np.zeros(lat2d.shape))},
        coords={lat_name: (dims, lat2d), lon_name: (dims, lon2d)},
    )
    ds.to_netcdf(folder / file_name)
    return lat2d, lon2d


@pytest.fixture
def racmo23_grid(tmp_path, monkeypatch):
    lat2d, lon2d = _write_grid(tmp_path, "ts.KNMI-1979.XPEN055.RACMO2.4p1_v5_historical.DD.nc")
    _write_grid(tmp_path, "ts.KNMI-1980.XPEN055.RACMO2.4p1_v5_historical.DD.nc")
    (tmp_path / "_READBEFOREUSE.txt").write_text("not a data file")
    monkeypatch.setitem(config, "RACMO23_data_path", str(tmp_path))
    clear_cache()
    yield lat2d, lon2d
    clear_cache()


def test_model_point_at_returns_nearest_cell_and_distance(racmo23_grid):
    lat2d, lon2d = racmo23_grid
    mp = model_point_at(-67.5, -63.336, "RACMO23")
    assert mp.rcm_name == "RACMO23"
    assert mp.release == "XPEN055.RACMO2.4p1_v5_historical.DD"
    assert mp.method == "nearest"
    assert (mp.cell_lat, mp.cell_lon) == (lat2d[mp.i, mp.j], lon2d[mp.i, mp.j])
    assert mp.distance_km == pytest.approx(float(great_circle_km(mp.cell_lat, mp.cell_lon, -67.5, -63.336)))
    all_distances = great_circle_km(lat2d, lon2d, -67.5, -63.336)
    assert mp.distance_km == pytest.approx(all_distances.min())
    assert mp.tag == f"RACMO23_{mp.i}_{mp.j}"


def test_model_point_by_site_name(racmo23_grid):
    site = get_site("CI-0")
    assert model_point("CI-0", "RACMO23") == model_point_at(site.lat, site.lon, "RACMO23")
    assert model_point(site, "RACMO23") == model_point_at(site.lat, site.lon, "RACMO23")


def test_sites_in_one_cell_share_a_model_point(racmo23_grid):
    # J409 and J109 are 0.003 degrees of longitude apart
    j409, j109 = model_point("J409", "RACMO23"), model_point("J109", "RACMO23")
    assert j409.tag == j109.tag
    groups = shared_model_points([get_site("J409"), get_site("J109"), get_site("CI-0")], "RACMO23")
    assert sorted(len(members) for members in groups.values()) == [1, 2]
    label = model_point_label("J409", groups)
    assert label.startswith(f"J409: {j409.tag}, ") and label.endswith("same cell as J109")
    assert "same cell" not in model_point_label("CI-0", groups)


def test_mar_grid(tmp_path, monkeypatch):
    lat2d, lon2d = _write_grid(tmp_path, "MARv3.14.3-27.5km-daily-ERA5-1979.nc", "LAT", "LON", ("Y", "X"))
    monkeypatch.setitem(config, "MAR_data_path", str(tmp_path))
    clear_cache()
    mp = model_point_at(-66.403, -63.376, "MAR")
    clear_cache()
    assert mp.release == "MARv3.14.3-27.5km-daily-ERA5"
    assert mp.distance_km == pytest.approx(great_circle_km(lat2d, lon2d, -66.403, -63.376).min())


def test_racmo_ant12_release_name(tmp_path, monkeypatch):
    name = "ts_ANT-12_ERA5_evaluation_r1i1p1f1_UU-IMAU_RACMO24P-NN_v1-r1_day_19790101-19801231.nc"
    _write_grid(tmp_path, name)
    monkeypatch.setitem(config, "RACMO_data_path", str(tmp_path))
    clear_cache()
    mp = model_point_at(-66.403, -63.376, "RACMO")
    clear_cache()
    assert mp.release == "ANT-12_ERA5_evaluation_r1i1p1f1_UU-IMAU_RACMO24P-NN_v1-r1_day"


def test_two_releases_in_one_folder_is_an_error(tmp_path, monkeypatch):
    _write_grid(tmp_path, "ts.KNMI-1979.XPEN055.RACMO2.4p1_v5_historical.DD.nc")
    _write_grid(tmp_path, "ts.KNMI-1979.XPEN055.RACMO2.4p2_v6_historical.DD.nc")
    monkeypatch.setitem(config, "RACMO23_data_path", str(tmp_path))
    clear_cache()
    with pytest.raises(ValueError, match="More than one RACMO23 release"):
        model_point_at(-66.403, -63.376, "RACMO23")
    clear_cache()


def test_release_without_opening_files(tmp_path):
    (tmp_path / "ts.KNMI-1979.XPEN055.RACMO2.4p1_v5_historical.DD.nc").write_text("not netCDF")
    clear_cache()
    assert rcm_release("RACMO23", str(tmp_path)) == "XPEN055.RACMO2.4p1_v5_historical.DD"
    clear_cache()


def test_unknown_rcm():
    with pytest.raises(ValueError, match="Unknown rcm_name"):
        model_point_at(-66.403, -63.376, "NOPE")


def test_tag():
    assert model_point_tag("RACMO", 261, 126) == "RACMO_261_126"
