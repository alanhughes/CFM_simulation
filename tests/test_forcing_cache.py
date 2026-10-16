"""Tests for the forcing cache path, its JSON file and the date check (FEAT-002, BUG-003). No luna data is read."""

import json
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cfm_config import config
from force.cache import ForcingCoverageError, check_date_coverage, forcing_csv_path, forcing_json_path, forcing_root
from force.grid import nearest_cell
from force.model_point import ModelPoint, clear_cache
from force.process import ProcessBase

RELEASE = "XPEN055.RACMO2.4p1_v5_historical.DD"


def _mp(**changes) -> ModelPoint:
    values = dict(rcm_name="RACMO23", release=RELEASE, i=147, j=169, cell_lat=-66.4, cell_lon=-63.4, distance_km=1.2)
    values.update(changes)
    return ModelPoint(**values)


def test_forcing_csv_path(monkeypatch):
    monkeypatch.setitem(config, "CFM_data_path", "/data")
    monkeypatch.delitem(config, "forcing_data_path", raising=False)
    assert forcing_root() == "/data/forcing"
    assert forcing_csv_path(_mp(), 1979, 2024) == f"/data/forcing/RACMO23/{RELEASE}/147_169_1979_2024.csv"
    assert forcing_json_path(forcing_csv_path(_mp(), 1979, 2024)).endswith("/147_169_1979_2024.json")


def test_forcing_path_from_config(monkeypatch):
    monkeypatch.setitem(config, "CFM_data_path", "/data/ab871c8")
    monkeypatch.setitem(config, "forcing_data_path", "/shared/forcing")
    monkeypatch.setitem(config, "start_year", 1980)
    monkeypatch.setitem(config, "end_year", 1990)
    assert forcing_csv_path(_mp(i=1, j=2)) == f"/shared/forcing/RACMO23/{RELEASE}/1_2_1980_1990.csv"


def test_forcing_path_changes_with_cell_and_release():
    paths = {forcing_csv_path(mp, 1979, 2024) for mp in [_mp(), _mp(j=170), _mp(release="other"), _mp(rcm_name="MAR")]}
    assert len(paths) == 4


def test_date_coverage_complete():
    check_date_coverage(pd.date_range("1979-01-01 12:00", "2024-12-31 12:00", freq="D"), 1979, 2024, "RACMO23")


def test_date_coverage_stops_early():
    """BUG-003: the ANT-12 data stops at 2023-12-31, but end_year is 2024."""
    dates = pd.date_range("1979-01-01 12:00", "2023-12-31 12:00", freq="D")
    expected = r"RACMO \(ANT-12\): no data for the years 2024.*stops at 2023-12-31"
    with pytest.raises(ForcingCoverageError, match=expected):
        check_date_coverage(dates, 1979, 2024, "RACMO (ANT-12)")


def test_date_coverage_gap_and_late_start():
    dates = pd.date_range("1979-03-01", "1989-12-31", freq="D").append(pd.date_range("1992-01-01", "1995-12-31"))
    with pytest.raises(ForcingCoverageError) as error:
        check_date_coverage(dates, 1979, 1995, "MAR")
    assert "1990-1991" in str(error.value) and "after January 1979" in str(error.value)


# ---- ProcessBase with a fake RCM on a synthetic grid ----


def _grid():
    lat1d = np.arange(-69.0, -66.0, 0.1)
    lon1d = np.arange(-66.0, -60.0, 0.25)
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)
    return lat2d, lon2d


def _year_dataset(year: int, lat2d, lon2d) -> xr.Dataset:
    time = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    ts = np.broadcast_to(np.arange(len(time))[:, None, None], (len(time),) + lat2d.shape).astype(float)
    return xr.Dataset(
        {"ts": (("time", "rlat", "rlon"), ts)},
        coords={"time": time, "lat": (("rlat", "rlon"), lat2d), "lon": (("rlat", "rlon"), lon2d)},
    )


class FakeProcess(ProcessBase):
    _rcm_name = "RACMO23"
    _map_name = "RACMO23_to_CFM_column_map"
    _unit_factors = {"TSKIN": "none"}
    years = None

    def process(self) -> None:
        super().process(self._save_path)

    def _read_data(self):
        lat2d, lon2d = _grid()
        return [_year_dataset(year, lat2d, lon2d) for year in self.years]

    def _xr_to_input_dataframe(self, xr_data):
        y_idx, x_idx, _ = nearest_cell(xr_data["lat"], xr_data["lon"], self._borehole_lat, self._borehole_lon)
        self._check_cell(y_idx, x_idx)
        df = xr_data.isel(rlat=y_idx, rlon=x_idx)["ts"].to_dataframe()[["ts"]]
        return df.rename(columns={"ts": "TSKIN"})


@pytest.fixture
def fake_rcm(tmp_path, monkeypatch):
    data = tmp_path / "rcm"
    data.mkdir()
    lat2d, lon2d = _grid()
    _year_dataset(1979, lat2d, lon2d).to_netcdf(data / f"ts.KNMI-1979.{RELEASE}.nc")
    monkeypatch.setitem(config, "RACMO23_data_path", str(data))
    monkeypatch.setitem(config, "CFM_data_path", str(tmp_path / "CFM_data"))
    monkeypatch.delitem(config, "forcing_data_path", raising=False)
    monkeypatch.setitem(config, "start_year", 1979)
    monkeypatch.setitem(config, "end_year", 1981)
    clear_cache()
    yield tmp_path
    clear_cache()


def test_process_writes_csv_and_json(fake_rcm):
    process = FakeProcess(-66.403, -63.376)
    process.years = [1979, 1980, 1981]
    process.process()

    mp = process.model_point
    csv_path = fake_rcm / "CFM_data" / "forcing" / "RACMO23" / RELEASE / f"{mp.i}_{mp.j}_1979_1981.csv"
    assert csv_path.exists() and not os.path.exists(f"{csv_path}.tmp")
    record = json.loads(csv_path.with_suffix(".json").read_text())
    assert record["model_point"]["tag"] == mp.tag and record["model_point"]["release"] == RELEASE
    assert record["first_date"].startswith("1979-01-01") and record["last_date"].startswith("1981-12-31")
    assert record["requested_location"] == {"lat": -66.403, "lon": -63.376, "distance_km": mp.distance_km}
    assert record["column_map"] == config["RACMO23_to_CFM_column_map"]
    assert record["unit_factors"] == {"TSKIN": "none"}
    assert record["n_rows"] == 365 + 366 + 365

    # a second site in the same cell uses the same CSV
    neighbour = FakeProcess(-66.403 + 0.001, -63.376)
    assert neighbour._save_path == str(csv_path)


def test_process_stops_if_a_year_is_missing(fake_rcm):
    process = FakeProcess(-66.403, -63.376)
    process.years = [1979, 1980]  # end_year is 1981
    with pytest.raises(ForcingCoverageError, match="no data for the years 1981"):
        process.process()
    assert not os.path.exists(process._save_path)
    assert not os.path.exists(f"{process._save_path}.tmp")


def test_existing_csv_is_checked(fake_rcm):
    process = FakeProcess(-66.403, -63.376)
    process.years = [1979, 1980, 1981]
    process.process()
    # a shorter year range has a different file name, so it is a new extraction ...
    config["end_year"] = 1982
    process.years = [1979, 1980, 1981]
    with pytest.raises(ForcingCoverageError, match="1982"):
        process.process()
    # ... and an old CSV without JSON in the cache is checked from its own dates
    config["end_year"] = 1981
    os.remove(forcing_json_path(process._save_path))
    process.process()
    pd.DataFrame({"TSKIN": [1.0]}, index=pd.to_datetime(["1979-01-01"])).to_csv(process._save_path)
    with pytest.raises(ForcingCoverageError, match="stops at 1979-01-01"):
        process.process()


def test_cell_check(fake_rcm):
    process = FakeProcess(-66.403, -63.376)
    mp = process.model_point
    with pytest.raises(ValueError, match="do not use one grid"):
        process._check_cell(mp.i + 1, mp.j)
