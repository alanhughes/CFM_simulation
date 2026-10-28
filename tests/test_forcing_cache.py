"""Tests for the forcing cache path, its JSON file and the date check (FEAT-002, BUG-003). No luna data is read."""

import errno
import hashlib
import json
import logging
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cfm_config import config, share_available
from force.cache import (
    ForcingCoverageError,
    check_date_coverage,
    find_forcing_csv,
    forcing_csv_path,
    forcing_json_path,
    forcing_root,
)
from force.extract import process_fast
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


def _forcing(years) -> pd.DataFrame:
    """Daily forcing of the years, as process_fast gives it to save_forcing."""
    time = pd.DatetimeIndex([]).append([pd.date_range(f"{year}-01-01", f"{year}-12-31") for year in years])
    return pd.DataFrame({"TSKIN": np.arange(len(time), dtype=float)}, index=pd.Index(time, name="time"))


class FakeProcess(ProcessBase):
    _rcm_name = "RACMO23"
    _map_name = "RACMO23_to_CFM_column_map"
    _unit_factors = {"TSKIN": "none"}

    def _read_data(self):
        raise NotImplementedError


@pytest.fixture
def fake_rcm(tmp_path, monkeypatch):
    # the model point is found from the coordinates of this file
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


def test_save_forcing_writes_csv_and_json(fake_rcm):
    process = FakeProcess(-66.403, -63.376)
    df = _forcing([1979, 1980, 1981])
    process.save_forcing(process._save_path, df, ["/data/b.nc", "/data/a.nc"], extra={"extraction": "test"})

    mp = process.model_point
    csv_path = fake_rcm / "CFM_data" / "forcing" / "RACMO23" / RELEASE / f"{mp.i}_{mp.j}_1979_1981.csv"
    assert csv_path.exists() and not os.path.exists(f"{csv_path}.tmp")
    assert csv_path.read_text() == df.to_csv()
    record = json.loads(csv_path.with_suffix(".json").read_text())
    assert record["model_point"]["tag"] == mp.tag and record["model_point"]["release"] == RELEASE
    assert record["first_date"].startswith("1979-01-01") and record["last_date"].startswith("1981-12-31")
    assert record["requested_location"] == {"lat": -66.403, "lon": -63.376, "distance_km": mp.distance_km}
    assert record["column_map"] == config["RACMO23_to_CFM_column_map"]
    assert record["unit_factors"] == {"TSKIN": "none"}
    assert record["n_rows"] == 365 + 366 + 365
    assert record["source_files"] == ["a.nc", "b.nc"] and record["extraction"] == "test"
    # the JSON file is written from the temporary CSV, before the rename to the final name
    assert record["csv_sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()

    # a second site in the same cell uses the same CSV
    neighbour = FakeProcess(-66.403 + 0.001, -63.376)
    assert neighbour._save_path == str(csv_path)


def test_save_forcing_stops_if_a_year_is_missing(fake_rcm):
    process = FakeProcess(-66.403, -63.376)
    with pytest.raises(ForcingCoverageError, match="no data for the years 1981"):
        process.save_forcing(process._save_path, _forcing([1979, 1980]), [])  # end_year is 1981
    assert not os.path.exists(process._save_path)
    assert not os.path.exists(f"{process._save_path}.tmp")


def test_save_forcing_writes_a_temporary_name_first(fake_rcm, monkeypatch):
    """A run that stops before the end leaves no CSV that looks complete."""
    import force.process

    def stop(*args, **kwargs):
        raise RuntimeError("stopped")

    process = FakeProcess(-66.403, -63.376)
    monkeypatch.setattr(force.process, "write_forcing_json", stop)
    with pytest.raises(RuntimeError, match="stopped"):
        process.save_forcing(process._save_path, _forcing([1979, 1980, 1981]), [])
    assert os.path.exists(f"{process._save_path}.tmp")
    assert not os.path.exists(process._save_path)
    assert not os.path.exists(forcing_json_path(process._save_path))


def test_existing_csv_is_checked_and_skipped(fake_rcm):
    """process_fast checks an existing CSV and does not read the RCM files."""
    process = FakeProcess(-66.403, -63.376)
    process.save_forcing(process._save_path, _forcing([1979, 1980, 1981]), [])
    mtime = os.stat(process._save_path).st_mtime_ns
    assert process_fast([process]) == []
    assert os.stat(process._save_path).st_mtime_ns == mtime
    # a longer year range has a different file name, so it is a new extraction ...
    config["end_year"] = 1982
    assert not os.path.exists(process._save_path)
    with pytest.raises(ForcingCoverageError, match="1982"):
        process.save_forcing(process._save_path, _forcing([1979, 1980, 1981]), [])
    # ... and an old CSV without JSON in the cache is checked from its own dates
    config["end_year"] = 1981
    os.remove(forcing_json_path(process._save_path))
    assert process_fast([process]) == []
    pd.DataFrame({"TSKIN": [1.0]}, index=pd.to_datetime(["1979-01-01"])).to_csv(process._save_path)
    with pytest.raises(ForcingCoverageError, match="stops at 1979-01-01"):
        process_fast([process])


def test_existing_json_dates_are_checked(fake_rcm):
    """The dates of an existing CSV come from its JSON file, if it has one (BUG-003)."""
    process = FakeProcess(-66.403, -63.376)
    process.save_forcing(process._save_path, _forcing([1979, 1980, 1981]), [])
    json_path = forcing_json_path(process._save_path)
    record = json.loads(open(json_path).read())
    record["last_date"] = "1980-12-31"
    with open(json_path, "w") as file:
        json.dump(record, file)
    with pytest.raises(ForcingCoverageError, match="no data for the years 1981"):
        process._check_existing(process._save_path, 1979, 1981, "RACMO23")


def test_cell_check(fake_rcm):
    process = FakeProcess(-66.403, -63.376)
    mp = process.model_point
    with pytest.raises(ValueError, match="do not use one grid"):
        process._check_cell(mp.i + 1, mp.j)


# FEAT-008: find_forcing_csv reads the share copy when the CSV is not local


@pytest.fixture
def share_cache(tmp_path, monkeypatch):
    """A local and a share forcing cache in tmp_path, not made. Returns (local csv path, share csv path)."""
    monkeypatch.setitem(config, "CFM_data_path", str(tmp_path / "local"))
    monkeypatch.setitem(config, "forcing_data_path", str(tmp_path / "local" / "forcing"))
    monkeypatch.setitem(config, "CFM_share_data_path", str(tmp_path / "share"))
    local = forcing_csv_path(_mp(), 1979, 2024)
    share = str(tmp_path / "share" / "forcing" / "RACMO23" / RELEASE / "147_169_1979_2024.csv")
    return local, share


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


def test_find_forcing_csv_local(share_cache):
    local, share = share_cache
    _touch(local)
    _touch(share)
    assert find_forcing_csv(_mp(), 1979, 2024) == local


def test_find_forcing_csv_share_only(share_cache):
    local, share = share_cache
    _touch(share)
    assert find_forcing_csv(_mp(), 1979, 2024) == share
    # the path of a write stays local
    assert forcing_csv_path(_mp(), 1979, 2024) == local


def test_find_forcing_csv_neither(share_cache):
    local, share = share_cache
    os.makedirs(os.path.dirname(share))
    assert find_forcing_csv(_mp(), 1979, 2024) == local


def test_find_forcing_csv_share_not_set(share_cache, monkeypatch):
    local, share = share_cache
    _touch(share)
    monkeypatch.delitem(config, "CFM_share_data_path")
    assert find_forcing_csv(_mp(), 1979, 2024) == local


@pytest.fixture
def share_stats(monkeypatch):
    """Count the os.stat calls for paths under the share. Set .fail to a folder prefix to make those calls give EIO."""

    class Stats:
        calls: list = []
        prefix: str = ""
        fail: bool = False

    stats = Stats()
    stats.calls = []
    real_stat = os.stat

    def counting_stat(path, *args, **kwargs):
        if stats.prefix and str(path).startswith(stats.prefix):
            stats.calls.append(str(path))
            if stats.fail:
                raise OSError(errno.EIO, "Input/output error")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", counting_stat)
    return stats


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING and "not available" in r.message]


def test_find_forcing_csv_share_not_available(share_cache, share_stats, caplog):
    local, share = share_cache
    _touch(share)
    share_stats.prefix, share_stats.fail = str(config["CFM_share_data_path"]), True
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            assert find_forcing_csv(_mp(), 1979, 2024) == local
    assert len(_warnings(caplog)) == 1
    assert len(share_stats.calls) == 1  # the share folder, one time
    assert not share_available()
    # a local CSV is still found while the share is down
    _touch(local)
    assert find_forcing_csv(_mp(), 1979, 2024) == local
    assert len(share_stats.calls) == 1


def test_find_forcing_csv_share_file_fails_later(share_cache, share_stats, caplog):
    local, share = share_cache
    _touch(share)
    share_stats.prefix = str(config["CFM_share_data_path"])
    assert share_available()
    assert len(share_stats.calls) == 1
    share_stats.fail = True
    # the share folder was found, then the share file gives an error
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            assert find_forcing_csv(_mp(), 1979, 2024) == local
    assert len(_warnings(caplog)) == 1
    assert len(share_stats.calls) == 2  # no probe after the error
    assert not share_available()


def test_find_forcing_csv_forcing_data_path_outside_data_path(share_cache, tmp_path, monkeypatch):
    """The share path is relative to forcing_root(), whatever its value."""
    monkeypatch.setitem(config, "forcing_data_path", str(tmp_path / "elsewhere" / "cache"))
    local = forcing_csv_path(_mp(), 1979, 2024)
    assert local.startswith(str(tmp_path / "elsewhere" / "cache"))
    share = str(tmp_path / "share" / "forcing" / "RACMO23" / RELEASE / "147_169_1979_2024.csv")
    _touch(share)
    assert find_forcing_csv(_mp(), 1979, 2024) == share


def test_process_fast_csv_only_on_share(fake_rcm, monkeypatch):
    """A CSV that is only on the share is checked and is not extracted again (FEAT-008)."""
    import force.extract

    process = FakeProcess(-66.403, -63.376)
    process.save_forcing(process._save_path, _forcing([1979, 1980, 1981]), [])
    local = process._save_path
    share = fake_rcm / "share" / "forcing" / os.path.relpath(local, forcing_root())
    os.makedirs(os.path.dirname(share))
    for path in (local, forcing_json_path(local)):
        os.replace(path, os.path.join(os.path.dirname(share), os.path.basename(path)))
    monkeypatch.setitem(config, "CFM_share_data_path", str(fake_rcm / "share"))

    def no_extraction(*args, **kwargs):
        raise AssertionError("the forcing was extracted again")

    checked = []
    real_check = ProcessBase._check_existing

    def recording_check(path, *args):
        checked.append(path)
        return real_check(path, *args)

    monkeypatch.setattr(force.extract, "list_files", no_extraction)
    monkeypatch.setattr(force.extract, "forcing_for_cells", no_extraction)
    monkeypatch.setattr(FakeProcess, "_check_existing", staticmethod(recording_check))
    assert process_fast([process]) == []
    assert checked == [str(share)]
    assert not os.path.exists(local)

    # a share CSV with a bad date range stops with ForcingCoverageError
    share_json = forcing_json_path(str(share))
    with open(share_json) as file:
        record = json.load(file)
    with open(share_json, "w") as file:
        json.dump({**record, "last_date": "1980-12-31"}, file)
    with pytest.raises(ForcingCoverageError, match="1981"):
        process_fast([process])
    with open(share_json, "w") as file:
        json.dump(record, file)

    # with overwrite the CSV is extracted again and written to the local path; the share CSV does not change
    share_bytes = share.read_bytes()
    mp = process.model_point
    monkeypatch.setattr(force.extract, "list_files", lambda settings: {"2000": ["a.nc"]})
    monkeypatch.setattr(force.extract, "read_grid", lambda path, spec: _grid())
    monkeypatch.setattr(
        force.extract,
        "forcing_for_cells",
        lambda *args, **kwargs: ({(mp.i, mp.j): _forcing([1979, 1980, 1981])}, ["a.nc"]),
    )
    assert process_fast([process], overwrite=True, use_cache=False, cache_sites=[]) == [local]
    assert os.path.exists(local)
    assert share.read_bytes() == share_bytes
