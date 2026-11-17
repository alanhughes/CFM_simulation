"""Tests for RCM indexes (FEAT-012): an index folder gives the same release, model points and years as its data
folder, and the forcing is never made from an index. Synthetic grids only; no luna data is read."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from cfm_config import config
from force.cache import forcing_csv_path
from force.model_point import clear_cache, model_point, model_point_at, rcm_release
from force.process import ProcessRACMO
from force.rcm_index import GRID_FILE, INDEX_FILE, is_rcm_index, match_data_files
from force.years import data_days, run_years

GROUPS = [("19790101", "19801231"), ("19810101", "20101231"), ("20110101", "20231231")]
NAME = "{var}_ANT-12_TEST_day_{first}-{last}.nc"
RELEASE = "ANT-12_TEST_day"

_spec = importlib.util.spec_from_file_location(
    "make_rcm_index", Path(__file__).resolve().parents[1] / "scripts" / "make_rcm_index.py"
)
make_rcm_index = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_rcm_index)


@pytest.fixture(autouse=True)
def _clear():
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def racmo(tmp_path, monkeypatch):
    """A RACMO data folder: one real grid file (the first name in sort order), empty files with the other names, and a
    file that is not data. The column map gives the source variables tas, pr and prsn."""
    folder = tmp_path / "day"
    folder.mkdir()
    names = sorted(NAME.format(var=v, first=a, last=b) for v in ("tas", "pr", "prsn") for a, b in GROUPS)
    lon2d, lat2d = np.meshgrid(np.arange(-66.0, -60.0, 0.25), np.arange(-69.0, -66.0, 0.1))
    dims = ("rlat", "rlon")
    xr.Dataset({"tas": (dims, np.zeros(lat2d.shape))}, coords={"lat": (dims, lat2d), "lon": (dims, lon2d)}).to_netcdf(
        folder / names[0]
    )
    for name in names[1:]:
        (folder / name).touch()
    (folder / "README.txt").write_text("not data")
    (folder / ".tas_hidden_19790101-20231231.nc").touch()
    monkeypatch.setitem(config, "RACMO_data_path", str(folder))
    monkeypatch.setitem(config, "RACMO_to_CFM_column_map", {"tas": "T2m", "rf": "RAIN"})
    monkeypatch.setitem(config, "start_year", 1979)
    monkeypatch.setitem(config, "end_year", 2024)
    monkeypatch.setitem(config, "spin_end_year", 1989)
    return folder


def _index(racmo, tmp_path, monkeypatch):
    out = tmp_path / "rcm_index"
    make_rcm_index.main(["--rcm", "RACMO", "--out", str(out)])
    clear_cache()
    monkeypatch.setitem(config, "RACMO_data_path", str(out))
    return out


def test_index_gives_the_same_model_points_release_and_years(racmo, tmp_path, monkeypatch):
    folder = {"release": rcm_release("RACMO"), "days": data_days("RACMO"), "years": run_years("RACMO")}
    folder["points"] = {name: model_point(name, "RACMO") for name in ("CI-0", "CI-22")}
    out = _index(racmo, tmp_path, monkeypatch)

    assert is_rcm_index(str(out)) and not is_rcm_index(str(racmo))
    assert rcm_release("RACMO") == folder["release"] == RELEASE
    assert data_days("RACMO") == folder["days"]
    assert run_years("RACMO") == folder["years"]
    assert run_years("RACMO").end_year == 2023  # the end year is clipped as with the data folder (DISC-001)
    assert {name: model_point(name, "RACMO") for name in ("CI-0", "CI-22")} == folder["points"]
    # the forcing CSV path, so the run folder and hash_inputs.model_point, are the same
    assert forcing_csv_path(model_point("CI-0", "RACMO")) == forcing_csv_path(folder["points"]["CI-0"])


def test_index_records_the_folder(racmo, tmp_path, monkeypatch):
    out = _index(racmo, tmp_path, monkeypatch)
    record = json.loads((out / INDEX_FILE).read_text())
    assert record["release"] == RELEASE
    assert record["data_path"] == str(racmo)
    assert record["coord_file"] == sorted(p.name for p in racmo.glob("*_ANT-12_*.nc"))[0]
    assert "README.txt" in record["files"]
    assert not (tmp_path / "rcm_index.tmp").exists()


def test_match_data_files_is_glob(racmo):
    assert sorted(match_data_files(str(racmo), "tas_*.nc")) == sorted(map(str, racmo.glob("tas_*.nc")))


def test_missing_data_folder_gives_the_usual_error(tmp_path, monkeypatch):
    monkeypatch.setitem(config, "RACMO_data_path", str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError, match="No RACMO files that match .*absent/tas_"):
        data_days("RACMO")


def test_index_is_never_replaced(racmo, tmp_path, monkeypatch):
    _index(racmo, tmp_path, monkeypatch)
    monkeypatch.setitem(config, "RACMO_data_path", str(racmo))
    with pytest.raises(FileExistsError):
        make_rcm_index.main(["--rcm", "RACMO", "--out", str(tmp_path / "rcm_index")])


def test_index_of_an_index_is_refused(racmo, tmp_path, monkeypatch):
    _index(racmo, tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="is an RCM index"):
        make_rcm_index.main(["--rcm", "RACMO", "--out", str(tmp_path / "other")])


def test_index_with_a_different_release_is_refused(racmo, tmp_path, monkeypatch):
    out = _index(racmo, tmp_path, monkeypatch)
    record = json.loads((out / INDEX_FILE).read_text())
    record["release"] = "OTHER"
    (out / INDEX_FILE).write_text(json.dumps(record))
    with pytest.raises(ValueError, match="is for RACMO OTHER"):
        model_point_at(-66.4, -63.4, "RACMO")


def test_missing_grid_file_is_refused(racmo, tmp_path, monkeypatch):
    out = _index(racmo, tmp_path, monkeypatch)
    (out / GRID_FILE).unlink()
    with pytest.raises(FileNotFoundError, match=f"has no {GRID_FILE}"):
        model_point_at(-66.4, -63.4, "RACMO")


def test_grid_file_that_does_not_agree_is_refused(racmo, tmp_path, monkeypatch):
    out = _index(racmo, tmp_path, monkeypatch)
    with np.load(out / GRID_FILE) as grid:
        lat, lon = grid["lat"], grid["lon"] + 0.01
    np.savez_compressed(out / GRID_FILE, lat=lat, lon=lon)
    with pytest.raises(ValueError, match="does not agree with grid_sha256"):
        model_point_at(-66.4, -63.4, "RACMO")


def test_no_forcing_is_made_from_an_index(racmo, tmp_path, monkeypatch):
    _index(racmo, tmp_path, monkeypatch)
    processor = ProcessRACMO(-66.4, -63.4)
    with pytest.raises(FileNotFoundError, match="is an RCM index") as error:
        processor.process()
    assert processor._save_path in str(error.value)


def test_cached_forcing_is_used_with_an_index(racmo, tmp_path, monkeypatch):
    """With the CSV and its .json file in the cache, process() checks them and makes nothing."""
    _index(racmo, tmp_path, monkeypatch)
    monkeypatch.setitem(config, "end_year", 2023)
    processor = ProcessRACMO(-66.4, -63.4)
    csv = Path(processor._save_path)
    csv.parent.mkdir(parents=True)
    csv.write_text("time,T2m\n")
    csv.with_suffix(".json").write_text(
        json.dumps({"first_date": "1979-01-01 12:00:00", "last_date": "2023-12-31 12:00:00"})
    )
    processor.process()
