"""Tests for scripts/migrate_runs.py (FEAT-001 step 6). Run from the repo root.

The old folders are small files in tmp_path. The RCM grid is a stub, so no luna data is read.
"""

import importlib.util
import os
from pathlib import Path

import pandas as pd
import pytest

from cfm_config import config
from force.cache import sha256_file
from force.model_point import ModelPoint
from sim.manifest import COMPLETE, read_manifest
from sim.runs import find_runs

_spec = importlib.util.spec_from_file_location(
    "migrate_runs", Path(__file__).resolve().parents[1] / "scripts" / "migrate_runs.py"
)
migrate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migrate)

NAME = "CFMoutput_-66.403_-63.376_1979_2024_GSFC2020_bucket_RACMO"


def make_source(
    root: Path, results: bytes = b"results", config_json: bool = True, csv_end: str = "2023-12-31", coords=None
) -> Path:
    """An old CFM_data folder: cfm_output/<name>/ with .hdf5 files, and cfm_input/ with the forcing CSV.

    coords is the (lat, lon) text of the folder name, ("-66.403", "-63.376") by default."""
    lat, lon = coords or ("-66.403", "-63.376")
    name = f"CFMoutput_{lat}_{lon}_1979_2024_GSFC2020_bucket_RACMO"
    output = root / "cfm_output"
    (output / name).mkdir(parents=True)
    (output / name / "CFMresults.hdf5").write_bytes(results)
    (output / name / "CFMspin.hdf5").write_bytes(b"spin")
    (output / name / "CFMforcing.hdf5").write_bytes(b"forcing")
    if config_json:
        (output / (name.replace("CFMoutput_", "CFMconfig_") + ".json")).write_text(
            '{"physRho": "GSFC2020", "resultsFolder": "/old/path"}'
        )
    (root / "cfm_input").mkdir()
    dates = pd.date_range("1979-01-01", csv_end, freq="D")
    csv = root / "cfm_input" / f"RACMO_{lat}_{lon}_1979_2024.csv"
    pd.DataFrame({"TIME": dates, "T2m": 250.0}).to_csv(csv, index=False)
    os.utime(csv, (0, 0))  # older than the results
    return output


@pytest.fixture
def data_path(tmp_path, monkeypatch):
    monkeypatch.setitem(config, "CFM_data_path", str(tmp_path / "CFM_data"))
    monkeypatch.setattr(migrate, "old_cell", lambda rcm, lat, lon: (261, 125))
    monkeypatch.setattr(
        migrate,
        "cell_model_point",
        lambda rcm, i, j, lat, lon: ModelPoint(rcm, "release", i, j, -66.4, -63.4, 7.7, "degrees (before BUG-002)"),
    )
    monkeypatch.setattr(migrate, "wrapper_identity", lambda: {"commit": "w"})
    migrate._sha256.cache_clear()
    migrate._read_csv.cache_clear()
    return tmp_path


def snapshot(folder: Path) -> dict:
    """The SHA-256 and the modification time of each file under folder."""
    return {str(p): (sha256_file(p), p.stat().st_mtime) for p in folder.rglob("*") if p.is_file()}


def test_dry_run_writes_nothing(data_path, capsys):
    source = make_source(data_path / "old")
    assert migrate.main(["--source", "3e3bf7e-speersm", str(source)]) == 0
    assert not (data_path / "CFM_data").exists()
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "COPY 1" in out
    plan = migrate.build_plans([("3e3bf7e-speersm", str(source))])[0]
    # the model point is the cell of the old search, and the end year is the end of the forcing (BUG-003)
    assert plan.target.startswith(
        str(data_path / "CFM_data" / "runs" / "RACMO_261_125" / "GSFC2020_bucket_1979-2023" / "3e3bf7e-speersm")
    )
    assert plan.site == "CI-0"


def test_copy(data_path):
    old = data_path / "old"
    source = make_source(old)
    before = snapshot(old)
    assert migrate.main(["--source", "3e3bf7e-speersm", str(source), "--copy"]) == 0
    assert snapshot(old) == before  # no source file is changed

    (run,) = find_runs(model_point="RACMO_261_125")
    manifest = read_manifest(run.folder)
    assert manifest["status"] == COMPLETE
    assert manifest["hash_inputs"]["hash_version"] == "migrated-1"
    assert manifest["migrated"]["source"] == str(source / NAME)
    assert manifest["run"]["sites"] == ["CI-0"]
    assert manifest["run"]["site_coords"] == {"CI-0": [-66.403, -63.376]}
    assert run.cfm == "3e3bf7e-speersm"
    assert sorted(os.listdir(run.folder)) == [
        "CFMconfig.json", "CFMforcing.hdf5", "CFMresults.hdf5", "CFMspin.hdf5", "manifest.json", "run.log"
    ]
    assert (Path(run.folder) / "CFMresults.hdf5").read_bytes() == b"results"

    # a second copy finds the run
    plan = migrate.build_plans([("3e3bf7e-speersm", str(source))])[0]
    assert plan.action == "EXISTS"


def test_duplicate_is_compared(data_path, capsys):
    first, second = make_source(data_path / "a"), make_source(data_path / "b")
    sources = ["--source", "3e3bf7e-speersm", str(first), "--source", "3e3bf7e-speersm", str(second)]
    assert migrate.main(sources + ["--copy"]) == 0
    assert "SAME" in capsys.readouterr().out
    assert len(find_runs()) == 1


def test_duplicate_sites_have_their_own_coordinates(data_path):
    """Two sites in one cell with the same results: one run, and each site has the coordinates of its own folder."""
    first = make_source(data_path / "a", coords=("-68.395", "-64.713"))
    second = make_source(data_path / "b", coords=("-68.395", "-64.710"))
    assert migrate.main(["--source", "x", str(first), "--source", "x", str(second), "--copy"]) == 0
    (run,) = find_runs()
    assert run.manifest["run"]["sites"] == ["J409", "J109"]
    assert run.manifest["run"]["site_coords"] == {"J409": [-68.395, -64.713], "J109": [-68.395, -64.71]}


def test_duplicate_with_other_bytes_stops(data_path):
    first, second = make_source(data_path / "a", b"results"), make_source(data_path / "b", b"RESULTS")
    sources = ["--source", "3e3bf7e-speersm", str(first), "--source", "3e3bf7e-speersm", str(second)]
    with pytest.raises(RuntimeError, match="is not the same"):
        migrate.main(sources + ["--copy"])


def test_other_results_or_config_give_other_runs(data_path):
    a = make_source(data_path / "a", b"results")
    b = make_source(data_path / "b", b"longer results")
    c = make_source(data_path / "c", config_json=False)
    plans = migrate.build_plans([("x", str(a)), ("x", str(b)), ("x", str(c))])
    assert [plan.action for plan in plans] == ["COPY", "COPY", "COPY"]
    assert len({plan.run_hash for plan in plans}) == 3


def test_folder_without_results_is_skipped(data_path):
    source = make_source(data_path / "old")
    os.remove(source / NAME / "CFMresults.hdf5")
    (plan,) = migrate.build_plans([("x", str(source))])
    assert plan.action == "SKIP"
