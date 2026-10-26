"""Tests for the matrix mode of scripts/compare_cfm_runs.py (FEAT-004). Run from the repo root.

The run folders are made with manifests only, and summary() is a stub, so no .hdf5 file is read.
"""

import importlib.util
import os
from pathlib import Path

import pytest

import sim.runs
from cfm_config import config
from sim.manifest import COMPLETE, run_folder, write_manifest

_spec = importlib.util.spec_from_file_location(
    "compare_cfm_runs", Path(__file__).resolve().parents[1] / "scripts" / "compare_cfm_runs.py"
)
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)


def make_run(physRho, cfm, hash_, spin_end_year, tag="MAR_119_17", meltwater_solver=None):
    folder = run_folder(tag, physRho, "bucket", 1979, 2024, cfm, hash_)
    os.makedirs(folder)
    write_manifest(
        folder,
        {
            "cfm_config": {"meltwater_solver": meltwater_solver} if meltwater_solver else {},
            "status": COMPLETE,
            "run_hash": hash_,
            "run": {
                "sites": [],
                "model_point": {"tag": tag},
                "rcm_name": "MAR",
                "physRho": physRho,
                "liquid": "bucket",
                "start_year": 1979,
                "end_year": 2024,
                "spin_end_year": spin_end_year,
            },
            "cfm": {"label": cfm},
            "finished": "2026-10-25T10:00:00",
        },
    )
    return folder


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setitem(config, "CFM_data_path", str(tmp_path))
    monkeypatch.setattr(sim.runs, "_warned_sites", set())
    monkeypatch.setattr(compare, "summary", lambda new_dir, ref_dir: {"bit_for_bit": False})
    make_run("GSFC2020", "old", "a" * 64, 1995)
    make_run("GSFC2020", "new", "b" * 64, 1989)
    make_run("Crocus", "old", "c" * 64, 1989)
    make_run("Crocus", "new", "d" * 64, 1989)


def of_scheme(rows, physRho):
    return [row for row in rows if row["physRho"] == physRho]


def test_matrix_pairs_the_runs_of_one_spin_end_year(runs):
    rows = compare.matrix("old", ["new"], {})
    (crocus,) = of_scheme(rows, "Crocus")
    assert crocus["run"] == "dddddddd" and crocus["baseline_run"] == "cccccccc"
    assert crocus["spin_end_year"] == crocus["baseline_spin_end_year"] == 1989
    assert "note" not in crocus


def test_matrix_does_not_pair_different_spin_end_years(runs):
    rows = of_scheme(compare.matrix("old", ["new"], {}), "GSFC2020")
    assert sorted(row["note"] for row in rows) == [
        "no run of new with spin_end_year 1995", "no run of old with spin_end_year 1989"
    ]
    assert not any("run" in row for row in rows)
    new_only = next(row for row in rows if row["note"].startswith("no run of old"))
    assert new_only["spin_end_year"] == 1989 and "baseline_spin_end_year" not in new_only


def test_matrix_with_allow_spin_mismatch(runs):
    (gsfc,) = of_scheme(compare.matrix("old", ["new"], {}, allow_spin_mismatch=True), "GSFC2020")
    assert gsfc["spin_end_year"] == 1989 and gsfc["baseline_spin_end_year"] == 1995
    assert gsfc["note"] == "spin-up differs"
    assert gsfc["run"] == "bbbbbbbb" and gsfc["baseline_run"] == "aaaaaaaa"
    (crocus,) = of_scheme(compare.matrix("old", ["new"], {}, allow_spin_mismatch=True), "Crocus")
    assert "note" not in crocus


def test_matrix_of_one_spin_end_year(runs):
    rows = compare.matrix("old", ["new"], {"spin_end_year": 1989}, allow_spin_mismatch=True)
    (gsfc,) = of_scheme(rows, "GSFC2020")
    assert gsfc["note"] == "no run of old"
    assert gsfc["spin_end_year"] == 1989 and "baseline_spin_end_year" not in gsfc
    assert of_scheme(rows, "Crocus")[0]["run"] == "dddddddd"


def test_matrix_pairs_the_runs_of_one_meltwater_solver(runs):
    # a run of new with another solver does not replace the enthalpy run of new in the pair
    make_run("Crocus", "new", "e" * 64, 1989, meltwater_solver="ncz")
    rows = of_scheme(compare.matrix("old", ["new"], {}), "Crocus")
    (enthalpy,) = [row for row in rows if row["meltwater_solver"] == "enthalpy"]
    assert enthalpy["run"] == "dddddddd" and enthalpy["baseline_run"] == "cccccccc"
    (ncz,) = [row for row in rows if row["meltwater_solver"] == "ncz"]
    assert ncz["note"] == "no run of old with spin_end_year 1989"
    (only,) = of_scheme(compare.matrix("old", ["new"], {"meltwater_solver": "enthalpy"}), "Crocus")
    assert only["run"] == "dddddddd"
