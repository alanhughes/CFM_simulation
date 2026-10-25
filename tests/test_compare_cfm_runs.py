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


def make_run(physRho, cfm, hash_, spin_end_year, tag="MAR_119_17"):
    folder = run_folder(tag, physRho, "bucket", 1979, 2024, cfm, hash_)
    os.makedirs(folder)
    write_manifest(
        folder,
        {
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


def by_scheme(rows):
    return {row["physRho"]: row for row in rows}


def test_matrix_shows_the_spin_end_year(runs):
    rows = by_scheme(compare.matrix("old", ["new"], {}))
    assert rows["GSFC2020"]["spin_end_year"] == 1989 and rows["GSFC2020"]["baseline_spin_end_year"] == 1995
    assert rows["GSFC2020"]["note"] == "spin-up differs"
    assert rows["GSFC2020"]["run"] == "bbbbbbbb"  # the pair is still compared
    assert rows["Crocus"]["spin_end_year"] == rows["Crocus"]["baseline_spin_end_year"] == 1989
    assert "note" not in rows["Crocus"]


def test_matrix_of_one_spin_end_year(runs):
    rows = by_scheme(compare.matrix("old", ["new"], {"spin_end_year": 1989}))
    assert rows["GSFC2020"]["note"] == "no run of old"
    assert rows["GSFC2020"]["spin_end_year"] == 1989 and "baseline_spin_end_year" not in rows["GSFC2020"]
    assert rows["Crocus"]["run"] == "dddddddd"
