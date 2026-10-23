"""Tests that scripts/run_cfm.py resolves the years once and gives them to sim.worker (DISC-001).

The forcing, the identity checks and the worker are replaced with stubs. The RACMO data is empty files with ANT-12
names in tmp_path, which stop at 2023-12-31.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from cfm_config import config
from force.cache import ForcingCoverageError
from force.model_point import clear_cache

_spec = importlib.util.spec_from_file_location(
    "run_cfm", Path(__file__).resolve().parents[1] / "scripts" / "run_cfm.py"
)
run_cfm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_cfm)

VERSION = {"label": "test", "repo": "/nowhere/CFM_main", "python": sys.executable}


@pytest.fixture
def calls(tmp_path, monkeypatch):
    """Stub the forcing, the identity checks and the worker. Returns what they were called with."""
    racmo = tmp_path / "racmo"
    racmo.mkdir()
    for var in ("tas", "pr", "prsn"):
        for first, last in (("19790101", "20101231"), ("20110101", "20231231")):
            (racmo / f"{var}_ANT-12_TEST_day_{first}-{last}.nc").touch()
    for key, value in {
        "RACMO_data_path": str(racmo),
        "RACMO_to_CFM_column_map": {"tas": "T2m", "rf": "RAIN"},
        "start_year": 1979,
        "end_year": 2024,
        "spin_end_year": 1989,
    }.items():
        monkeypatch.setitem(config, key, value)
    clear_cache()

    seen = {"forcing": [], "commands": []}

    class FakeProcess:
        def __init__(self, lat, lon):
            pass

        def process(self):
            seen["forcing"].append((config["start_year"], config["end_year"]))

    class Done:
        returncode = 0

    def run(command, env=None):
        seen["commands"].append(command)
        return Done()

    monkeypatch.setitem(run_cfm.PROCESSORS, "RACMO", FakeProcess)
    monkeypatch.setattr(run_cfm, "cfm_version", lambda label: VERSION)
    monkeypatch.setattr(run_cfm, "measure_cfm_identity", lambda version: {})
    monkeypatch.setattr(run_cfm, "check_cfm_identity", lambda identity: None)
    monkeypatch.setattr(run_cfm, "wrapper_identity", lambda: {})
    monkeypatch.setattr(run_cfm, "check_wrapper_identity", lambda identity: None)
    monkeypatch.setattr(run_cfm.subprocess, "run", run)
    yield seen
    clear_cache()


def _option(command, name):
    return command[command.index(name) + 1]


def _worker_years(command):
    names = ("--start-year", "--end-year", "--requested-end-year", "--end-year-source")
    return tuple(_option(command, name) for name in names)


ARGS = ["--site", "CI-0", "--rcm", "RACMO", "--physrho", "GSFC2020", "--liquid", "bucket"]


def test_default_years_are_clipped_and_given_to_the_worker(calls):
    assert run_cfm.main(ARGS) == 0
    assert calls["forcing"] == [(1979, 2023)]
    (command,) = calls["commands"]
    assert command[1:3] == ["-m", "sim.worker"]
    assert _worker_years(command) == ("1979", "2023", "2024", "config")


def test_command_line_years_are_given_to_the_worker(calls):
    assert run_cfm.main(ARGS + ["--start-year", "1980", "--end-year", "2020"]) == 0
    assert calls["forcing"] == [(1980, 2020)]
    assert _worker_years(calls["commands"][0]) == ("1980", "2020", "2020", "cli")


def test_command_line_end_year_after_the_data_stops_before_the_forcing(calls):
    with pytest.raises(ForcingCoverageError):
        run_cfm.main(ARGS + ["--end-year", "2024"])
    assert calls["forcing"] == [] and calls["commands"] == []
