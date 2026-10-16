"""Tests for the run manifest and the skip logic in CFMRun.run (BUG-004). Run from the repo root.

The CFM itself is not run: _read_force_data, _run_cfm and the CFM identity are replaced with stubs.
"""

import os
import subprocess
import sys

import pytest

import sim.run
from cfm_config import config
from sim.manifest import COMPLETE, FAILED, RUNNING, read_manifest, write_manifest
from sim.run import CFMRun

SITE = (-66.403, -63.376, "GSFC2020", "RACMO", "bucket")
VERSION = {"label": "test", "repo": "/nowhere/CFM_main", "python": sys.executable, "allow_uncommitted": False}


@pytest.fixture
def data_path(tmp_path, monkeypatch):
    monkeypatch.setitem(config, "CFM_data_path", str(tmp_path))
    monkeypatch.setitem(config, "cfm_config", dict(config["cfm_config"]))
    os.makedirs(tmp_path / "cfm_output")
    return tmp_path


@pytest.fixture
def cfm_calls(monkeypatch):
    """Stub out the forcing read and the CFM run, and count the CFM runs."""
    calls = []

    def read_force_data(self):
        self._force_data_path = "forcing.csv"

    def run_cfm(self):
        calls.append(self._cfm_config["resultsFolder"])
        open(os.path.join(self._cfm_config["resultsFolder"], "CFMresults.hdf5"), "w").close()

    monkeypatch.setattr(CFMRun, "_read_force_data", read_force_data)
    monkeypatch.setattr(CFMRun, "_run_cfm", run_cfm)
    return calls


@pytest.fixture(autouse=True)
def cfm_identity(monkeypatch):
    """Stub out the CFM identity, which would otherwise run git and pip freeze on VERSION's repo."""
    monkeypatch.setattr(sim.run, "measure_cfm_identity", lambda version: {**version, "commit": "abc", "status": []})
    monkeypatch.setattr(sim.run, "check_cfm_identity", lambda identity: None)


def new_run(**kwargs) -> CFMRun:
    return CFMRun(*SITE, VERSION, **kwargs)


def folder() -> str:
    return new_run()._cfm_config["resultsFolder"]


def dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def test_new_run_writes_complete_manifest(data_path, cfm_calls):
    new_run().run()
    manifest = read_manifest(folder())
    assert len(cfm_calls) == 1
    assert manifest["status"] == COMPLETE
    assert manifest["run"]["physRho"] == "GSFC2020"
    assert manifest["run"]["forcing_csv"] == "forcing.csv"
    assert "runtime_s" in manifest
    assert manifest["cfm"]["label"] == "test"
    assert manifest["cfm"]["commit"] == "abc"
    assert os.path.exists(os.path.join(folder(), "run.log"))


def test_site_is_recorded(data_path, cfm_calls):
    new_run(site="CI-0").run()
    assert read_manifest(folder())["run"]["site"] == "CI-0"


def test_complete_run_is_skipped(data_path, cfm_calls):
    new_run().run()
    new_run().run()
    assert len(cfm_calls) == 1


def test_force_reruns_complete_run(data_path, cfm_calls):
    new_run().run()
    new_run(force=True).run()
    assert len(cfm_calls) == 2


def test_failed_run_is_marked_failed_and_rerun(data_path, cfm_calls, monkeypatch):
    def crash(self):
        raise ValueError("boom")

    with monkeypatch.context() as patch:
        patch.setattr(CFMRun, "_run_cfm", crash)
        with pytest.raises(ValueError):
            new_run().run()
    manifest = read_manifest(folder())
    assert manifest["status"] == FAILED
    assert manifest["error"] == "ValueError: boom"

    new_run().run()
    assert len(cfm_calls) == 1
    assert read_manifest(folder())["status"] == COMPLETE


def test_stale_running_manifest_is_rerun(data_path, cfm_calls):
    os.makedirs(folder())
    write_manifest(folder(), {"status": RUNNING, "host": os.uname().nodename, "pid": dead_pid()})
    new_run().run()
    assert len(cfm_calls) == 1


@pytest.mark.parametrize("force", [False, True])
def test_live_running_manifest_is_not_rerun(data_path, cfm_calls, force):
    os.makedirs(folder())
    write_manifest(folder(), {"status": RUNNING, "host": os.uname().nodename, "pid": os.getpid()})
    with pytest.raises(RuntimeError, match="being run"):
        new_run(force=force).run()
    assert cfm_calls == []


def test_running_manifest_from_other_host_needs_force(data_path, cfm_calls):
    os.makedirs(folder())
    write_manifest(folder(), {"status": RUNNING, "host": "another-host", "pid": 1})
    with pytest.raises(RuntimeError, match="--force"):
        new_run().run()
    new_run(force=True).run()
    assert len(cfm_calls) == 1


def test_empty_folder_without_manifest_is_rerun(data_path, cfm_calls):
    os.makedirs(folder())
    new_run().run()
    assert len(cfm_calls) == 1


def test_old_folder_with_results_but_no_manifest_is_skipped(data_path, cfm_calls):
    os.makedirs(folder())
    open(os.path.join(folder(), "CFMresults.hdf5"), "w").close()
    new_run().run()
    assert cfm_calls == []


def test_missing_forcing_leaves_no_folder(data_path):
    with pytest.raises(FileNotFoundError):
        new_run().run()
    assert not os.path.exists(folder())


def test_wrong_python_is_refused(data_path, cfm_calls):
    with pytest.raises(RuntimeError, match="runs with"):
        CFMRun(*SITE, {**VERSION, "python": "/usr/bin/false"}).run()
    assert cfm_calls == []
    assert not os.path.exists(folder())
