"""Tests for the run hash, the run folder, the manifest and the skip logic in CFMRun.run (FEAT-001, BUG-004).
Run from the repo root.

The CFM itself is not run: _run_cfm, the model point and the CFM and wrapper identities are replaced with stubs. The
forcing CSV is a small file in a forcing cache in tmp_path.
"""

import json
import os
import subprocess
import sys

import pytest

import sim.run
from cfm_config import config
from force.cache import forcing_csv_path, forcing_json_path
from force.model_point import ModelPoint
from sim.manifest import COMPLETE, FAILED, RUNNING, read_manifest, write_manifest
from sim.run import CFMRun

SITE = (-66.403, -63.376, "GSFC2020", "RACMO", "bucket")
VERSION = {"label": "test", "repo": "/nowhere/CFM_main", "python": sys.executable}
MODEL_POINT = ModelPoint("RACMO", "ANT-12_release", 261, 126, -66.4, -63.4, 1.2)
CFM_IDENTITY = {"commit": "abc", "status": [], "code_diff_sha256": "d", "python_version": "3", "pip_freeze": ["a==1"]}


@pytest.fixture
def data_path(tmp_path, monkeypatch):
    """CFM_data_path in tmp_path, with a forcing CSV for MODEL_POINT."""
    monkeypatch.setitem(config, "CFM_data_path", str(tmp_path))
    monkeypatch.setitem(config, "forcing_data_path", str(tmp_path / "forcing"))
    monkeypatch.setitem(config, "cfm_config", dict(config["cfm_config"]))
    csv = forcing_csv_path(MODEL_POINT)
    os.makedirs(os.path.dirname(csv))
    with open(csv, "w") as file:
        file.write("time,TSKIN\n1979-01-01,250.0\n")
    return tmp_path


@pytest.fixture
def cfm_calls(monkeypatch):
    """Stub out the CFM run, and count the CFM runs."""
    calls = []

    def run_cfm(self):
        calls.append(self._cfm_config["resultsFolder"])
        open(os.path.join(self._cfm_config["resultsFolder"], "CFMresults.hdf5"), "w").close()

    monkeypatch.setattr(CFMRun, "_run_cfm", run_cfm)
    return calls


@pytest.fixture(autouse=True)
def identities(monkeypatch):
    """Stub out the model point (which reads the RCM grid) and the identities (which run git and pip freeze)."""
    cfm = dict(CFM_IDENTITY)
    wrapper = {"commit": "w", "status": [], "src_tree": "tree", "run_code_sha256": "code"}
    monkeypatch.setattr(sim.run, "model_point_at", lambda lat, lon, rcm_name: MODEL_POINT)
    monkeypatch.setattr(sim.run, "measure_cfm_identity", lambda version: {**version, **cfm})
    monkeypatch.setattr(sim.run, "check_cfm_identity", lambda identity: None)
    monkeypatch.setattr(sim.run, "wrapper_identity", lambda: dict(wrapper))
    monkeypatch.setattr(sim.run, "check_wrapper_identity", lambda identity: None)
    return {"cfm": cfm, "wrapper": wrapper}


def new_run(**kwargs) -> CFMRun:
    return CFMRun(*SITE, VERSION, **kwargs)


def folder(**kwargs) -> str:
    return new_run(**kwargs).prepare()


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
    assert manifest["run"]["forcing_csv"] == forcing_csv_path(MODEL_POINT)
    assert manifest["run"]["model_point"]["tag"] == "RACMO_261_126"
    assert manifest["run_hash"].startswith(os.path.basename(folder()))
    assert manifest["hash_inputs"]["model_point"] == {"tag": "RACMO_261_126", "release": "ANT-12_release"}
    assert "runtime_s" in manifest
    assert manifest["cfm"]["label"] == "test"
    assert manifest["cfm"]["commit"] == "abc"
    assert os.path.exists(os.path.join(folder(), "run.log"))


def test_folder_layout(data_path):
    path = os.path.relpath(folder(), data_path)
    parts = path.split(os.sep)
    assert parts[:5] == ["runs", "RACMO_261_126", "GSFC2020_bucket_1979-2024", "test"] + parts[4:5]
    assert len(parts) == 5 and len(parts[4]) == 8


def test_requested_end_year_is_in_the_manifest_not_the_hash(data_path, cfm_calls):
    """DISC-001: the folder and the hash have the end year of the run; the manifest also has the one asked for."""
    new_run(requested_end_year=2030, end_year_source="config").run()
    manifest = read_manifest(folder())
    assert manifest["run"]["end_year"] == config["end_year"]
    assert manifest["run"]["requested_end_year"] == 2030
    assert manifest["run"]["end_year_source"] == "config"
    assert folder(requested_end_year=2031, end_year_source="cli") == folder()  # the same run
    assert read_manifest(folder())["run"]["requested_end_year"] == 2030
    new_run(force=True).run()
    assert read_manifest(folder())["run"]["requested_end_year"] == config["end_year"]  # default: the end year


def test_cfm_config_json_is_in_the_run_folder(data_path, monkeypatch):
    """The real _run_cfm up to the CFM: the config JSON goes in the run folder, which keeps run.log and the manifest."""
    seen = {}

    class Stop(Exception):
        pass

    def make_spin_files(*args, **kwargs):
        return {"SUBLIM": 1.0}, 12.0, 5.0, 10.0, 100.0, None

    def firn_density(config_path, **kwargs):
        seen["config"] = config_path
        raise Stop

    monkeypatch.setattr(sim.run, "import_cfm", lambda repo: (make_spin_files, firn_density))
    with pytest.raises(Stop):
        new_run().run()
    assert seen["config"] == os.path.join(folder(), "CFMconfig.json")
    assert sorted(os.listdir(folder())) == ["CFMconfig.json", "manifest.json", "run.log"]
    assert read_manifest(folder())["cfm_config"]["resultsFolder"] == folder()


def test_sites_are_recorded(data_path, cfm_calls):
    new_run(site="CI-0").run()
    CFMRun(-66.41, -63.38, *SITE[2:], VERSION, site="CI-0-twin").run()  # another site in the same cell
    new_run(site="CI-0").run()
    assert len(cfm_calls) == 1
    run = read_manifest(folder())["run"]
    assert run["sites"] == ["CI-0", "CI-0-twin"]
    # _add_site records the coordinates of the site that it adds
    assert run["site_coords"] == {"CI-0": [-66.403, -63.376], "CI-0-twin": [-66.41, -63.38]}


def test_site_coords_are_not_in_the_run_hash(data_path, cfm_calls):
    """The run is for the model point: the site names and coordinates do not change the run hash (FEAT-002)."""
    new_run(site="CI-0").run()
    other = CFMRun(-66.5, -63.5, *SITE[2:], VERSION, site="other")
    other.prepare()
    manifest = read_manifest(folder())
    assert manifest["run_hash"] == other._run_hash
    assert manifest["hash_inputs"] == other._hash_inputs
    assert manifest["run"]["site_coords"] == {"CI-0": [-66.403, -63.376]}
    hashed = json.dumps(manifest["hash_inputs"])
    assert "site_coords" not in hashed and "CI-0" not in hashed and "-66.403" not in hashed


@pytest.mark.parametrize(
    "change",
    [
        lambda ids: config["cfm_config"].update(conductivity="Calonne2011"),
        lambda ids: config.update(sublim_sf=1),
        lambda ids: ids["cfm"].update(commit="def"),
        lambda ids: ids["cfm"].update(code_diff_sha256="e"),
        lambda ids: ids["cfm"].update(pip_freeze=["a==2"]),
        lambda ids: ids["wrapper"].update(run_code_sha256="code2"),
        lambda ids: open(forcing_csv_path(MODEL_POINT), "a").write("1979-01-02,251.0\n"),
    ],
    ids=["cfm_config", "wrapper_param", "cfm_commit", "cfm_patch", "pip_freeze", "run_code", "forcing"],
)
def test_a_change_gives_a_new_run(data_path, cfm_calls, identities, monkeypatch, change):
    monkeypatch.setitem(config, "sublim_sf", config["sublim_sf"])
    new_run().run()
    change(identities)
    new_run().run()
    assert len(cfm_calls) == 2
    assert cfm_calls[0] != cfm_calls[1]
    assert os.path.dirname(cfm_calls[0]) == os.path.dirname(cfm_calls[1])


@pytest.mark.parametrize(
    "change",
    [
        lambda ids: ids["cfm"].update(status=[" M CFMoutput_example/x.hdf5"], diff_sha256="x"),
        lambda ids: ids["wrapper"].update(commit="w2", status=[" M README.md"], src_tree="tree2"),
        lambda ids: ids["cfm"].update(repo="/elsewhere/CFM_main", python="/elsewhere/python"),
    ],
    ids=["cfm_data_change", "wrapper_commit", "paths"],
)
def test_a_change_that_does_not_change_results_keeps_the_run(data_path, cfm_calls, identities, change):
    new_run(site="CI-0").run()
    change(identities)
    new_run(site="other").run()
    assert len(cfm_calls) == 1


def test_hash_ignores_the_forcing_json(data_path, cfm_calls):
    """The old and the fast extraction (FEAT-003) give the same CSV but a different JSON file: the same run."""
    json_path = forcing_json_path(forcing_csv_path(MODEL_POINT))
    with open(json_path, "w") as file:
        json.dump({"method": "old"}, file)
    new_run().run()
    with open(json_path, "w") as file:
        json.dump({"method": "fast"}, file)
    new_run().run()
    assert len(cfm_calls) == 1


def test_forcing_json_with_other_sha256_is_refused(data_path, cfm_calls):
    with open(forcing_json_path(forcing_csv_path(MODEL_POINT)), "w") as file:
        json.dump({"csv_sha256": "0" * 64}, file)
    with pytest.raises(RuntimeError, match="changed after it was made"):
        new_run().run()
    assert not os.path.exists(data_path / "runs")


def test_manifest_with_other_hash_is_refused(data_path, cfm_calls):
    os.makedirs(folder())
    write_manifest(folder(), {"status": COMPLETE, "run_hash": "0" * 64})
    for force in (False, True):
        with pytest.raises(RuntimeError, match="contains the run"):
            new_run(force=force).run()
    assert cfm_calls == []


def run_hash() -> str:
    run = new_run()
    run.prepare()
    return run._run_hash


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
    write_manifest(
        folder(), {"status": RUNNING, "host": os.uname().nodename, "pid": dead_pid(), "run_hash": run_hash()}
    )
    new_run().run()
    assert len(cfm_calls) == 1


@pytest.mark.parametrize("force", [False, True])
def test_live_running_manifest_is_not_rerun(data_path, cfm_calls, force):
    os.makedirs(folder())
    write_manifest(
        folder(), {"status": RUNNING, "host": os.uname().nodename, "pid": os.getpid(), "run_hash": run_hash()}
    )
    with pytest.raises(RuntimeError, match="being run"):
        new_run(force=force).run()
    assert cfm_calls == []


def test_running_manifest_from_other_host_needs_force(data_path, cfm_calls):
    os.makedirs(folder())
    write_manifest(folder(), {"status": RUNNING, "host": "another-host", "pid": 1, "run_hash": run_hash()})
    with pytest.raises(RuntimeError, match="--force"):
        new_run().run()
    new_run(force=True).run()
    assert len(cfm_calls) == 1


def test_empty_folder_without_manifest_is_rerun(data_path, cfm_calls):
    os.makedirs(folder())
    new_run().run()
    assert len(cfm_calls) == 1


def test_folder_with_results_but_no_manifest_needs_force(data_path, cfm_calls):
    os.makedirs(folder())
    open(os.path.join(folder(), "CFMresults.hdf5"), "w").close()
    with pytest.raises(RuntimeError, match="no manifest"):
        new_run().run()
    assert cfm_calls == []
    new_run(force=True).run()
    assert len(cfm_calls) == 1


def test_missing_forcing_leaves_no_folder(data_path):
    os.remove(forcing_csv_path(MODEL_POINT))
    with pytest.raises(FileNotFoundError):
        new_run().run()
    assert not os.path.exists(data_path / "runs")


def test_wrong_python_is_refused(data_path, cfm_calls):
    with pytest.raises(RuntimeError, match="runs with"):
        CFMRun(*SITE, {**VERSION, "python": "/usr/bin/false"}).run()
    assert cfm_calls == []
    assert not os.path.exists(data_path / "runs")
