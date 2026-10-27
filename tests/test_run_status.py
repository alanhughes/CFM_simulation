"""Tests for the run hash, the run folder, the manifest and the skip logic in CFMRun.run (FEAT-001, BUG-004).
Run from the repo root.

The CFM itself is not run: _run_cfm, the model point and the CFM and wrapper identities are replaced with stubs. The
forcing CSV is a small file in a forcing cache in tmp_path.
"""

import errno
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import sim.manifest
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
    wrapper = {"commit": "w", "status": [], "src_tree": "tree", "run_code_version": 1, "run_code_sha256": "code"}
    monkeypatch.setattr(sim.run, "model_point_at", lambda lat, lon, rcm_name: MODEL_POINT)
    monkeypatch.setattr(sim.run, "measure_cfm_identity", lambda version: {**version, **cfm})
    monkeypatch.setattr(sim.run, "check_cfm_identity", lambda identity: None)
    monkeypatch.setattr(sim.run, "wrapper_identity", lambda: dict(wrapper))
    monkeypatch.setattr(sim.run, "check_wrapper_identity", lambda identity: None)
    return {"cfm": cfm, "wrapper": wrapper}


@pytest.fixture(autouse=True)
def drift_not_warned(monkeypatch):
    """Each test is a new process for the drift warning, which occurs one time in each process (FEAT-005)."""
    monkeypatch.setattr(sim.run, "_run_code_drift_warned", False)


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
        lambda ids: ids["wrapper"].update(run_code_version=2),
        lambda ids: open(forcing_csv_path(MODEL_POINT), "a").write("1979-01-02,251.0\n"),
    ],
    ids=["cfm_config", "wrapper_param", "cfm_commit", "cfm_patch", "pip_freeze", "run_code_version", "forcing"],
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
        # the run code changed, with the same RUN_CODE_VERSION (FEAT-005)
        lambda ids: ids["wrapper"].update(commit="w2", src_tree="tree2", run_code_sha256="code2"),
        lambda ids: ids["cfm"].update(repo="/elsewhere/CFM_main", python="/elsewhere/python"),
    ],
    ids=["cfm_data_change", "wrapper_commit", "run_code_same_version", "paths"],
)
def test_a_change_that_does_not_change_results_keeps_the_run(data_path, cfm_calls, identities, change):
    new_run(site="CI-0").run()
    change(identities)
    new_run(site="other").run()
    assert len(cfm_calls) == 1


def test_manifest_records_the_run_code(data_path, cfm_calls):
    """The run hash has the run code version only; the manifest also records the code (FEAT-005)."""
    new_run().run()
    manifest = read_manifest(folder())
    assert manifest["hash_inputs"]["hash_version"] == 3
    assert manifest["hash_inputs"]["wrapper_run_code_version"] == 1
    assert "wrapper_run_code_sha256" not in manifest["hash_inputs"]
    assert manifest["wrapper"]["run_code_version"] == 1
    assert manifest["wrapper"]["run_code_sha256"] == "code"
    assert manifest["wrapper"]["commit"] == "w"
    assert manifest["wrapper"]["src_tree"] == "tree"


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


DRIFT = "The run code changed since the last run of run code version 1 (commit w, finished"


def drift_warnings(caplog) -> int:
    return sum(DRIFT in record.getMessage() for record in caplog.records)


def test_changed_run_code_of_the_same_version_warns_once(data_path, cfm_calls, identities, caplog):
    """FEAT-005: the same version with a different run_code_sha256 gives the same run, and a warning."""
    new_run().run()
    assert drift_warnings(caplog) == 0
    identities["wrapper"].update(commit="w2", run_code_sha256="code2")
    new_run().run()
    folder()
    assert drift_warnings(caplog) == 1
    assert "Increase RUN_CODE_VERSION in src/sim/identity.py" in caplog.text
    assert len(cfm_calls) == 1  # the complete run is skipped: the warning does not stop or change the run
    config["cfm_config"].update(conductivity="Calonne2011")
    new_run().run()
    assert len(cfm_calls) == 2


def test_same_run_code_does_not_warn(data_path, cfm_calls, identities, caplog):
    new_run().run()
    identities["wrapper"].update(commit="w2", src_tree="tree2")
    new_run(force=True).run()
    assert drift_warnings(caplog) == 0
    assert len(cfm_calls) == 2


def test_other_runs_do_not_warn(data_path, cfm_calls, identities, caplog):
    """Only a complete new run of the same version counts: not another version, a failed or migrated run, a manifest
    from before FEAT-005 (hash_version 2, no wrapper.run_code_version) or a manifest that cannot be read. They all
    finished after the run of now, so each one would be the last run if it counted."""
    new_run().run()
    other = {**identities["wrapper"], "run_code_sha256": "other"}
    before_feat_005 = {key: value for key, value in other.items() if key != "run_code_version"}
    manifests = {
        "version2": {"status": COMPLETE, "wrapper": {**other, "run_code_version": 2}},
        "failed": {"status": FAILED, "wrapper": other},
        "migrated": {"status": COMPLETE, "wrapper": other, "migrated": {"from": "old"}},
        "hash2": {"status": COMPLETE, "wrapper": before_feat_005},
        "no_wrapper": {"status": COMPLETE, "wrapper": None},
    }
    for name, manifest in manifests.items():
        path = data_path / "runs" / "RACMO_1_1" / "GSFC2020_bucket_1979-2024" / "test" / name
        os.makedirs(path)
        write_manifest(str(path), {"run_hash": name, "finished": "2100-01-01T00:00:00+00:00", **manifest})
    broken = data_path / "runs" / "RACMO_1_1" / "GSFC2020_bucket_1979-2024" / "test" / "broken"
    os.makedirs(broken)
    (broken / "manifest.json").write_text("{")
    new_run(force=True).run()
    assert drift_warnings(caplog) == 0
    assert len(cfm_calls) == 2


def write_run(data_path, name: str, finished: str, wrapper: dict) -> None:
    """A complete new run of another model point, with its finished time and wrapper."""
    path = data_path / "runs" / "RACMO_1_1" / "GSFC2020_bucket_1979-2024" / "test" / name
    os.makedirs(path)
    write_manifest(str(path), {"run_hash": name, "status": COMPLETE, "finished": finished, "wrapper": wrapper})


def test_only_the_last_run_counts(data_path, cfm_calls, identities, caplog):
    """FEAT-005 open question 3: the drift check compares with the complete run of the version that finished last."""
    wrapper = identities["wrapper"]
    write_run(data_path, "older", "2026-10-01T10:00:00+00:00", {**wrapper, "commit": "w0", "run_code_sha256": "old"})
    write_run(data_path, "last", "2026-10-02T10:00:00+00:00", dict(wrapper))
    folder()
    assert "The run code changed" not in caplog.text  # an older run differs, but the last run is the same

    write_run(data_path, "later", "2026-10-03T08:00:00+02:00", {**wrapper, "commit": "w9", "run_code_sha256": "new"})
    folder()
    assert "since the last run of run code version 1 (commit w9, finished 2026-10-03T08:00:00+02:00)" in caplog.text


def test_a_complete_run_of_the_new_code_ends_the_warning(data_path, cfm_calls, identities, caplog, monkeypatch):
    new_run().run()
    identities["wrapper"].update(commit="w2", run_code_sha256="code2")
    config["cfm_config"].update(conductivity="Calonne2011")
    new_run().run()  # warns, and runs at the new code
    assert drift_warnings(caplog) == 1
    assert len(cfm_calls) == 2
    caplog.clear()
    monkeypatch.setattr(sim.run, "_run_code_drift_warned", False)  # a new process
    for record in (data_path / "runs").rglob("manifest.json"):  # the run of the old code finished first
        manifest = json.loads(record.read_text())
        if manifest["wrapper"]["run_code_sha256"] == "code":
            write_manifest(str(record.parent), {**manifest, "finished": "2026-10-01T00:00:00+00:00"})
    folder()
    assert "The run code changed" not in caplog.text


# a run on the share (FEAT-008). The share is a folder in tmp_path, never the real share; a share run is a run made in
# the local root and then moved to the folder with the same relative path in the share root, as the sync copies it


@pytest.fixture
def share(data_path, tmp_path, monkeypatch):
    """CFM_share_data_path in tmp_path, with a runs/ folder."""
    data = tmp_path / "share"
    (data / "runs").mkdir(parents=True)
    monkeypatch.setitem(config, "CFM_share_data_path", str(data))
    return data


def share_folder(share) -> str:
    """The share folder with the same relative path as the local run folder."""
    return os.path.join(share, os.path.relpath(folder(), config["CFM_data_path"]))


def move_to_share(share, copy: bool = False) -> str:
    """Move (or copy) the local run to the share. Returns the share folder."""
    target = share_folder(share)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    (shutil.copytree if copy else shutil.move)(folder(), target)
    return target


def share_not_available(monkeypatch, share) -> list:
    """A "soft" mount that stops: os.stat of a share path raises OSError. Returns the paths it was called with."""
    real_stat = os.stat
    attempts = []

    def stat(path, *args, **kwargs):
        if str(path).startswith(str(share)):
            attempts.append(path)
            raise OSError(errno.EIO, "Input/output error", str(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(sim.manifest.os, "stat", stat)
    return attempts


def test_complete_share_run_is_skipped(share, cfm_calls, caplog):
    new_run(site="CI-0").run()
    target = move_to_share(share)
    new_run(site="CI-0").run()
    assert len(cfm_calls) == 1
    assert f"Complete run found on the share at {target}, skipping." in caplog.text
    assert not os.path.exists(folder())  # no local folder is made
    assert read_manifest(target)["run"]["sites"] == ["CI-0"]


def test_share_run_with_other_hash_is_refused(share, cfm_calls):
    target = share_folder(share)
    os.makedirs(target)
    write_manifest(target, {"status": COMPLETE, "run_hash": "0" * 64})
    for force in (False, True):
        with pytest.raises(RuntimeError, match="contains the run"):
            new_run(force=force).run()
    assert cfm_calls == []
    assert not os.path.exists(folder())


def test_force_runs_in_the_local_folder(share, cfm_calls):
    new_run(site="CI-0").run()
    target = move_to_share(share)
    before = {name: (Path(target) / name).read_bytes() for name in os.listdir(target)}
    new_run(site="CI-0", force=True).run()
    assert len(cfm_calls) == 2 and cfm_calls[1] == folder()
    assert read_manifest(folder())["status"] == COMPLETE
    assert {name: (Path(target) / name).read_bytes() for name in os.listdir(target)} == before  # the share copy
    # the local run is found first now
    new_run(site="CI-0").run()
    assert len(cfm_calls) == 2


def test_share_run_that_is_not_complete_runs_in_the_local_folder(share, cfm_calls):
    new_run().run()
    target = move_to_share(share)
    write_manifest(target, {**read_manifest(target), "status": FAILED})
    new_run().run()
    assert len(cfm_calls) == 2
    assert read_manifest(folder())["status"] == COMPLETE
    assert read_manifest(target)["status"] == FAILED


def test_new_site_of_a_share_run_is_written_on_the_share(share, cfm_calls):
    new_run(site="CI-0").run()
    target = move_to_share(share)
    CFMRun(-66.41, -63.38, *SITE[2:], VERSION, site="CI-0-twin").run()  # another site in the same cell
    assert len(cfm_calls) == 1
    run = read_manifest(target)["run"]
    assert run["sites"] == ["CI-0", "CI-0-twin"]
    assert run["site_coords"] == {"CI-0": [-66.403, -63.376], "CI-0-twin": [-66.41, -63.38]}
    assert not os.path.exists(folder())
    assert not os.path.exists(os.path.join(target, "manifest.json.tmp"))


def test_new_site_of_a_run_in_both_locations_is_written_locally(share, cfm_calls):
    new_run(site="CI-0").run()
    target = move_to_share(share, copy=True)
    CFMRun(-66.41, -63.38, *SITE[2:], VERSION, site="CI-0-twin").run()
    assert len(cfm_calls) == 1
    assert read_manifest(folder())["run"]["sites"] == ["CI-0", "CI-0-twin"]
    assert read_manifest(target)["run"]["sites"] == ["CI-0"]  # the sync writes the union later


def test_failed_write_of_a_share_site_warns(share, cfm_calls, monkeypatch, caplog):
    new_run(site="CI-0").run()
    target = move_to_share(share)
    real_write = sim.run.write_manifest

    def write(folder_, manifest):
        if str(folder_).startswith(str(share)):
            raise OSError(errno.EIO, "Input/output error", folder_)
        real_write(folder_, manifest)

    monkeypatch.setattr(sim.run, "write_manifest", write)
    CFMRun(-66.41, -63.38, *SITE[2:], VERSION, site="CI-0-twin").run()  # no error
    assert len(cfm_calls) == 1
    assert f"Cannot add site CI-0-twin to the sites of {target}" in caplog.text
    assert read_manifest(target)["run"]["sites"] == ["CI-0"]


def test_failed_write_of_a_local_site_is_an_error(data_path, cfm_calls, monkeypatch):
    new_run(site="CI-0").run()

    def write(folder_, manifest):
        raise OSError(errno.ENOSPC, "No space left on device", folder_)

    monkeypatch.setattr(sim.run, "write_manifest", write)
    with pytest.raises(OSError):
        CFMRun(-66.41, -63.38, *SITE[2:], VERSION, site="CI-0-twin").run()


def test_share_not_available_runs_in_the_local_folder(share, cfm_calls, monkeypatch, caplog):
    new_run().run()
    move_to_share(share)
    sim.manifest.reset_share_check()  # a new process
    attempts = share_not_available(monkeypatch, share)
    new_run().run()
    new_run().run()
    assert len(cfm_calls) == 2  # the share run is not seen; the second run finds the new local run
    assert read_manifest(folder())["status"] == COMPLETE
    assert len(attempts) == 1  # one check of the share in the process
    assert "is not available" in caplog.text


def test_drift_finds_the_last_run_on_the_share(data_path, share, cfm_calls, identities, caplog):
    """The drift check compares with the last run of the version in the two roots (FEAT-008)."""
    wrapper = identities["wrapper"]
    write_run(data_path, "older", "2026-10-01T10:00:00+00:00", dict(wrapper))
    write_run(share, "last", "2026-10-02T10:00:00+00:00", {**wrapper, "commit": "w9", "run_code_sha256": "new"})
    folder()
    assert "since the last run of run code version 1 (commit w9, finished 2026-10-02T10:00:00+00:00)" in caplog.text


def test_drift_with_no_local_run(share, cfm_calls, identities, caplog):
    """After prune_local_runs.py removes the local copies, the last run of the version is on the share only."""
    wrapper = identities["wrapper"]
    write_run(share, "last", "2026-10-02T10:00:00+00:00", {**wrapper, "commit": "w9", "run_code_sha256": "new"})
    assert not os.path.exists(os.path.join(config["CFM_data_path"], "runs"))
    folder()
    assert "since the last run of run code version 1 (commit w9, finished 2026-10-02T10:00:00+00:00)" in caplog.text
    # an explicit root is that root only
    assert sim.manifest.run_code_drift(wrapper, root=str(share / "runs"))["commit"] == "w9"
    assert sim.manifest.run_code_drift(wrapper, root=os.path.join(config["CFM_data_path"], "runs")) is None
