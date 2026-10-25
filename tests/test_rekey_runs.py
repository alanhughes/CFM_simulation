"""Tests for scripts/rekey_runs.py (FEAT-005, MIG-003). Run from the repo root.

A small git repo in tmp_path stands in for this repo, so the run code of a commit is known. The CFM is not run, as in
tests/test_run_status.py. A run of hash_version 2 is made from the run_hash_inputs() of dev d53fd13 (the code before
FEAT-005, read from git), with the same arguments as the run of the code of now.
"""

import importlib.util
import json
import logging
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import sim.identity
import sim.run
from cfm_config import config
from force.cache import forcing_csv_path
from force.model_point import ModelPoint
from sim.identity import run_code_sha256, wrapper_identity
from sim.manifest import COMPLETE, FAILED, read_manifest, run_code_drift, run_hash, write_manifest
from sim.run import CFMRun

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rekey_runs", REPO / "scripts" / "rekey_runs.py")
rekey = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rekey)

SITE = (-66.403, -63.376, "GSFC2020", "RACMO", "bucket")
VERSION = {"label": "test", "repo": "/nowhere/CFM_main", "python": sys.executable}
MODEL_POINT = ModelPoint("RACMO", "ANT-12_release", 261, 126, -66.4, -63.4, 1.2)
CFM_IDENTITY = {"commit": "abc", "status": [], "code_diff_sha256": "d", "python_version": "3", "pip_freeze": ["a==1"]}
OLD_RULE = ("src/plot/",)


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def commit_all(repo):
    git(repo, "add", ".")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "change")


@pytest.fixture
def wrapper_repo(tmp_path, monkeypatch):
    """This repo: src/ with run code, src/plot/ and src/sim/runs.py (run code under the rule of hash_version 2)."""
    repo = tmp_path / "wrapper"
    (repo / "src" / "sim").mkdir(parents=True)
    (repo / "src" / "plot").mkdir()
    (repo / "src" / "sim" / "run.py").write_text("x = 1\n")
    (repo / "src" / "sim" / "runs.py").write_text("z = 1\n")
    (repo / "src" / "plot" / "results.py").write_text("y = 1\n")
    git(repo, "init", "-q")
    commit_all(repo)
    monkeypatch.setattr(sim.identity, "_REPO_ROOT", str(repo))
    rekey._run_code_sha256.cache_clear()
    yield repo
    rekey._run_code_sha256.cache_clear()


@pytest.fixture
def data_path(tmp_path, monkeypatch):
    """CFM_data_path in tmp_path, with a forcing CSV for MODEL_POINT."""
    monkeypatch.setitem(config, "CFM_data_path", str(tmp_path / "CFM_data"))
    monkeypatch.setitem(config, "forcing_data_path", str(tmp_path / "CFM_data" / "forcing"))
    monkeypatch.setitem(config, "cfm_config", dict(config["cfm_config"]))
    csv = forcing_csv_path(MODEL_POINT)
    os.makedirs(os.path.dirname(csv))
    with open(csv, "w") as file:
        file.write("time,TSKIN\n1979-01-01,250.0\n")
    return tmp_path / "CFM_data"


@pytest.fixture
def cfm_calls(monkeypatch, wrapper_repo, data_path):
    """Stub the CFM, the model point and the CFM identity. The wrapper identity is the real one, of wrapper_repo."""
    calls = []

    def run_cfm(self):
        calls.append(self._cfm_config["resultsFolder"])
        with open(os.path.join(self._cfm_config["resultsFolder"], "CFMresults.hdf5"), "w") as file:
            file.write("results")

    monkeypatch.setattr(CFMRun, "_run_cfm", run_cfm)
    monkeypatch.setattr(sim.run, "model_point_at", lambda lat, lon, rcm_name: MODEL_POINT)
    monkeypatch.setattr(sim.run, "measure_cfm_identity", lambda version: {**version, **CFM_IDENTITY})
    monkeypatch.setattr(sim.run, "check_cfm_identity", lambda identity: None)
    monkeypatch.setattr(sim.run, "_run_code_drift_warned", False)
    return calls


def old_manifest_module() -> types.ModuleType:
    """sim.manifest of dev d53fd13, before FEAT-005: HASH_VERSION 2, with wrapper_run_code_sha256."""
    source = subprocess.run(
        ["git", "-C", str(REPO), "show", "d53fd13:src/sim/manifest.py"], capture_output=True, text=True, check=True
    ).stdout
    module = types.ModuleType("old_manifest")
    exec(compile(source, "d53fd13:src/sim/manifest.py", "exec"), module.__dict__)
    assert module.HASH_VERSION == 2
    return module


def make_old_run(monkeypatch, physRho: str = SITE[2]) -> dict:
    """A complete run of hash_version 2, as the code of d53fd13 wrote it. Returns the run of the code of now (its
    folder and manifest), which the old run must become."""
    seen = {}
    real = sim.run.run_hash_inputs

    def record(*args):
        seen["args"] = args
        return real(*args)

    with monkeypatch.context() as patch:
        patch.setattr(sim.run, "run_hash_inputs", record)
        run = CFMRun(*SITE[:2], physRho, *SITE[3:], VERSION, site="CI-0")
        new_folder = run.prepare()
    now_inputs = run._hash_inputs

    args = list(seen["args"])
    wrapper = {key: value for key, value in args[6].items() if key != "run_code_version"}
    wrapper["run_code_sha256"] = run_code_sha256(wrapper["commit"], OLD_RULE)
    args[6] = wrapper
    old_inputs = old_manifest_module().run_hash_inputs(*args)
    old_hash = run_hash(old_inputs)
    old_folder = os.path.join(os.path.dirname(new_folder), old_hash[:8])

    # the run of the code of now, then the manifest and folder of hash_version 2
    with monkeypatch.context() as patch:
        patch.setattr(sim.run, "run_folder", lambda *args: old_folder)
        CFMRun(*SITE[:2], physRho, *SITE[3:], VERSION, site="CI-0").run()
    manifest = read_manifest(old_folder)
    manifest.update(run_hash=old_hash, hash_inputs=old_inputs, wrapper=wrapper)
    manifest["cfm_config"] = {**config["cfm_config"], "resultsFolder": old_folder}
    write_manifest(old_folder, manifest)
    with open(os.path.join(old_folder, "CFMconfig.json"), "w") as file:
        file.write('{"resultsFolder": "old"}\n')
    return {
        "old_folder": old_folder,
        "new_folder": new_folder,
        "hash_inputs": now_inputs,
        "run_hash": run_hash(now_inputs),
    }


def files(folder) -> dict:
    return {name: Path(folder, name).read_bytes() for name in sorted(os.listdir(folder))}


def test_rekey_gives_the_run_hash_of_now(cfm_calls, monkeypatch, capsys, caplog):
    """KEY TEST: the re-keyed run has the hash_inputs and the run hash that CFMRun.prepare() gives now. Thus, the same
    run is skipped, and gives no drift warning."""
    expected = make_old_run(monkeypatch)
    old, new = expected["old_folder"], expected["new_folder"]
    assert old != new and os.path.dirname(old) == os.path.dirname(new)
    before = files(old)

    assert rekey.main([]) == 0  # the dry run changes nothing
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "Run code version: 1 (RUN_CODE_VERSION in src/sim/identity.py)"
    assert f"REKEY   {old}\n        -> {new}" in out
    manifest = read_manifest(old)
    assert f"run_hash {manifest['run_hash']} -> {expected['run_hash']}" in out
    commit = manifest["wrapper"]["commit"]
    assert f"{run_code_sha256(commit, OLD_RULE)} -> {run_code_sha256(commit)} (commit {commit}" in out
    assert files(old) == before and not os.path.exists(new)

    assert rekey.main(["--apply"]) == 0
    assert not os.path.exists(old)
    after = files(new)
    assert {name: data for name, data in after.items() if name != "manifest.json"} == {
        name: data for name, data in before.items() if name != "manifest.json"
    }  # CFMconfig.json, run.log and CFMresults.hdf5 are not changed
    manifest = read_manifest(new)
    assert manifest["hash_inputs"] == expected["hash_inputs"]
    assert manifest["run_hash"] == expected["run_hash"]
    assert manifest["cfm_config"]["resultsFolder"] == new
    assert manifest["wrapper"]["run_code_version"] == 1
    assert manifest["wrapper"]["run_code_sha256"] == wrapper_identity()["run_code_sha256"]
    assert manifest["wrapper"]["run_code_sha256"] != run_code_sha256(commit, OLD_RULE)
    rekeyed = manifest["rekeyed"]
    assert rekeyed["from_run_hash"] == json.loads(before["manifest.json"])["run_hash"]
    assert rekeyed["from_hash_version"] == 2
    assert rekeyed["from_run_code_sha256"] == run_code_sha256(commit, OLD_RULE)
    assert rekeyed["from_folder"] == old
    assert rekeyed["script"] == "scripts/rekey_runs.py" and rekeyed["date"] and "FEAT-005" in rekeyed["reason"]

    # run_cfm.py with the same options: the complete run is found, and skipped
    caplog.clear()
    caplog.set_level(logging.INFO)
    CFMRun(*SITE, VERSION, site="CI-0").run()
    assert cfm_calls == [old]  # only the run of make_old_run
    assert "Complete run found" in caplog.text
    assert "The run code changed" not in caplog.text
    assert run_code_drift(wrapper_identity()) is None


def test_second_apply_selects_nothing(cfm_calls, monkeypatch, capsys):
    expected = make_old_run(monkeypatch)
    assert rekey.main(["--apply"]) == 0
    manifest = read_manifest(expected["new_folder"])
    capsys.readouterr()
    assert rekey.main(["--apply"]) == 0
    assert "Summary: 0 runs of hash_version 2" in capsys.readouterr().out
    assert read_manifest(expected["new_folder"]) == manifest


def test_a_stop_after_the_rename_is_finished(cfm_calls, monkeypatch, capsys):
    """The rename is first and the manifest last. A stop between the two leaves the new folder with the manifest of
    hash_version 2, and the next run of the script writes the manifest (FINISH)."""
    expected = make_old_run(monkeypatch)

    def stop(folder, manifest):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(rekey, "write_manifest", stop)
        with pytest.raises(KeyboardInterrupt):
            rekey.main(["--apply"])
    assert not os.path.exists(expected["old_folder"])
    assert read_manifest(expected["new_folder"])["hash_inputs"]["hash_version"] == 2

    capsys.readouterr()
    assert rekey.main([]) == 0
    assert f"FINISH  {expected['new_folder']}" in capsys.readouterr().out
    assert rekey.main(["--apply"]) == 0
    manifest = read_manifest(expected["new_folder"])
    assert manifest["run_hash"] == expected["run_hash"]
    assert manifest["hash_inputs"] == expected["hash_inputs"]
    assert manifest["rekeyed"]["from_folder"] == expected["old_folder"]
    assert manifest["cfm_config"]["resultsFolder"] == expected["new_folder"]


def change_old_run(folder: str, change) -> None:
    """Change the manifest of an old run, and give it a correct run_hash and folder name again."""
    manifest = read_manifest(folder)
    change(manifest)
    manifest["run_hash"] = run_hash(manifest["hash_inputs"])
    write_manifest(folder, manifest)
    os.rename(folder, os.path.join(os.path.dirname(folder), manifest["run_hash"][:8]))


def other_code(manifest):
    manifest["wrapper"]["run_code_sha256"] = manifest["hash_inputs"]["wrapper_run_code_sha256"] = "0" * 64


def unknown_commit(manifest):
    manifest["wrapper"]["commit"] = "1" * 40


@pytest.mark.parametrize("change", [other_code, unknown_commit], ids=["other_code", "unknown_commit"])
def test_a_run_of_other_code_is_refused(cfm_calls, monkeypatch, capsys, change):
    """A run gets the version only if its recorded run code is the code of its own commit."""
    expected = make_old_run(monkeypatch)
    change_old_run(expected["old_folder"], change)
    (folder,) = [entry.path for entry in os.scandir(os.path.dirname(expected["old_folder"]))]
    before = files(folder)
    assert rekey.main(["--apply"]) == 1
    assert "REFUSE" in capsys.readouterr().out
    assert files(folder) == before
    assert sorted(os.listdir(os.path.dirname(folder))) == [os.path.basename(folder)]


def test_an_existing_target_is_refused(cfm_calls, monkeypatch, capsys):
    expected = make_old_run(monkeypatch)
    os.makedirs(expected["new_folder"])
    before = files(expected["old_folder"])
    assert rekey.main(["--apply"]) == 1
    assert "the new folder" in capsys.readouterr().out
    assert files(expected["old_folder"]) == before
    assert os.listdir(expected["new_folder"]) == []


def test_a_wrong_run_hash_is_refused(cfm_calls, monkeypatch, capsys):
    expected = make_old_run(monkeypatch)
    manifest = read_manifest(expected["old_folder"])
    manifest["hash_inputs"]["forcing_csv_sha256"] = "changed"
    write_manifest(expected["old_folder"], manifest)
    assert rekey.main(["--apply"]) == 1
    assert "not the hash of its hash_inputs" in capsys.readouterr().out
    assert os.path.exists(expected["old_folder"])


@pytest.mark.parametrize(
    "change",
    [
        lambda manifest: manifest.update(status=FAILED),
        lambda manifest: manifest.update(migrated={"source": "old"}),
        lambda manifest: manifest["hash_inputs"].update(hash_version="migrated-2"),
    ],
    ids=["failed", "migrated", "migrated_hash"],
)
def test_other_runs_are_not_selected(cfm_calls, monkeypatch, capsys, change):
    expected = make_old_run(monkeypatch)
    manifest = read_manifest(expected["old_folder"])
    change(manifest)
    write_manifest(expected["old_folder"], manifest)
    before = files(expected["old_folder"])
    assert rekey.main(["--apply"]) == 0
    assert "Summary: 0 runs" in capsys.readouterr().out
    assert files(expected["old_folder"]) == before


def test_a_new_run_is_not_selected(cfm_calls, capsys):
    CFMRun(*SITE, VERSION).run()
    assert rekey.main(["--apply"]) == 0
    assert "Summary: 0 runs" in capsys.readouterr().out
    assert COMPLETE == read_manifest(cfm_calls[0])["status"]


def test_the_rekey_does_not_hide_a_later_change(cfm_calls, monkeypatch, wrapper_repo):
    """The re-keyed run has the run code of its commit. After a later change to the run code, the drift warning
    occurs, as for a new run."""
    make_old_run(monkeypatch)
    assert rekey.main(["--apply"]) == 0
    assert run_code_drift(wrapper_identity()) is None
    (wrapper_repo / "src" / "sim" / "runs.py").write_text("z = 2\n")  # not run code now
    commit_all(wrapper_repo)
    assert run_code_drift(wrapper_identity()) is None
    (wrapper_repo / "src" / "sim" / "run.py").write_text("x = 2\n")
    commit_all(wrapper_repo)
    assert run_code_drift(wrapper_identity()) is not None


def test_one_run_at_the_new_code_ends_the_warning(cfm_calls, monkeypatch, wrapper_repo, caplog):
    """A re-keyed set of runs of one commit, then a change to the run code that needs no new version: the first run
    warns, and after one complete run at the new code, no run warns (FEAT-005 open question 3)."""
    old = [make_old_run(monkeypatch, physRho) for physRho in ("GSFC2020", "KuipersMunneke2015")]
    assert rekey.main(["--apply"]) == 0
    for run in old:  # they finished some days before; the runs of this test can finish in the same second
        write_manifest(run["new_folder"], {**read_manifest(run["new_folder"]), "finished": "2026-10-20T12:00:00+00:00"})
    old_commit = read_manifest(old[0]["new_folder"])["wrapper"]["commit"]
    (wrapper_repo / "src" / "sim" / "run.py").write_text("x = 1  # a comment\n")
    commit_all(wrapper_repo)
    caplog.set_level(logging.INFO)

    CFMRun(*SITE[:2], "Barnola1991", *SITE[3:], VERSION, site="CI-0").run()  # a new run, at the new code
    assert f"since the last run of run code version 1 (commit {old_commit}," in caplog.text
    assert len(cfm_calls) == 3

    caplog.clear()
    monkeypatch.setattr(sim.run, "_run_code_drift_warned", False)  # a new process
    assert run_code_drift(wrapper_identity()) is None
    CFMRun(*SITE, VERSION, site="CI-0").run()  # a re-keyed run: skipped, with no warning
    assert "Complete run found" in caplog.text
    assert "The run code changed" not in caplog.text
    assert len(cfm_calls) == 3


def test_the_version_is_run_code_version(cfm_calls, monkeypatch, capsys):
    """MIG-003 open question 1: the script has no --run-code-version. It uses RUN_CODE_VERSION, and prints it first."""
    with pytest.raises(SystemExit):
        rekey.main(["--run-code-version", "1"])
    expected = make_old_run(monkeypatch)
    monkeypatch.setattr(rekey, "RUN_CODE_VERSION", 2)
    capsys.readouterr()
    assert rekey.main(["--apply"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "Run code version: 2 (RUN_CODE_VERSION in src/sim/identity.py)"
    (folder,) = [entry.path for entry in os.scandir(os.path.dirname(expected["old_folder"]))]
    manifest = read_manifest(folder)
    assert manifest["hash_inputs"]["wrapper_run_code_version"] == 2
    assert manifest["wrapper"]["run_code_version"] == 2
    assert manifest["rekeyed"]["run_code_version"] == 2
    assert manifest["run_hash"] != expected["run_hash"]  # not the run of version 1
