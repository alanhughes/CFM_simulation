"""Tests for scripts/infer_cfm_config.py (FEAT-009, MIG-005). Run from the repo root.

The local root and the share root are in tmp_path. The run folders have a manifest and a small CFMforcing.hdf5 only.
The real share is never read.
"""

import importlib.util
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from cfm_config import config
from sim.manifest import read_manifest, write_manifest
from sim.runs import find_runs

_spec = importlib.util.spec_from_file_location(
    "infer_cfm_config", Path(__file__).resolve().parents[1] / "scripts" / "infer_cfm_config.py"
)
infer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(infer)

RELATIVE = os.path.join("RACMO_261_125", "GSFC2020_bucket_1979-2023", "3e3bf7e-speersm")


def make_run(root: Path, hash8: str, seb: bool = True, cfm_config=None, migrated: bool = True) -> Path:
    """A run folder with a manifest and a CFMforcing.hdf5 of the form of this value of SEB."""
    folder = root / "runs" / RELATIVE / hash8
    folder.mkdir(parents=True)
    with h5py.File(folder / "CFMforcing.hdf5", "w") as file:
        file["main/time"] = np.arange(4.0)
        if seb:
            file["SEB/T2m"] = np.zeros(4)
        else:
            file["main/SMELT"] = np.zeros(4)
    manifest = {
        "status": "complete",
        "run_hash": hash8 * 8,
        "hash_inputs": {"hash_version": "migrated-2"},
        "run": {"model_point": {"tag": "RACMO_261_125"}, "rcm_name": "RACMO", "physRho": "GSFC2020"},
        "cfm": {"label": "3e3bf7e-speersm"},
        "cfm_config": cfm_config,
        "finished": "2026-10-25T12:00:00",
    }
    if migrated:
        manifest["migrated"] = {"unknown": ["cfm_config"]}
    write_manifest(str(folder), manifest)
    return folder


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """A local root and a share root, both with runs/."""
    local, share = tmp_path / "local", tmp_path / "share"
    (local / "runs").mkdir(parents=True)
    (share / "runs").mkdir(parents=True)
    monkeypatch.setitem(config, "CFM_data_path", str(local))
    monkeypatch.setitem(config, "CFM_share_data_path", str(share))
    return local, share


def snapshot(*roots: Path) -> dict:
    return {str(p): p.read_bytes() for root in roots for p in root.rglob("*") if p.is_file()}


def test_dry_run_writes_nothing(roots, capsys):
    local, share = roots
    make_run(local, "11111111")
    make_run(share, "11111111")
    make_run(local, "22222222", seb=False)
    before = snapshot(local, share)
    assert infer.main([]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "ADD 2, SAME 0, DIFFER 0, ERROR 0" in out
    assert "Manifests to write: local 2, share 1" in out
    assert "{'SEB': False} 1, {'SEB': True} 1" in out
    assert snapshot(local, share) == before


def test_apply_writes_the_same_keys_into_the_two_copies(roots, capsys):
    local, share = roots
    folders = [make_run(local, "11111111"), make_run(share, "11111111")]
    seb_false = make_run(local, "22222222", seb=False)
    forcing = (folders[0] / "CFMforcing.hdf5").read_bytes()
    assert infer.main(["--apply"]) == 0
    manifests = [read_manifest(str(folder)) for folder in folders]
    assert manifests[0] == manifests[1]
    assert manifests[0]["migrated"]["inferred_cfm_config"] == {"SEB": True}
    assert manifests[0]["migrated"]["inferred_from"] == {"SEB": "CFMforcing.hdf5: group SEB, no main/SMELT"}
    assert manifests[0]["migrated"]["unknown"] == ["cfm_config"]
    assert manifests[0]["cfm_config"] is None and manifests[0]["run_hash"] == "1" * 64
    assert (folders[0] / "CFMforcing.hdf5").read_bytes() == forcing
    (run,) = find_runs(cfm_config={"SEB": False})
    assert run.folder == str(seb_false)
    (run,) = find_runs(cfm_config={"SEB": True})
    assert run.location == "both"
    # a second run gives SAME and writes nothing
    before = snapshot(local, share)
    capsys.readouterr()
    assert infer.main(["--apply"]) == 0
    assert "ADD 0, SAME 2" in capsys.readouterr().out
    assert snapshot(local, share) == before


def test_runs_with_a_cfm_config_or_not_migrated_are_not_selected(roots, capsys):
    local, _ = roots
    make_run(local, "11111111", cfm_config={"SEB": True})  # migrated with a CFMconfig JSON
    make_run(local, "22222222", migrated=False, cfm_config={"SEB": True})  # a new run
    assert infer.main([]) == 0
    assert "Summary: 0 migrated runs" in capsys.readouterr().out


def test_copies_that_disagree_are_refused(roots, capsys):
    local, share = roots
    make_run(local, "11111111", seb=True)
    make_run(share, "11111111", seb=False)
    make_run(local, "22222222")
    before = snapshot(local, share)
    assert infer.main(["--apply"]) == 1
    captured = capsys.readouterr()
    assert "ERROR 1" in captured.out and "different values" in captured.out
    assert "nothing is changed" in captured.err
    assert snapshot(local, share) == before


def test_a_different_inferred_value_is_refused(roots, capsys):
    local, share = roots
    folder = make_run(local, "11111111")
    manifest = read_manifest(str(folder))
    manifest["migrated"]["inferred_cfm_config"] = {"SEB": False}
    write_manifest(str(folder), manifest)
    before = snapshot(local, share)
    assert infer.main(["--apply"]) == 1
    assert "DIFFER 1" in capsys.readouterr().out
    assert snapshot(local, share) == before


def test_no_inferred_value_is_written_as_empty(roots):
    local, _ = roots
    folder = make_run(local, "11111111")
    (folder / "CFMforcing.hdf5").write_bytes(b"not hdf5")
    assert infer.main(["--apply"]) == 0
    migrated = read_manifest(str(folder))["migrated"]
    assert migrated["inferred_cfm_config"] == {} and migrated["inferred_from"] == {}
    assert find_runs(cfm_config={"SEB": True}) == []


def test_manifest_that_changed_after_the_plan_is_refused(roots):
    local, _ = roots
    folder = make_run(local, "11111111")
    (plan,) = infer.build_plans([str(local / "runs")])
    manifest = read_manifest(str(folder))
    manifest["finished"] = "2026-10-30T00:00:00"
    write_manifest(str(folder), manifest)
    with pytest.raises(RuntimeError, match="changed after the plan"):
        infer.apply(plan)
    assert "inferred_cfm_config" not in read_manifest(str(folder))["migrated"]


def test_share_not_available_stops(roots, monkeypatch, capsys):
    local, _ = roots
    make_run(local, "11111111")
    monkeypatch.setattr(infer, "share_available", lambda: False)
    before = snapshot(local)
    assert infer.main(["--apply"]) == 1
    assert "not available" in capsys.readouterr().err
    assert snapshot(local) == before


def test_sync_runs_gives_same_after_the_apply(roots):
    """The risk of FEAT-009: the local and the share manifests must not differ after MIG-005."""
    import shutil

    _sync_spec = importlib.util.spec_from_file_location(
        "sync_runs", Path(__file__).resolve().parents[1] / "scripts" / "sync_runs.py"
    )
    sync = importlib.util.module_from_spec(_sync_spec)
    _sync_spec.loader.exec_module(sync)
    local, share = roots
    folder = make_run(local, "11111111")
    target = share / "runs" / RELATIVE / "11111111"
    shutil.copytree(folder, target)
    rel = os.path.join(RELATIVE, "11111111")
    assert sync.classify_run(str(folder), str(target), rel).cls == sync.SAME
    assert infer.main(["--apply"]) == 0
    assert sync.classify_run(str(folder), str(target), rel).cls == sync.SAME
