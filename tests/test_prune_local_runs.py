"""Tests for scripts/prune_local_runs.py (FEAT-008). Run from the repo root.

The local root and the share root are two temporary folders. The .hdf5 files are tiny fake files, and the dataset
comparison of compare_cfm_runs is a stub.
"""

import errno
import importlib.util
import os
from pathlib import Path

import pytest

import sim.runs
from cfm_config import config
from sim.manifest import COMPLETE, read_manifest, write_manifest

_scripts = Path(__file__).resolve().parents[1] / "scripts"
_spec = importlib.util.spec_from_file_location("prune_local_runs", _scripts / "prune_local_runs.py")
prune = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prune)
sync = prune.sync

HASH_A, HASH_B, HASH_C, HASH_D = "a" * 64, "b" * 64, "c" * 64, "d" * 64


def rel_of(hash_, model_point="MAR_119_17", cfm="ab871c8"):
    return os.path.join(model_point, "GSFC2020_bucket_1979-2024", cfm, hash_[:8])


@pytest.fixture
def roots(tmp_path, monkeypatch):
    local, share = tmp_path / "local", tmp_path / "share"
    share.mkdir()
    local.mkdir()
    monkeypatch.setitem(config, "CFM_data_path", str(local))
    monkeypatch.setitem(config, "CFM_share_data_path", str(share))
    monkeypatch.setattr(sync.compare, "summary", lambda new, ref: {"bit_for_bit": False})
    monkeypatch.setattr(sim.runs, "_warned_sites", set())
    return local, share


def make_run(root, hash_=HASH_A, sites=("CI-0",), status=COMPLETE, data=b"results", model_point="MAR_119_17",
             cfm="ab871c8", manifest_hash=None):
    folder = Path(root) / "runs" / rel_of(hash_, model_point, cfm)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "CFMresults.hdf5").write_bytes(data)
    (folder / "run.log").write_text("log\n")
    run = {"sites": list(sites), "site_coords": {s: [-70.0 + i, 20.0] for i, s in enumerate(sites)},
           "model_point": {"tag": model_point}}
    write_manifest(
        str(folder),
        {"status": status, "run_hash": manifest_hash or hash_, "finished": "2026-10-25T10:00:00",
         "cfm": {"commit": "abc", "label": cfm}, "run": run},
    )
    return folder


def prune_args(*args):
    return prune.main(list(args))


def snapshot_tree(root):
    """{path: (bytes, mtime_ns)} of every file under root, and {path: None} of every folder."""
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns) if p.is_file() else None
            for p in Path(root).rglob("*")}


def test_prune_main(roots, capsys):
    local, share = roots
    a = make_run(local, HASH_A)
    make_run(share, HASH_A)
    b = make_run(local, HASH_B)
    c = make_run(local, HASH_C)
    make_run(share, HASH_C, data=b"resultX")  # same size, different bytes
    d = make_run(local, HASH_D)
    make_run(share, HASH_D, manifest_hash="e" * 64)  # same path, different run hash
    before = snapshot_tree(share)
    local_before = snapshot_tree(local)

    assert prune_args() == 0
    out = capsys.readouterr().out
    assert all(folder.exists() for folder in (a, b, c, d))
    assert snapshot_tree(local) == local_before and snapshot_tree(share) == before  # the dry run writes nothing
    assert "1 runs selected" in out
    size = sum(size for size, _ in prune.snapshot(str(a))[1].values())
    assert f"selected      {prune.human(size):>10s}  {rel_of(HASH_A)}" in out
    assert f"not selected  {rel_of(HASH_B)}: no share copy" in out
    assert f"{rel_of(HASH_C)}: DIFFER: CFMresults.hdf5: different" in out
    assert f"{rel_of(HASH_D)}: CONFLICT: different run hash" in out
    assert "Dry run: nothing is removed. Use --apply to remove." in out

    assert prune_args("--apply") == 0
    assert not a.exists()
    assert all(folder.exists() for folder in (b, c, d))
    assert a.parent.exists()  # the parent folders stay
    assert snapshot_tree(share) == before
    (log,) = (local / "sync-logs").iterdir()
    lines = {line.split("\t")[0]: line.split("\t") for line in log.read_text().splitlines()}
    assert lines[rel_of(HASH_A)][1] == "SAME" and lines[rel_of(HASH_A)][3] == "removed"
    assert lines[rel_of(HASH_B)][3].startswith("not selected: no share copy")
    assert len(lines) == 4


def test_local_sites_not_on_share_not_removed(roots):
    local, share = roots
    folder = make_run(local, sites=("CI-0", "CI-22"))
    make_run(share, sites=("CI-0",))
    assert prune_args("--apply") == 0
    assert folder.exists()
    assert "do sync_runs.py --apply first" in next((local / "sync-logs").iterdir()).read_text()


def test_share_has_extra_site_is_removed(roots):
    local, share = roots
    folder = make_run(local, sites=("CI-0",))
    make_run(share, sites=("CI-0", "CI-22"))
    assert prune_args("--apply") == 0
    assert not folder.exists()


def test_share_only_file_not_removed(roots):
    local, share = roots
    folder = make_run(local)
    target = make_run(share)
    (target / "extra.txt").write_text("x")
    assert prune_args("--apply") == 0
    assert folder.exists()
    assert "extra.txt: in the share copy only" in next((local / "sync-logs").iterdir()).read_text()


def test_not_complete_not_selected(roots, capsys):
    local, share = roots
    folder = make_run(local, status="failed")
    make_run(share)
    assert prune_args() == 0
    assert "the local status is 'failed', not 'complete'" in capsys.readouterr().out
    assert prune_args("--apply") == 0
    assert folder.exists()
    assert next((local / "sync-logs").iterdir()).read_text().split("\t")[1] == "-"


def test_oserror_on_share(roots, monkeypatch, capsys):
    local, share = roots
    folder = make_run(local)
    make_run(share)
    original = sync._size_sha

    def failing(path):
        if str(path).startswith(str(share)):
            raise OSError(errno.EIO, "Input/output error")
        return original(path)

    monkeypatch.setattr(sync, "_size_sha", failing)
    assert prune_args("--apply") == 1
    assert folder.exists()
    assert "ERROR" in next((local / "sync-logs").iterdir()).read_text()


def test_share_not_set(roots):
    local, share = roots
    folder = make_run(local)
    del config["CFM_share_data_path"]
    assert prune_args("--apply") == 1
    assert folder.exists()
    assert not (local / "sync-logs").exists()


def test_share_not_available(roots, monkeypatch):
    local, share = roots
    folder = make_run(local)
    make_run(share)
    monkeypatch.setattr(prune, "share_available", lambda: False)
    assert prune_args("--apply") == 1
    assert folder.exists()


def test_share_root_equal_to_local_root(roots):
    local, share = roots
    folder = make_run(local)
    config["CFM_share_data_path"] = str(local)
    assert prune_args("--apply") == 1
    assert (folder / "CFMresults.hdf5").exists()
    assert not (local / "sync-logs").exists()


def test_share_root_inside_local_root(roots):
    local, share = roots
    folder = make_run(local)
    config["CFM_share_data_path"] = str(local / "runs")
    assert prune_args("--apply") == 1
    assert folder.exists()


def test_run_changes_after_check(roots, monkeypatch):
    local, share = roots
    folder = make_run(local)
    make_run(share)
    original = sync.classify_run

    def classify_then_change(*args, **kwargs):
        run = original(*args, **kwargs)
        (folder / "late.txt").write_text("new")
        return run

    monkeypatch.setattr(sync, "classify_run", classify_then_change)
    assert prune_args("--apply") == 1
    assert (folder / "late.txt").exists()
    assert "FAILED: the local run changed during the check" in next((local / "sync-logs").iterdir()).read_text()


def test_filters(roots, capsys):
    local, share = roots
    inside_filter = make_run(local, HASH_A, cfm="ab871c8")
    make_run(share, HASH_A, cfm="ab871c8")
    other_cfm = make_run(local, HASH_B, cfm="1234567")
    make_run(share, HASH_B, cfm="1234567")
    other_point = make_run(local, HASH_C, model_point="RACMO_1_1")
    make_run(share, HASH_C, model_point="RACMO_1_1")
    assert prune_args("--cfm", "ab871c8", "--model-point", "MAR_119_17") == 0
    out = capsys.readouterr().out
    assert "1 local runs" in out and rel_of(HASH_B, cfm="1234567") not in out
    assert prune_args("--apply", "--cfm", "ab871c8", "--model-point", "MAR_119_17") == 0
    assert not inside_filter.exists() and other_cfm.exists() and other_point.exists()


def test_drop_cache_only_share_files(roots, monkeypatch):
    local, share = roots
    make_run(local)
    make_run(share)
    dropped = []
    monkeypatch.setattr(sync, "_drop_cache", dropped.append)
    local_folder, target = str(local / "runs" / rel_of(HASH_A)), str(share / "runs" / rel_of(HASH_A))
    assert sync.classify_run(local_folder, target, rel_of(HASH_A)).cls == sync.SAME
    assert dropped == []  # the default: sync does not drop the cache
    assert prune_args() == 0
    assert sorted(dropped) == [os.path.join(target, "CFMresults.hdf5"), os.path.join(target, "run.log")]
    assert all(path.startswith(str(share)) for path in dropped)


def test_symlinked_run_folder_not_removed(roots, tmp_path):
    local, share = roots
    make_run(share)
    elsewhere = make_run(tmp_path / "elsewhere")
    link = local / "runs" / rel_of(HASH_A)
    link.parent.mkdir(parents=True)
    link.symlink_to(elsewhere, target_is_directory=True)
    assert prune_args("--apply") == 1
    assert elsewhere.exists() and link.is_symlink()
    assert "FAILED" in next((local / "sync-logs").iterdir()).read_text()


def test_run_rewritten_with_same_size_after_check(roots, monkeypatch):
    local, share = roots
    folder = make_run(local)
    make_run(share)
    original = sync.classify_run

    def classify_then_rewrite(*args, **kwargs):
        run = original(*args, **kwargs)
        path = folder / "CFMresults.hdf5"
        path.write_bytes(b"resultY")  # the same size
        os.utime(path, ns=(1, 1))
        return run

    monkeypatch.setattr(sync, "classify_run", classify_then_rewrite)
    assert prune_args("--apply") == 1
    assert (folder / "CFMresults.hdf5").read_bytes() == b"resultY"


def test_local_manifest_tmp_not_removed(roots):
    local, share = roots
    folder = make_run(local)
    make_run(share)
    (folder / "manifest.json.tmp").write_text("{}")
    assert prune_args("--apply") == 0
    assert folder.exists()
    assert "manifest.json.tmp" in next((local / "sync-logs").iterdir()).read_text()


def test_no_run_hash_not_removed(roots):
    local, share = roots
    folders = []
    for root in (local, share):
        folder = make_run(root)
        manifest = read_manifest(str(folder))
        manifest["run_hash"] = None
        write_manifest(str(folder), manifest)
        folders.append(folder)
    assert sync.classify_run(str(folders[0]), str(folders[1]), rel_of(HASH_A)).cls == sync.SAME
    assert prune_args("--apply") == 0
    assert folders[0].exists()
    assert "the local manifest has no run hash" in next((local / "sync-logs").iterdir()).read_text()


def test_share_file_symlink_to_local_not_removed(roots):
    local, share = roots
    folder = make_run(local)
    target = make_run(share)
    (target / "CFMresults.hdf5").unlink()
    (target / "CFMresults.hdf5").symlink_to(folder / "CFMresults.hdf5")
    assert prune_args("--apply") == 0
    assert (folder / "CFMresults.hdf5").read_bytes() == b"results"
    assert "symbolic link: CFMresults.hdf5" in next((local / "sync-logs").iterdir()).read_text()


def test_oserror_on_share_folder(roots, monkeypatch):
    local, share = roots
    folder = make_run(local)
    make_run(share)
    original = os.lstat

    def failing(path, *args, **kwargs):
        if str(path).startswith(str(share / "runs")):
            raise OSError(errno.EIO, "Input/output error")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", failing)
    assert prune_args("--apply") == 1
    assert folder.exists()
    assert "\tERROR\t" in next((local / "sync-logs").iterdir()).read_text()


def test_partial_removal_failed(roots, monkeypatch):
    local, share = roots
    folder = make_run(local)
    make_run(share)

    def failing(path, *args, **kwargs):
        os.remove(os.path.join(path, "run.log"))
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(prune.shutil, "rmtree", failing)
    assert prune_args("--apply") == 1
    assert "a part of the folder is possibly removed" in next((local / "sync-logs").iterdir()).read_text()
