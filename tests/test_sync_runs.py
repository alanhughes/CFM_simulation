"""Tests for scripts/sync_runs.py (FEAT-008). Run from the repo root.

The local root and the share root are two temporary folders. The .hdf5 files are tiny fake files, and the dataset
comparison of compare_cfm_runs is a stub.
"""

import errno
import importlib.util
import os
import shutil
from pathlib import Path

import pytest

import sim.runs
from cfm_config import config
from sim.manifest import COMPLETE, read_manifest, write_manifest

_spec = importlib.util.spec_from_file_location(
    "sync_runs", Path(__file__).resolve().parents[1] / "scripts" / "sync_runs.py"
)
sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync)

REL = os.path.join("MAR_119_17", "GSFC2020_bucket_1979-2024", "ab871c8", "aaaaaaaa")
HASH = "a" * 64


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


def make_run(root, rel=REL, sites=("CI-0",), status=COMPLETE, hash_=HASH, data=b"results", extra_run=None):
    folder = Path(root) / "runs" / rel
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "CFMresults.hdf5").write_bytes(data)
    (folder / "run.log").write_text("log\n")
    run = {"sites": list(sites), "site_coords": {s: [-70.0 + i, 20.0] for i, s in enumerate(sites)}}
    run.update(extra_run or {})
    write_manifest(
        str(folder),
        {"status": status, "run_hash": hash_, "finished": "2026-10-25T10:00:00", "cfm": {"commit": "abc"}, "run": run},
    )
    return folder


def sync_args(*args):
    return sync.main(list(args))


def tree(root):
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob("*"))


def test_new_is_copied(roots, capsys):
    local, share = roots
    make_run(local)
    assert sync_args() == 0
    assert tree(share) == []  # the dry run writes nothing
    assert not (local / "sync-logs").exists()
    assert sync_args("--apply") == 0
    target = share / "runs" / REL
    assert (target / "CFMresults.hdf5").read_bytes() == b"results"
    assert read_manifest(str(target)) == read_manifest(str(local / "runs" / REL))
    assert not (share / ".incoming" / REL).exists()
    (log,) = (local / "sync-logs").iterdir()
    assert "NEW" in log.read_text() and "copied" in log.read_text()
    assert len(sim.runs.find_runs(root=str(share / "runs"))) == 1


def test_not_complete_local_run_is_ignored(roots):
    local, share = roots
    make_run(local, status="failed")
    assert sync_args("--apply") == 0
    assert not (share / "runs" / REL).exists()


def test_same(roots):
    local, share = roots
    make_run(local)
    make_run(share)
    before = tree(share)
    assert sync_args("--apply") == 0
    assert tree(share) == before
    assert "SAME" in next((local / "sync-logs").iterdir()).read_text()


def test_tmp_manifest_in_share_is_ignored(roots, capsys):
    local, share = roots
    make_run(local)
    make_run(share)
    (share / "runs" / REL / "manifest.json.tmp").write_text("{")
    assert sync_args() == 0
    assert "SAME" in capsys.readouterr().out


def test_sites_union_keeps_share_order(roots):
    local, share = roots
    make_run(local, sites=("CI-22", "CI-0", "WI-70"))
    make_run(share, sites=("CI-0", "CI-5"))
    share_coords = read_manifest(str(share / "runs" / REL))["run"]["site_coords"]
    assert sync_args("--apply") == 0
    run = read_manifest(str(share / "runs" / REL))["run"]
    assert run["sites"] == ["CI-0", "CI-5", "CI-22", "WI-70"]
    assert run["site_coords"]["CI-0"] == share_coords["CI-0"]
    assert set(run["site_coords"]) == {"CI-0", "CI-5", "CI-22", "WI-70"}
    assert not (share / "runs" / REL / "manifest.json.tmp").exists()


def test_share_has_all_sites_is_same(roots, capsys):
    local, share = roots
    make_run(local, sites=("CI-0",))
    make_run(share, sites=("CI-0", "CI-22"))
    assert sync_args() == 0
    assert "SAME" in capsys.readouterr().out


def test_old_manifest_without_site_coords(roots):
    local, share = roots
    make_run(share, sites=("CI-0",), extra_run={"site_coords": None, "borehole_lat": -70.0, "borehole_lon": 20.0})
    make_run(local, sites=("CI-0",), extra_run={"site_coords": None, "borehole_lat": -70.0, "borehole_lon": 20.0})
    assert sync.classify_run(str(local / "runs" / REL), str(share / "runs" / REL), REL).cls == sync.SAME


def test_conflict_exit_code(roots):
    local, share = roots
    make_run(local)
    make_run(share, hash_="b" * 64)
    assert sync_args() == 1
    assert sync_args("--apply") == 1
    assert read_manifest(str(share / "runs" / REL))["run_hash"] == "b" * 64


@pytest.mark.parametrize("make_share", ["data", "status", "extra_file", "no_manifest"])
def test_differ_classes(roots, make_share):
    local, share = roots
    make_run(local)
    kwargs = {"data": {"data": b"other"}, "status": {"status": "failed"}, "extra_file": {}, "no_manifest": {}}[make_share]
    folder = make_run(share, **kwargs)
    if make_share == "extra_file":
        (folder / "extra.hdf5").write_bytes(b"x")
    if make_share == "no_manifest":
        os.remove(folder / "manifest.json")
    assert sync.classify_run(str(local / "runs" / REL), str(folder), REL).cls == sync.DIFFER


def test_differ_report(roots, capsys):
    local, share = roots
    make_run(local)
    make_run(share, data=b"other")
    sync_args()
    out = capsys.readouterr().out
    assert "finished 2026-10-25T10:00:00" in out and "CFM commit abc" in out
    assert "CFMresults.hdf5" in out and "all datasets identical: no" in out


def test_differ_keep_and_skip(roots):
    local, share = roots
    make_run(local)
    make_run(share, data=b"other")
    for decision in ("keep", "skip"):
        assert sync_args("--apply", "--on-differ", decision) == 0
        assert (share / "runs" / REL / "CFMresults.hdf5").read_bytes() == b"other"
        assert not (share / "replaced").exists()


def test_differ_replace(roots):
    local, share = roots
    make_run(local)
    make_run(share, data=b"other")
    assert sync_args("--apply", "--on-differ", "replace") == 0
    assert (share / "runs" / REL / "CFMresults.hdf5").read_bytes() == b"results"
    (day,) = (share / "replaced").iterdir()
    assert (day / REL / "CFMresults.hdf5").read_bytes() == b"other"
    assert not (share / ".incoming" / REL).exists()


def test_replace_refused_if_replaced_exists(roots):
    local, share = roots
    make_run(local)
    make_run(share, data=b"other")
    today = sync.datetime.date.today().isoformat()
    (share / "replaced" / today / REL).mkdir(parents=True)
    assert sync_args("--apply", "--on-differ", "replace") == 1
    assert (share / "runs" / REL / "CFMresults.hdf5").read_bytes() == b"other"


def test_ask_without_terminal_stops_before_copies(roots, capsys):
    local, share = roots
    make_run(local)
    other = "MAR_119_17/GSFC2020_bucket_1979-2024/ab871c8/bbbbbbbb"
    make_run(local, rel=other, hash_="b" * 64)
    make_run(share, data=b"other")
    assert sync_args("--apply") == 1
    assert not (share / "runs" / other).exists()
    assert not (share / ".incoming").exists()
    # the dry run does not need a decision
    assert sync_args() == 0


def test_ask_with_terminal(roots, monkeypatch):
    local, share = roots
    make_run(local)
    make_run(share, data=b"other")

    class Tty:
        def isatty(self):
            return True

    monkeypatch.setattr(sync.sys, "stdin", Tty())
    answers = iter(["x", "r"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert sync_args("--apply") == 0
    assert (share / "runs" / REL / "CFMresults.hdf5").read_bytes() == b"results"


def test_copy_stops_before_rename(roots, monkeypatch):
    local, share = roots
    make_run(local)

    def stop(source, target):
        raise RuntimeError("network stopped")

    monkeypatch.setattr(sync, "_rename", stop)
    assert sync_args("--apply") == 1
    assert (share / ".incoming" / REL / "manifest.json").exists()
    assert not (share / "runs" / REL).exists()
    assert sim.runs.find_runs(root=str(share / "runs")) == []
    monkeypatch.undo()
    (share / ".incoming" / REL / "stale.txt").write_text("old")
    monkeypatch.setitem(config, "CFM_data_path", str(local))
    monkeypatch.setitem(config, "CFM_share_data_path", str(share))
    assert sync_args("--apply") == 0
    assert (share / "runs" / REL / "manifest.json").exists()
    assert not (share / "runs" / REL / "stale.txt").exists()
    assert not (share / ".incoming" / REL).exists()


def test_failed_sha_check_stays_in_incoming(roots, monkeypatch):
    local, share = roots
    make_run(local)

    def bad_copy(source, target):
        Path(target).write_bytes(b"x" * os.path.getsize(source))

    monkeypatch.setattr(sync, "_copy_file", bad_copy)
    assert sync_args("--apply") == 1
    assert (share / ".incoming" / REL).exists()
    assert not (share / "runs" / REL).exists()
    assert "FAILED" in next((local / "sync-logs").iterdir()).read_text()


def test_target_appears_before_rename(roots, monkeypatch):
    local, share = roots
    make_run(local)
    real = sync._copy_checked

    def copy_then_other_vm(source, target):
        real(source, target)
        if source.endswith("manifest.json"):
            shutil.copytree(local / "runs" / REL, share / "runs" / REL)

    monkeypatch.setattr(sync, "_copy_checked", copy_then_other_vm)
    assert sync_args("--apply") == 1
    assert (share / ".incoming" / REL).exists()


def test_share_not_set_or_not_available(roots, tmp_path):
    local, share = roots
    make_run(local)
    config["CFM_share_data_path"] = str(tmp_path / "missing")
    assert sync_args() == 1
    del config["CFM_share_data_path"]
    assert sync_args() == 1


# the tests of the review of the first version


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in Path(root).rglob("*") if p.is_file()}


def test_sites_not_lost_when_a_site_is_added_during_the_sync(roots, monkeypatch):
    local, share = roots
    make_run(local, sites=("CI-0", "CI-22"))
    make_run(share, sites=("CI-0",))
    real = sync.classify_run

    def classify_then_add(local_dir, target, rel):
        run = real(local_dir, target, rel)
        manifest = read_manifest(target)  # another VM: _add_site during the long classification
        manifest["run"]["sites"].append("WI-70")
        manifest["run"]["site_coords"]["WI-70"] = [1, 2]
        write_manifest(target, manifest)
        return run

    monkeypatch.setattr(sync, "classify_run", classify_then_add)
    assert sync_args("--apply") == 0
    run = read_manifest(str(share / "runs" / REL))["run"]
    assert run["sites"] == ["CI-0", "WI-70", "CI-22"]
    assert run["site_coords"]["WI-70"] == [1, 2]


def test_sites_fails_if_the_share_manifest_changed_in_another_key(roots, monkeypatch):
    local, share = roots
    make_run(local, sites=("CI-0", "CI-22"))
    make_run(share, sites=("CI-0",))
    real = sync.classify_run

    def classify_then_change(local_dir, target, rel):
        run = real(local_dir, target, rel)
        manifest = read_manifest(target)
        manifest["finished"] = "2027-01-01T00:00:00"
        write_manifest(target, manifest)
        return run

    monkeypatch.setattr(sync, "classify_run", classify_then_change)
    assert sync_args("--apply") == 1
    assert read_manifest(str(share / "runs" / REL))["run"]["sites"] == ["CI-0"]
    assert "FAILED" in next((local / "sync-logs").iterdir()).read_text()


def test_io_error_on_target_is_error_not_new(roots, monkeypatch, capsys):
    local, share = roots
    make_run(local)
    make_run(share, data=b"other")
    target = str(share / "runs" / REL)
    real = os.lstat

    def lstat(path, *args, **kwargs):
        if str(path) == target:
            raise OSError(errno.EIO, "input/output error")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", lstat)
    assert sync_args("--apply") == 1
    assert "ERROR" in capsys.readouterr().out
    assert not (share / ".incoming").exists()
    assert (share / "runs" / REL / "CFMresults.hdf5").read_bytes() == b"other"


def test_stale_file_in_incoming_does_not_reach_target(roots, monkeypatch):
    local, share = roots
    make_run(local)
    incoming = share / ".incoming" / REL
    incoming.mkdir(parents=True)
    (incoming / "stale.txt").write_text("x")
    real = sync._exists
    # as if the check of .incoming did not see the folder, so that it is not removed
    monkeypatch.setattr(sync, "_exists", lambda path: False if path == str(incoming) else real(path))
    assert sync_args("--apply") == 1
    assert not (share / "runs" / REL).exists()
    assert (incoming / "stale.txt").exists()


def test_differ_report_survives_an_oserror(roots, monkeypatch, capsys):
    local, share = roots
    make_run(local)
    make_run(share, data=b"other")

    def summary(new, ref):
        raise OSError(errno.EIO, "input/output error")

    monkeypatch.setattr(sync.compare, "summary", summary)
    assert sync_args("--on-differ", "keep") == 0
    assert sync_args("--apply", "--on-differ", "keep") == 0
    assert "not available" in capsys.readouterr().out


def test_differ_report_does_not_read_the_files_again(roots, monkeypatch, capsys):
    local, share = roots
    make_run(local)
    make_run(share, data=b"other")
    run = sync.classify_run(str(local / "runs" / REL), str(share / "runs" / REL), REL)

    def fail(path):
        raise OSError(errno.EIO, "input/output error")

    monkeypatch.setattr(sync, "sha256_file", fail)
    sync.describe_differ(run)
    out = capsys.readouterr().out
    assert "7 bytes, SHA-256" in out and "5 bytes, SHA-256" in out


def test_dry_run_writes_nothing_for_sites_differ_conflict(roots):
    local, share = roots
    rels = [REL.replace("aaaaaaaa", name) for name in ("aaaaaaaa", "bbbbbbbb", "cccccccc")]
    make_run(local, rel=rels[0], sites=("CI-0", "CI-22"))
    make_run(share, rel=rels[0], sites=("CI-0",))
    make_run(local, rel=rels[1], hash_="b" * 64)
    make_run(share, rel=rels[1], hash_="b" * 64, data=b"other")
    make_run(local, rel=rels[2], hash_="c" * 64)
    make_run(share, rel=rels[2], hash_="d" * 64)
    before_local, before_share = snapshot(local), snapshot(share)
    assert sync_args() == 1  # the CONFLICT
    assert snapshot(local) == before_local and snapshot(share) == before_share
    assert not (local / "sync-logs").exists() and not (share / ".incoming").exists()


def test_local_tree_unchanged_after_apply_and_replace(roots):
    local, share = roots
    make_run(local)
    other = REL.replace("aaaaaaaa", "bbbbbbbb")
    make_run(local, rel=other, hash_="b" * 64)
    make_run(share, rel=other, hash_="b" * 64, data=b"other")
    before = snapshot(local)
    assert sync_args("--apply", "--on-differ", "replace") == 0
    after = {name: data for name, data in snapshot(local).items() if not name.startswith("sync-logs")}
    assert after == before


@pytest.mark.parametrize("key", ["finished", "commit"])
def test_different_manifest_field_is_differ(roots, key):
    local, share = roots
    make_run(local)
    folder = make_run(share)
    manifest = read_manifest(str(folder))
    if key == "finished":
        manifest["finished"] = "2026-10-26T10:00:00"
    else:
        manifest["cfm"]["commit"] = "def"
    write_manifest(str(folder), manifest)
    run = sync.classify_run(str(local / "runs" / REL), str(folder), REL)
    assert run.cls == sync.DIFFER and not run.diff_files


def test_file_only_in_local_copy_is_differ(roots):
    local, share = roots
    folder = make_run(local)
    (folder / "extra.hdf5").write_bytes(b"x")
    make_run(share)
    run = sync.classify_run(str(folder), str(share / "runs" / REL), REL)
    assert run.cls == sync.DIFFER and run.diff_files == ["extra.hdf5"]


def test_conflict_does_not_read_the_files(roots, monkeypatch):
    local, share = roots
    make_run(local)
    make_run(share, hash_="b" * 64)
    monkeypatch.setattr(sync, "sha256_file", lambda path: 1 / 0)
    assert sync.classify_run(str(local / "runs" / REL), str(share / "runs" / REL), REL).cls == sync.CONFLICT


def test_share_manifest_without_run_hash_is_conflict(roots):
    local, share = roots
    make_run(local)
    folder = make_run(share)
    manifest = read_manifest(str(folder))
    del manifest["run_hash"]
    write_manifest(str(folder), manifest)
    assert sync.classify_run(str(local / "runs" / REL), str(folder), REL).cls == sync.CONFLICT


def test_error_class_gives_exit_1(roots, monkeypatch):
    local, share = roots
    make_run(local)

    def fail(local_dir, target, rel):
        raise OSError(errno.EIO, "input/output error")

    monkeypatch.setattr(sync, "classify_run", fail)
    assert sync_args() == 1


def test_other_incoming_folder_survives(roots):
    local, share = roots
    make_run(local)
    other = share / ".incoming" / "MAR_1_1" / "x" / "y" / "zzzzzzzz"
    other.mkdir(parents=True)
    (other / "file").write_text("keep")
    assert sync_args("--apply") == 0
    assert (other / "file").read_text() == "keep"


def test_replace_second_rename_fails(roots, monkeypatch):
    local, share = roots
    make_run(local)
    make_run(share, data=b"other")
    calls = []
    real = sync._rename

    def rename(source, target):
        calls.append(target)
        if len(calls) == 2:
            raise OSError(errno.EIO, "input/output error")
        real(source, target)

    monkeypatch.setattr(sync, "_rename", rename)
    assert sync_args("--apply", "--on-differ", "replace") == 1
    (day,) = (share / "replaced").iterdir()
    assert (day / REL / "CFMresults.hdf5").read_bytes() == b"other"
    assert (share / ".incoming" / REL / "CFMresults.hdf5").read_bytes() == b"results"
    assert not (share / "runs" / REL).exists()
    log = next((local / "sync-logs").iterdir()).read_text()
    assert "replaced" in log and ".incoming" in log


def test_apply_makes_share_runs(roots):
    local, share = roots
    assert not (share / "runs").exists()
    assert sync_args("--apply") == 0
    assert (share / "runs").is_dir()


def test_fadvise_warning_once(roots, monkeypatch, capsys):
    local, share = roots
    make_run(local)

    def fadvise(fd, offset, length, advice):
        raise OSError(errno.EINVAL, "invalid")

    monkeypatch.setattr(sync, "_warned_fadvise", False)
    monkeypatch.setattr(os, "posix_fadvise", fadvise)
    assert sync_args("--apply") == 0
    assert capsys.readouterr().err.count("posix_fadvise") == 1


def test_log_is_not_overwritten_and_has_notes(roots, monkeypatch):
    local, share = roots
    make_run(local)
    make_run(share, data=b"other")

    class FixedTime(sync.datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 28, 12, 0, 0)

    monkeypatch.setattr(sync.datetime, "datetime", FixedTime)
    assert sync_args("--apply", "--on-differ", "keep") == 0
    assert sync_args("--apply", "--on-differ", "keep") == 0
    logs = sorted((local / "sync-logs").iterdir())
    assert len(logs) == 2
    assert "CFMresults.hdf5: different" in logs[0].read_text()


def test_log_written_when_decision_is_missing(roots):
    local, share = roots
    make_run(local)
    make_run(share, data=b"other")
    assert sync_args("--apply") == 1
    assert "no decision" in next((local / "sync-logs").iterdir()).read_text()


def test_ask_eof_gives_no_decision(roots, monkeypatch):
    local, share = roots
    make_run(local)
    make_run(share, data=b"other")

    class Tty:
        def isatty(self):
            return True

    def eof(prompt):
        raise EOFError

    monkeypatch.setattr(sync.sys, "stdin", Tty())
    monkeypatch.setattr("builtins.input", eof)
    assert sync_args("--apply") == 1
    assert (share / "runs" / REL / "CFMresults.hdf5").read_bytes() == b"other"


# the forcing CSVs and the melt maps (second commit)

from types import SimpleNamespace  # noqa: E402

from force.cache import find_forcing_csv  # noqa: E402

MP = SimpleNamespace(rcm_name="MAR", release="v3.12", i=121, j=13)
CSV = os.path.join("MAR", "v3.12", "121_13_1979_2024.csv")
JSON = os.path.join("MAR", "v3.12", "121_13_1979_2024.json")
MELT = "MAR_avg_yearly_melt_map_1979-2024.nc"


def make_forcing(root, data=b"csv", json_data=b"{}", folder="forcing"):
    for rel, content in ((CSV, data), (JSON, json_data)):
        path = Path(root) / folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def test_forcing_new_copied_and_found(roots, monkeypatch):
    local, share = roots
    make_forcing(local)
    assert sync_args() == 0
    assert tree(share) == []  # the dry run writes nothing
    assert sync_args("--apply") == 0
    assert (share / "forcing" / CSV).read_bytes() == b"csv"
    assert (share / "forcing" / JSON).read_bytes() == b"{}"
    assert not (share / ".incoming" / "forcing" / CSV).exists()
    assert str(local) in str(sync.forcing_root())
    shutil.rmtree(local / "forcing")
    assert find_forcing_csv(MP, 1979, 2024) == str(share / "forcing" / CSV)
    log = next((local / "sync-logs").iterdir()).read_text()
    assert f"forcing{os.sep}{CSV}\tNEW\t-\tcopied" in log


def test_forcing_same_and_other_files_ignored(roots, capsys):
    local, share = roots
    make_forcing(local)
    make_forcing(share)
    (local / "forcing" / "notes.txt").write_text("x")
    before = snapshot(share)
    assert sync_args("--apply") == 0
    assert snapshot(share) == before
    assert "Files: NEW 0, SAME 2" in capsys.readouterr().out


def test_forcing_different_is_refused_and_json_first(roots, capsys):
    local, share = roots
    make_forcing(local)
    make_forcing(share, json_data=b'{"other": 1}')  # the JSON differs, the CSV is the same
    (share / "forcing" / CSV).unlink()  # the CSV would be NEW
    assert sync_args("--apply") == 1
    assert (share / "forcing" / JSON).read_bytes() == b'{"other": 1}'
    assert not (share / "forcing" / CSV).exists()  # not copied: its JSON is refused
    assert not (share / ".incoming").exists()
    make_forcing(share, data=b"other")
    assert sync_args("--apply") == 1
    assert (share / "forcing" / CSV).read_bytes() == b"other"


def test_json_copied_before_csv(roots, monkeypatch):
    local, share = roots
    make_forcing(local)
    order = []
    real = sync._publish
    monkeypatch.setattr(sync, "_publish", lambda s, t: (order.append(os.path.basename(t)), real(s, t)))
    assert sync_args("--apply") == 0
    assert order == ["121_13_1979_2024.json", "121_13_1979_2024.csv"]


def test_csv_without_json_is_error(roots):
    local, share = roots
    make_forcing(local)
    (local / "forcing" / JSON).unlink()
    assert sync_args("--apply") == 1
    assert not (share / "forcing" / CSV).exists()


def test_forcing_data_path_not_default(roots, tmp_path, monkeypatch):
    local, share = roots
    monkeypatch.setitem(config, "forcing_data_path", str(tmp_path / "elsewhere"))
    make_forcing(tmp_path, folder="elsewhere")
    assert sync_args("--apply") == 0
    assert (share / "forcing" / CSV).read_bytes() == b"csv"
    shutil.rmtree(tmp_path / "elsewhere")
    assert find_forcing_csv(MP, 1979, 2024) == str(share / "forcing" / CSV)


def test_melt_new_same_different(roots):
    local, share = roots
    (local / "melt").mkdir()
    (local / "melt" / MELT).write_bytes(b"nc")
    (local / "melt" / "readme.txt").write_text("x")
    assert sync_args() == 0
    assert tree(share) == []
    assert sync_args("--apply") == 0
    assert (share / "melt" / MELT).read_bytes() == b"nc"
    assert not (share / "melt" / "readme.txt").exists()
    assert sync_args("--apply") == 0  # SAME
    (local / "melt" / MELT).write_bytes(b"new nc")
    assert sync_args("--apply") == 1  # different: never overwritten
    assert (share / "melt" / MELT).read_bytes() == b"nc"


def test_no_local_melt_or_forcing_folder(roots):
    assert sync_args("--apply") == 0


def test_file_copy_stopped_before_rename(roots, monkeypatch):
    local, share = roots
    (local / "melt").mkdir()
    (local / "melt" / MELT).write_bytes(b"nc")

    def stop(source, target):
        raise RuntimeError("network stopped")

    with monkeypatch.context() as patch:
        patch.setattr(sync, "_publish", stop)
        assert sync_args("--apply") == 1
    assert (share / ".incoming" / "melt" / MELT).exists()
    assert not (share / "melt" / MELT).exists()
    assert sync_args("--apply") == 0
    assert (share / "melt" / MELT).read_bytes() == b"nc"
    assert not (share / ".incoming" / "melt" / MELT).exists()


def test_file_target_appears_before_rename(roots, monkeypatch):
    local, share = roots
    (local / "melt").mkdir()
    (local / "melt" / MELT).write_bytes(b"nc")
    real = sync._copy_checked

    def copy_then_other_vm(source, target):
        real(source, target)
        (share / "melt").mkdir(exist_ok=True)
        (share / "melt" / MELT).write_bytes(b"nc")

    monkeypatch.setattr(sync, "_copy_checked", copy_then_other_vm)
    assert sync_args("--apply") == 1
    assert (share / ".incoming" / "melt" / MELT).exists()


def test_missing_decision_also_stops_the_files(roots):
    local, share = roots
    make_run(local)
    make_run(share, data=b"other")
    make_forcing(local)
    assert sync_args("--apply") == 1
    assert not (share / "forcing").exists()


# the review of the second commit


def forcing_json(sha, **meta):
    import json

    return json.dumps({"csv_sha256": sha, **meta}).encode()


def test_json_with_other_metadata_is_same(roots, capsys):
    local, share = roots
    make_forcing(local, json_data=forcing_json("a" * 64, host="vm1", created="1"))
    make_forcing(share, json_data=forcing_json("a" * 64, host="vm2", created="2"))
    before = snapshot(share)
    assert sync_args("--apply") == 0
    assert snapshot(share) == before
    assert "Files: NEW 0, SAME 2" in capsys.readouterr().out
    assert "metadata" in next((local / "sync-logs").iterdir()).read_text()


def test_json_with_other_csv_sha256_is_conflict(roots):
    local, share = roots
    make_forcing(local, json_data=forcing_json("a" * 64))
    make_forcing(share, json_data=forcing_json("b" * 64))
    assert sync_args("--apply") == 1
    assert read_json(share / "forcing" / JSON)["csv_sha256"] == "b" * 64


def test_json_metadata_differs_but_csv_differs_is_conflict(roots):
    local, share = roots
    make_forcing(local, json_data=forcing_json("a" * 64, host="1"))
    make_forcing(share, data=b"other", json_data=forcing_json("a" * 64, host="2"))
    assert sync_args("--apply") == 1
    assert (share / "forcing" / CSV).read_bytes() == b"other"
    assert read_json(share / "forcing" / JSON)["host"] == "2"


def read_json(path):
    import json

    return json.loads(Path(path).read_text())


def test_csv_refused_then_json_not_copied(roots):
    local, share = roots
    make_forcing(local)
    (share / "forcing" / CSV).parent.mkdir(parents=True)
    (share / "forcing" / CSV).write_bytes(b"other")
    assert sync_args("--apply") == 1
    assert not (share / "forcing" / JSON).exists()
    assert not (share / ".incoming").exists()
    assert "refused" in next((local / "sync-logs").iterdir()).read_text()


def test_failed_json_copy_stops_its_csv(roots, monkeypatch):
    local, share = roots
    make_forcing(local)
    real = sync._copy_checked

    def fail_json(source, target):
        if target.endswith(".json"):
            raise RuntimeError("network stopped")
        real(source, target)

    monkeypatch.setattr(sync, "_copy_checked", fail_json)
    assert sync_args("--apply") == 1
    assert not (share / "forcing" / CSV).exists()
    assert not (share / ".incoming" / "forcing" / CSV).exists()


def test_csv_without_local_json_but_share_json_is_copied(roots):
    local, share = roots
    make_forcing(local)
    (local / "forcing" / JSON).unlink()
    (share / "forcing" / JSON).parent.mkdir(parents=True)
    (share / "forcing" / JSON).write_bytes(b"{}")
    assert sync_args("--apply") == 0
    assert (share / "forcing" / CSV).read_bytes() == b"csv"


def test_unreadable_forcing_folder_does_not_stop_the_runs(roots, monkeypatch, capsys):
    local, share = roots
    make_run(local)
    make_forcing(local)

    real = sync.all_files

    def listing(folder):
        if folder == sync.forcing_root():
            raise OSError(errno.EIO, "input/output error")
        return real(folder)

    monkeypatch.setattr(sync, "all_files", listing)
    assert sync_args("--apply") == 1
    assert (share / "runs" / REL / "manifest.json").exists()
    assert "cannot be listed" in capsys.readouterr().err
    assert not (share / "forcing").exists()


def test_file_target_untouched_when_it_appears(roots, monkeypatch):
    local, share = roots
    (local / "melt").mkdir()
    (local / "melt" / MELT).write_bytes(b"nc")
    real = sync._copy_checked

    def copy_then_other_vm(source, target):
        real(source, target)
        (share / "melt").mkdir(exist_ok=True)
        (share / "melt" / MELT).write_bytes(b"from the other VM")

    monkeypatch.setattr(sync, "_copy_checked", copy_then_other_vm)
    assert sync_args("--apply") == 1
    assert (share / "melt" / MELT).read_bytes() == b"from the other VM"
    assert (share / ".incoming" / "melt" / MELT).read_bytes() == b"nc"


def test_publish_without_hard_links_renames(roots, monkeypatch, tmp_path):
    def no_link(source, target):
        raise OSError(errno.EPERM, "not permitted")

    monkeypatch.setattr(os, "link", no_link)
    (tmp_path / "a").write_bytes(b"1")
    sync._publish(str(tmp_path / "a"), str(tmp_path / "b"))
    assert (tmp_path / "b").read_bytes() == b"1" and not (tmp_path / "a").exists()
    (tmp_path / "a").write_bytes(b"2")
    with pytest.raises(RuntimeError):
        sync._publish(str(tmp_path / "a"), str(tmp_path / "b"))
    assert (tmp_path / "b").read_bytes() == b"1"


def test_io_error_on_file_target_is_error(roots, monkeypatch):
    local, share = roots
    (local / "melt").mkdir()
    (local / "melt" / MELT).write_bytes(b"nc")
    target = str(share / "melt" / MELT)
    real = os.lstat

    def lstat(path, *args, **kwargs):
        if str(path) == target:
            raise OSError(errno.EIO, "input/output error")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", lstat)
    assert sync_args("--apply") == 1
    assert not (share / ".incoming").exists()


def test_file_sha_mismatch_is_failed_and_not_published(roots, monkeypatch):
    local, share = roots
    (local / "melt").mkdir()
    (local / "melt" / MELT).write_bytes(b"nc")

    def bad_copy(source, target):
        Path(target).write_bytes(b"xx")

    monkeypatch.setattr(sync, "_copy_file", bad_copy)
    assert sync_args("--apply") == 1
    assert (share / ".incoming" / "melt" / MELT).exists()
    assert not (share / "melt" / MELT).exists()
    assert "FAILED" in next((local / "sync-logs").iterdir()).read_text()
