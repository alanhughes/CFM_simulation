"""Tests for the CFM versions in config and the measured CFM identity (FEAT-001). Run from the repo root.

A small git repo in tmp_path stands in for the CFM checkout.
"""

import subprocess
import sys

import pytest

from cfm_config import cfm_version, config
from sim.identity import changed_paths, check_cfm_identity, measure_cfm_identity
from sim.run import import_cfm


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def cfm_repo(tmp_path):
    """A committed checkout with one .py file and one data file, like CFM_main."""
    repo = tmp_path / "CFM"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "melt.py").write_text("x = 1\n")
    (repo / "CFMresults.hdf5").write_text("data")
    git(repo, "add", ".")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    return repo


def identity(repo, allow_uncommitted=False):
    version = {"label": "v", "repo": str(repo), "python": sys.executable, "allow_uncommitted": allow_uncommitted}
    return measure_cfm_identity(version)


def test_clean_checkout_passes(cfm_repo):
    result = identity(cfm_repo)
    assert len(result["commit"]) == 40
    assert result["status"] == []
    assert result["code_changes"] == []
    assert not any("egg=cfm" in line or line.startswith("cfm") for line in result["pip_freeze"])
    check_cfm_identity(result)


def test_modified_code_is_refused(cfm_repo):
    (cfm_repo / "melt.py").write_text("x = 2\n")
    result = identity(cfm_repo)
    assert result["code_changes"] == ["melt.py"]
    with pytest.raises(RuntimeError, match="uncommitted code changes: melt.py"):
        check_cfm_identity(result)


def test_untracked_code_is_refused(cfm_repo):
    (cfm_repo / "new.py").write_text("")
    with pytest.raises(RuntimeError, match="new.py"):
        check_cfm_identity(identity(cfm_repo))


def test_allow_uncommitted_only_warns(cfm_repo, caplog):
    (cfm_repo / "melt.py").write_text("x = 2\n")
    result = identity(cfm_repo, allow_uncommitted=True)
    check_cfm_identity(result)
    assert "allow_uncommitted" in caplog.text


def test_diff_hash_changes_with_the_patch(cfm_repo):
    clean = identity(cfm_repo)["diff_sha256"]
    (cfm_repo / "melt.py").write_text("x = 2\n")
    patch_a = identity(cfm_repo)["diff_sha256"]
    (cfm_repo / "melt.py").write_text("x = 3\n")
    assert len({clean, patch_a, identity(cfm_repo)["diff_sha256"]}) == 3


def test_non_code_changes_only_warn(cfm_repo, caplog):
    (cfm_repo / "CFMresults.hdf5").write_text("new data")
    check_cfm_identity(identity(cfm_repo))
    assert "not code: CFMresults.hdf5" in caplog.text


def test_not_a_git_checkout(tmp_path):
    with pytest.raises(RuntimeError, match="not a git checkout"):
        identity(tmp_path)


def test_changed_paths_rename():
    assert changed_paths(" M a.py\nR  old.py -> new.py\n?? b.csv") == ["a.py", "new.py", "b.csv"]


def test_cfm_version_lookup(monkeypatch):
    versions = {
        "a": {"repo": "/a", "python": "/a/python"},
        "b": {"repo": "/b", "python": "/b/python", "allow_uncommitted": True},
    }
    monkeypatch.setitem(config, "cfm_versions", versions)
    monkeypatch.setitem(config, "default_cfm_version", "a")
    assert cfm_version() == {"label": "a", "repo": "/a", "python": "/a/python", "allow_uncommitted": False}
    assert cfm_version("b")["allow_uncommitted"] is True
    with pytest.raises(ValueError, match="Unknown CFM version 'c'"):
        cfm_version("c")


def test_cfm_version_from_old_config(monkeypatch):
    monkeypatch.delitem(config, "cfm_versions", raising=False)
    monkeypatch.setitem(config, "CFM_repo_path", "/old/CFM_main")
    with pytest.warns(UserWarning, match="no cfm_versions"):
        version = cfm_version()
    assert version == {"label": "default", "repo": "/old/CFM_main", "python": sys.executable, "allow_uncommitted": False}


def test_second_cfm_in_one_process_is_refused(tmp_path, monkeypatch):
    loaded = type(sys)("firn_density_nospin")
    loaded.__file__ = str(tmp_path / "one" / "firn_density_nospin.py")
    monkeypatch.setitem(sys.modules, "firn_density_nospin", loaded)
    with pytest.raises(RuntimeError, match="already imported the CFM"):
        import_cfm(str(tmp_path / "two"))


def test_pip_freeze_ignores_pythonpath(cfm_repo, tmp_path, monkeypatch):
    """A stale egg-info with bad metadata on PYTHONPATH (as in src/ of an old checkout) must not break pip freeze."""
    bad = tmp_path / "src" / "cfm.egg-info"
    bad.mkdir(parents=True)
    (bad / "requires.txt").write_text("numpy\n")  # no PKG-INFO, so no version
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "src"))
    assert identity(cfm_repo)["pip_freeze"]
