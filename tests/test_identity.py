"""Tests for the CFM versions in config and the measured CFM identity (FEAT-001). Run from the repo root.

A small git repo in tmp_path stands in for the CFM checkout.
"""

import subprocess
import sys

import pytest

from cfm_config import cfm_version, config
import sim.identity
from sim.identity import changed_paths, check_cfm_identity, check_wrapper_identity, measure_cfm_identity, wrapper_identity
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


def test_untracked_code_in_new_folder_is_refused(cfm_repo):
    """git status shows a new folder as one line ("?? sub/"), so without --untracked-files=all its .py were missed."""
    (cfm_repo / "sub").mkdir()
    (cfm_repo / "sub" / "new.py").write_text("")
    with pytest.raises(RuntimeError, match="sub/new.py"):
        check_cfm_identity(identity(cfm_repo))


def test_code_diff_hash_ignores_data_changes(cfm_repo):
    clean = identity(cfm_repo)
    (cfm_repo / "CFMresults.hdf5").write_text("new data")
    changed = identity(cfm_repo)
    assert changed["diff_sha256"] != clean["diff_sha256"]
    assert changed["code_diff_sha256"] == clean["code_diff_sha256"]


def test_code_diff_hash_from_a_subfolder(cfm_repo):
    """The CFM repo in config.yaml is CFM_main, a subfolder of the checkout. The code diff must include all .py files."""
    sub = cfm_repo / "CFM_main"
    sub.mkdir()
    (sub / "keep.txt").write_text("")
    clean = identity(sub)["code_diff_sha256"]
    (cfm_repo / "melt.py").write_text("x = 2\n")
    assert identity(sub)["code_diff_sha256"] != clean


@pytest.fixture
def wrapper_repo(tmp_path, monkeypatch):
    """A committed checkout with src/, docs and scripts, in place of this repo."""
    repo = tmp_path / "wrapper"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "run.py").write_text("x = 1\n")
    (repo / "src" / "plot").mkdir()
    (repo / "src" / "plot" / "results.py").write_text("y = 1\n")
    (repo / "README.md").write_text("docs\n")
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    monkeypatch.setattr(sim.identity, "_REPO_ROOT", str(repo))
    return repo


def commit_all(repo):
    git(repo, "add", ".")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "change")


def test_run_code_ignores_commits_outside_it(wrapper_repo):
    before = wrapper_identity()
    (wrapper_repo / "README.md").write_text("new docs\n")
    (wrapper_repo / "src" / "plot" / "results.py").write_text("y = 2\n")
    commit_all(wrapper_repo)
    after = wrapper_identity()
    assert after["commit"] != before["commit"]
    assert after["run_code_sha256"] == before["run_code_sha256"]
    (wrapper_repo / "src" / "run.py").write_text("x = 2\n")
    commit_all(wrapper_repo)
    assert wrapper_identity()["run_code_sha256"] != before["run_code_sha256"]


def test_uncommitted_src_code_is_refused(wrapper_repo):
    check_wrapper_identity(wrapper_identity())
    (wrapper_repo / "README.md").write_text("new docs\n")
    (wrapper_repo / "src" / "plot" / "results.py").write_text("y = 2\n")
    (wrapper_repo / "src" / "plot" / "new.py").write_text("")
    check_wrapper_identity(wrapper_identity())
    (wrapper_repo / "src" / "sub").mkdir()
    (wrapper_repo / "src" / "sub" / "new.py").write_text("")
    with pytest.raises(RuntimeError, match="src/sub/new.py"):
        check_wrapper_identity(wrapper_identity())


def test_wrapper_not_a_git_checkout_is_refused():
    with pytest.raises(RuntimeError, match="not a git checkout"):
        check_wrapper_identity(None)


def test_cfm_run_does_not_import_the_excluded_code():
    """The run hash leaves out RUN_CODE_EXCLUDED (src/plot/). This is safe only if a CFM run does not import it."""
    code = "import sys, sim.worker, sim.run, sim.manifest, sim.identity; print(sorted(m.split('.')[0] for m in sys.modules))"
    loaded = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    for excluded in sim.identity.RUN_CODE_EXCLUDED:
        package = excluded.removeprefix("src/").rstrip("/")
        assert f"'{package}'" not in loaded
