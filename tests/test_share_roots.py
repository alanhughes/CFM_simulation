"""Tests for the optional key CFM_share_data_path and the run roots, share_runs_root() and run_roots() (FEAT-008).

The two roots are temporary folders. The real share is never read: CFM_share_data_path is set to a folder in tmp_path,
and a share that is not available is a stub of os.stat that raises OSError. tests/conftest.py clears the result of the
check of the share root before each test.
"""

import errno
import logging
import os

import pytest

import sim.manifest
from cfm_config import config, expand_cpom_mount, share_data_path
from sim.manifest import reset_share_check, run_roots, runs_root, share_runs_root


@pytest.fixture
def local(tmp_path, monkeypatch):
    """CFM_data_path in tmp_path, with its runs/ folder, and no CFM_share_data_path (as in the test config)."""
    data = tmp_path / "local"
    (data / "runs").mkdir(parents=True)
    monkeypatch.setitem(config, "CFM_data_path", str(data))
    return data


@pytest.fixture
def share(tmp_path, monkeypatch, local):
    """CFM_share_data_path in tmp_path. The runs/ folder is not made."""
    data = tmp_path / "share"
    monkeypatch.setitem(config, "CFM_share_data_path", str(data))
    return data


# the config


def test_expand_cpom_mount_in_share_path():
    cfg = {
        "cpom_mount": "/mnt/CPOM",
        "CFM_data_path": "/home/user/CFM_data",
        "CFM_share_data_path": "{cpom_mount}/user/CFM_data",
        "forcing_data_path": "{CFM_data_path}/forcing",
    }
    expand_cpom_mount(cfg)
    assert cfg["CFM_share_data_path"] == "/mnt/CPOM/user/CFM_data"
    assert share_data_path(cfg) == "/mnt/CPOM/user/CFM_data"
    # the other placeholders are filled in where they are used
    assert cfg["forcing_data_path"] == "{CFM_data_path}/forcing"


def test_expand_cpom_mount_without_share_path():
    """A config.yaml without the key gives the same config as before FEAT-008."""
    cfg = {"cpom_mount": "/mnt/CPOM", "CFM_data_path": "/home/user/CFM_data", "in_situ_data_path": "{cpom_mount}/x"}
    assert expand_cpom_mount(cfg) == {
        "cpom_mount": "/mnt/CPOM",
        "CFM_data_path": "/home/user/CFM_data",
        "in_situ_data_path": "/mnt/CPOM/x",
    }
    assert "CFM_share_data_path" not in cfg
    assert share_data_path(cfg) is None


@pytest.mark.parametrize("value", [None, ""])
def test_empty_share_path_is_not_set(value):
    """"CFM_share_data_path:" with no value (None in YAML) or "" is the same as no key."""
    assert share_data_path({"cpom_mount": "/mnt/CPOM", "CFM_share_data_path": value}) is None


# the roots


def test_key_not_set(local):
    assert share_runs_root() is None
    assert run_roots() == [runs_root()]
    assert runs_root() == str(local / "runs")


def test_key_set_with_share_root(share):
    (share / "runs").mkdir(parents=True)
    assert share_runs_root() == str(share / "runs")
    assert run_roots() == [runs_root(), str(share / "runs")]


def test_share_root_missing(share):
    """The share folder exists, but has no runs/ folder."""
    share.mkdir()
    assert share_runs_root() == str(share / "runs")
    assert run_roots() == [runs_root()]


def test_share_root_is_a_file(share):
    share.mkdir()
    (share / "runs").write_text("")
    assert run_roots() == [runs_root()]


def test_share_not_available(share, monkeypatch, caplog):
    """A "soft" mount that stops gives an OSError after a timeout, not an empty folder: warn, use the local root."""
    (share / "runs").mkdir(parents=True)
    real_stat = os.stat

    def stat(path, *args, **kwargs):
        if str(path).startswith(str(share)):
            raise OSError(errno.EIO, "Input/output error", str(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(sim.manifest.os, "stat", stat)
    with caplog.at_level(logging.WARNING):
        assert run_roots() == [runs_root()]
    assert any(
        record.levelno == logging.WARNING and str(share / "runs") in record.getMessage() for record in caplog.records
    )


def test_local_root_missing(share):
    """No local runs/ folder (for example a new CFM_data_path): the share root only."""
    os.rmdir(runs_root())
    (share / "runs").mkdir(parents=True)
    assert run_roots() == [str(share / "runs")]


def test_no_root(local):
    os.rmdir(runs_root())
    assert run_roots() == []


def test_share_check_is_kept(share):
    """The share root is checked once in each process: a share runs/ folder made later is used after
    reset_share_check() only."""
    share.mkdir()
    assert run_roots() == [runs_root()]
    (share / "runs").mkdir()
    assert run_roots() == [runs_root()]
    reset_share_check()
    assert run_roots() == [runs_root(), str(share / "runs")]


def test_read_manifest_missing_is_none(tmp_path):
    assert sim.manifest.read_manifest(str(tmp_path)) is None
    (tmp_path / "file").write_text("")
    assert sim.manifest.read_manifest(str(tmp_path / "file")) is None  # NotADirectoryError


def test_read_manifest_oserror_is_raised(tmp_path, monkeypatch):
    """An I/O error of the soft share mount is not "no manifest" (os.path.exists would hide it)."""

    def failing(path, *args, **kwargs):
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr("builtins.open", failing)
    with pytest.raises(OSError):
        sim.manifest.read_manifest(str(tmp_path))
