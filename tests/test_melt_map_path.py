"""Tests of the path of the melt map for a read: local first, then the share (FEAT-008). No real data is read."""

import errno
import logging
import os

from cfm_config import config
from plot.melt import melt_map_path


def _paths(tmp_path, monkeypatch):
    monkeypatch.setitem(config, "CFM_data_path", str(tmp_path / "local"))
    monkeypatch.setitem(config, "CFM_share_data_path", str(tmp_path / "share"))
    name = f"racmo_avg_yearly_melt_map_{config['start_year']}-{config['end_year']}.nc"
    return str(tmp_path / "local" / "melt" / name), str(tmp_path / "share" / "melt" / name)


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


def test_melt_map_local(tmp_path, monkeypatch):
    local, share = _paths(tmp_path, monkeypatch)
    _touch(local)
    _touch(share)
    assert melt_map_path("racmo") == local


def test_melt_map_share_only(tmp_path, monkeypatch):
    local, share = _paths(tmp_path, monkeypatch)
    _touch(share)
    assert melt_map_path("racmo") == share


def test_melt_map_neither(tmp_path, monkeypatch):
    local, share = _paths(tmp_path, monkeypatch)
    os.makedirs(os.path.dirname(share))
    assert melt_map_path("racmo") == local


def test_melt_map_share_not_set(tmp_path, monkeypatch):
    local, share = _paths(tmp_path, monkeypatch)
    _touch(share)
    monkeypatch.delitem(config, "CFM_share_data_path")
    assert melt_map_path("racmo") == local


def test_melt_map_share_not_available(tmp_path, monkeypatch, caplog):
    local, share = _paths(tmp_path, monkeypatch)
    _touch(share)
    share_root = str(tmp_path / "share")
    real_stat, calls = os.stat, []

    def stat(path, *args, **kwargs):
        if str(path).startswith(share_root):
            calls.append(str(path))
            raise OSError(errno.EIO, "Input/output error")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat)
    with caplog.at_level(logging.WARNING):
        assert melt_map_path("racmo") == local
        assert melt_map_path("racmo") == local
    assert len([r for r in caplog.records if "not available" in r.message]) == 1
    assert len(calls) == 1
