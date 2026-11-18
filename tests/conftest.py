"""The config of the tests: config.example.yaml with its paths in a temporary folder, not config.yaml.

cfm_config still reads config.yaml from the current folder when it is imported, so the tests need one, but its values
are not used. For each test, the autouse fixture isolated_config replaces the contents of cfm_config.config (the same
dict object, as the modules import it with "from cfm_config import config") with the config of config.example.yaml:
cpom_mount and CFM_data_path are folders in a temporary folder that are not made, and CFM_share_data_path is not set.
So a test reads or writes real data, or the share, only if it sets the path itself. The original is restored after
the test.

The result of the check of the share root (sim.manifest.run_roots) and of the share (cfm_config.share_available,
FEAT-008) is kept for the process, so it is cleared before and after each test: a test with a share that is not
available does not change the next test.

A test that needs the real config.yaml (tests/test_reference_cfm.py) is marked with @pytest.mark.real_config.
"""

import copy
import os
import sys
from pathlib import Path

import pytest
import yaml

import cfm_config
from cfm_config import config
import sim.manifest
import sim.run
from cfm_config import reset_share_available
from force.cache import forcing_csv_path
from force.model_point import ModelPoint
from sim.manifest import reset_share_check
from sim.run import CFMRun

EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "config.example.yaml"

with open(EXAMPLE_CONFIG, "r") as file:
    _example = yaml.safe_load(file)


def pytest_configure(config):
    config.addinivalue_line("markers", "real_config: use config.yaml, not the test config of tests/conftest.py")


@pytest.fixture(autouse=True)
def isolated_config(request, tmp_path_factory):
    """cfm_config.config is config.example.yaml for the test, with cpom_mount and CFM_data_path in a new temporary
    folder and no CFM_share_data_path. Yields config. Other path keys that use {cpom_mount} or {CFM_data_path} are in
    the same folder."""
    config = cfm_config.config
    reset_share_check()
    reset_share_available()
    if request.node.get_closest_marker("real_config"):
        yield config
        reset_share_check()
        reset_share_available()
        return
    folder = tmp_path_factory.mktemp("config")
    cfg = copy.deepcopy(_example)
    cfg["cpom_mount"] = str(folder / "CPOM")
    cfg["CFM_data_path"] = str(folder / "CFM_data")
    del cfg["CFM_share_data_path"]
    original = dict(config)
    config.clear()
    config.update(cfm_config.expand_cpom_mount(cfg))
    yield config
    config.clear()
    config.update(original)
    reset_share_check()
    reset_share_available()


# the fixtures and helpers of the run tests (tests/test_run_status.py, tests/test_timing.py)

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


@pytest.fixture
def drift_not_warned(monkeypatch):
    """Each test is a new process for the drift warning, which occurs one time in each process (FEAT-005)."""
    monkeypatch.setattr(sim.run, "_run_code_drift_warned", False)


def new_run(**kwargs) -> CFMRun:
    return CFMRun(*SITE, VERSION, **kwargs)


def folder(**kwargs) -> str:
    return new_run(**kwargs).prepare()


def overrides_file(tmp_path, text: str, name: str = "power_law") -> str:
    """A --cfm-overrides file name.yaml in tmp_path with the contents text."""
    path = tmp_path / f"{name}.yaml"
    path.write_text(text)
    return str(path)
