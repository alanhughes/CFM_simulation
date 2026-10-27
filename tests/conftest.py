"""The config of the tests: config.example.yaml with its paths in a temporary folder, not config.yaml.

cfm_config still reads config.yaml from the current folder when it is imported, so the tests need one, but its values
are not used. For each test, the autouse fixture isolated_config replaces the contents of cfm_config.config (the same
dict object, as the modules import it with "from cfm_config import config") with the config of config.example.yaml:
cpom_mount and CFM_data_path are folders in a temporary folder that are not made, and CFM_share_data_path is not set.
So a test reads or writes real data, or the share, only if it sets the path itself. The original is restored after
the test.

The result of the check of the share root (sim.manifest.run_roots, FEAT-008) is kept for the process, so it is cleared
before and after each test: a test with a share that is not available does not change the next test.

A test that needs the real config.yaml (tests/test_reference_cfm.py) is marked with @pytest.mark.real_config.
"""

import copy
from pathlib import Path

import pytest
import yaml

import cfm_config
from sim.manifest import reset_share_check

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
    if request.node.get_closest_marker("real_config"):
        yield config
        reset_share_check()
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
