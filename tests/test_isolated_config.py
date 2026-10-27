"""Tests for the test config of tests/conftest.py: with no setup in the test, config points at no real data.

config.yaml is read here only to compare its paths with those of the test config; nothing is read from the paths.
"""

from pathlib import Path

import pytest
import yaml

from cfm_config import config, expand_cpom_mount, share_data_path
from sim.manifest import run_roots, share_runs_root


def _real_config():
    with open("config.yaml", "r") as file:
        return expand_cpom_mount(yaml.safe_load(file))


def test_no_share():
    assert "CFM_share_data_path" not in config
    assert share_data_path() is None
    assert share_runs_root() is None


def test_paths_are_not_real(tmp_path_factory):
    real = _real_config()
    assert config["CFM_data_path"] != real["CFM_data_path"]
    assert config["cpom_mount"] != real["cpom_mount"]
    # the folders are in the temporary folder of pytest, and are not made
    for key in ("cpom_mount", "CFM_data_path", "in_situ_data_path", "RACMO23_data_path", "BedMachine_data_path"):
        assert Path(config[key]).is_relative_to(tmp_path_factory.getbasetemp()), key
        assert not Path(config[key]).exists(), key
    assert run_roots() == []


def test_settings_of_the_example():
    """The settings that are not paths are those of config.example.yaml."""
    assert (config["start_year"], config["end_year"]) == (1979, 2024)
    assert config["cfm_config"]["physRho"] == "GSFC2020"
    assert "RACMO23_to_CFM_column_map" in config


def test_change_is_not_kept_1():
    config["cfm_config"]["physRho"] = "changed"
    config["end_year"] = 1980


def test_change_is_not_kept_2():
    """A test that changes config without monkeypatch does not change the config of the next test."""
    assert config["cfm_config"]["physRho"] == "GSFC2020"
    assert config["end_year"] == 2024


@pytest.mark.real_config
def test_real_config_marker():
    """A test with the real_config marker has the config of config.yaml."""
    assert config["CFM_data_path"] == _real_config()["CFM_data_path"]
