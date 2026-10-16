"""Tests for the CFM output names built in cfm_config.py (BUG-001). Run from the repo root."""

import pytest

from cfm_config import cfm_output_folder, config, format_coord, sites


@pytest.mark.parametrize(
    "value, expected",
    [
        (-67.000, "-67.000"),  # sites.yaml (CI-120)
        (-67.500, "-67.500"),  # sites.yaml (WI-70)
        (-64.149, "-64.149"),  # sites.yaml (J208)
        (-64.14850, "-64.149"),  # run_cfm_grid.sh (J208), :.3f gives -64.148
        (-67.10815, "-67.108"),  # run_cfm_grid.sh (J108)
        (-68.0086, "-68.009"),  # run_cfm.sh (JP21)
    ],
)
def test_format_coord(value, expected):
    assert format_coord(value) == expected


def test_cfm_output_folder_ci120(monkeypatch):
    monkeypatch.setitem(config, "CFM_data_path", "/data")
    monkeypatch.setitem(config, "start_year", 1979)
    monkeypatch.setitem(config, "end_year", 2024)
    assert (
        cfm_output_folder(-67.000, -61.486, "GSFC2020", "bucket", "RACMO")
        == "/data/cfm_output/CFMoutput_-67.000_-61.486_1979_2024_GSFC2020_bucket_RACMO"
    )


def test_borehole_sites_have_unique_names():
    table = sites(active_only=False)
    assert {"CI-120", "J108", "SI47"} <= table.keys()  # active and inactive sites were both found
    folders = {
        name: cfm_output_folder(site.lat, site.lon, "GSFC2020", "bucket", "RACMO") for name, site in table.items()
    }
    assert len(set(folders.values())) == len(folders), folders
