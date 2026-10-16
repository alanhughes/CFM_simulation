"""Tests for the coordinate format in cfm_config.py (BUG-001). Run from the repo root."""

import pytest

from cfm_config import format_coord, sites


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


def test_borehole_sites_have_unique_coordinates():
    table = sites(active_only=False)
    assert {"CI-120", "J108", "SI47"} <= table.keys()  # active and inactive sites were both found
    coords = {name: (format_coord(site.lat), format_coord(site.lon)) for name, site in table.items()}
    assert len(set(coords.values())) == len(coords), coords
