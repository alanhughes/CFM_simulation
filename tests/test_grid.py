"""Tests for the nearest grid cell search (BUG-002)."""

import numpy as np
import pytest

from force.grid import great_circle_km, nearest_cell


def test_great_circle_one_degree_of_latitude():
    assert great_circle_km(-67.0, -63.0, -68.0, -63.0) == pytest.approx(111.19, abs=0.01)


def test_great_circle_longitude_shrinks_with_latitude():
    at_67s = great_circle_km(-67.0, -63.0, -67.0, -62.0)
    assert at_67s == pytest.approx(111.19 * np.cos(np.radians(67.0)), rel=1e-3)


def test_nearest_cell_uses_real_distance():
    # Two cells near 67 S. Cell 0 is 0.20 deg of longitude east of the site (8.7 km). Cell 1 is 0.12 deg of latitude
    # north of the site (13.3 km). The old search in degrees selects cell 1.
    site_lat, site_lon = -67.0, -63.0
    lat2d = np.array([[site_lat, site_lat + 0.12]])
    lon2d = np.array([[site_lon + 0.20, site_lon]])

    i, j, distance = nearest_cell(lat2d, lon2d, site_lat, site_lon)
    assert (i, j) == (0, 0)
    assert distance == pytest.approx(8.69, abs=0.01)

    old = np.sqrt((lat2d - site_lat) ** 2 + (lon2d - site_lon) ** 2)
    assert np.unravel_index(old.argmin(), old.shape) == (0, 1)


def test_nearest_cell_on_rotated_grid():
    # Curvilinear grid (like RACMO rlat/rlon): rows and columns are not lines of latitude and longitude.
    ii, jj = np.meshgrid(np.arange(20), np.arange(30), indexing="ij")
    lat2d = -70.0 + 0.05 * ii + 0.02 * jj
    lon2d = -65.0 + 0.1 * jj - 0.03 * ii
    i, j, distance = nearest_cell(lat2d, lon2d, lat2d[7, 11], lon2d[7, 11])
    assert (i, j) == (7, 11)
    assert distance == pytest.approx(0.0, abs=1e-9)
