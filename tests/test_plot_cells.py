"""Tests of plot.cells on synthetic grids (FEAT-007). No RCM file and no network share is read."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import cfm_config
from force.grid import great_circle_km
from plot import cells, style

LAT0, LON0 = -67.5, -63.3
KM_PER_DEGREE = np.radians(1.0) * cells.EARTH_RADIUS_KM  # 111.19 km


def lattice(spacing_km: float, angle_deg: float, n: int = 9, m: int = 11):
    """Cell centres on a square lattice in local km around (LAT0, LON0), rotated by angle_deg, as lat2d and lon2d.
    Returns lat2d, lon2d and the lattice positions x, y (km)."""
    a = np.radians(angle_deg)
    ii, jj = np.meshgrid(np.arange(n) - n // 2, np.arange(m) - m // 2, indexing="ij")
    u, v = jj * spacing_km, ii * spacing_km  # column j along u, row i along v
    x, y = u * np.cos(a) - v * np.sin(a), u * np.sin(a) + v * np.cos(a)
    lat = LAT0 + np.degrees(y / cells.EARTH_RADIUS_KM)
    lon = LON0 + np.degrees(x / (cells.EARTH_RADIUS_KM * np.cos(np.radians(LAT0))))
    return lat, lon, x, y


# --- local_km ---------------------------------------------------------------------------------------------------------


def test_local_km_origin_and_axes():
    x, y = cells.local_km(LAT0, LON0, LAT0, LON0)
    assert (x, y) == (pytest.approx(0.0), pytest.approx(0.0))
    x, y = cells.local_km(LAT0 + 0.1, LON0, LAT0, LON0)
    assert x == pytest.approx(0.0) and y == pytest.approx(0.1 * KM_PER_DEGREE)
    x, y = cells.local_km(LAT0, LON0 + 0.1, LAT0, LON0)
    assert y == pytest.approx(0.0) and x == pytest.approx(0.1 * KM_PER_DEGREE * np.cos(np.radians(LAT0)))


def test_local_km_arrays_and_date_line():
    x, y = cells.local_km(np.array([[0.0, 0.0], [1.0, -1.0]]), np.array([[179.5, -179.5], [180.0, 180.0]]), 0.0, 180.0)
    assert x.shape == y.shape == (2, 2)
    np.testing.assert_allclose(x, np.array([[-0.5, 0.5], [0.0, 0.0]]) * KM_PER_DEGREE, atol=1e-9)


def test_local_km_distances_agree_with_great_circle():
    """At the scale of a map panel, distances in local km are within 0.5 % of the great circle distance."""
    lat = LAT0 + np.array([0.15, -0.12, 0.05, -0.2])
    lon = LON0 + np.array([0.3, -0.25, -0.4, 0.1])
    x, y = cells.local_km(lat, lon, LAT0, LON0)
    np.testing.assert_allclose(np.hypot(x, y), great_circle_km(LAT0, LON0, lat, lon), rtol=5e-3)


# --- cell corners -----------------------------------------------------------------------------------------------------


def test_regular_lat_lon_grid_gives_corners_half_way():
    lat1d, lon1d = np.arange(4) * 0.1 - 1.0, np.arange(5) * 0.1 + 10.0
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)
    clat, clon = cells.corner_grid(lat2d, lon2d)
    assert clat.shape == clon.shape == (5, 6)
    # interior corners half way between the centres, edge corners extrapolated by half a cell
    np.testing.assert_allclose(clat[:, 0], np.arange(5) * 0.1 - 1.05, atol=1e-4)
    np.testing.assert_allclose(clon[0, :], np.arange(6) * 0.1 + 9.95, atol=1e-4)


@pytest.mark.parametrize("angle", [0.0, 25.0, -60.0])
def test_rotated_lattice_gives_square_cells_of_the_spacing(angle):
    """A rotated lattice of 11 km: each cell is a square of side 11 km around its centre, rotated with the lattice."""
    lat2d, lon2d, x, y = lattice(11.0, angle)
    for i, j in [(4, 5), (0, 0), (8, 10), (0, 7)]:  # interior, corners and an edge of the array
        clat, clon = cells.cell_outline(lat2d, lon2d, i, j)
        cx, cy = cells.local_km(clat, clon, LAT0, LON0)
        sides = np.hypot(np.diff(np.append(cx, cx[0])), np.diff(np.append(cy, cy[0])))
        np.testing.assert_allclose(sides, 11.0, rtol=2e-3)
        np.testing.assert_allclose([cx.mean(), cy.mean()], [x[i, j], y[i, j]], atol=0.05)
        # the first side, corner [i, j] to [i, j+1], is along the rows (direction of j) of the lattice
        a = np.radians(angle)
        np.testing.assert_allclose([cx[1] - cx[0], cy[1] - cy[0]], [11 * np.cos(a), 11 * np.sin(a)], atol=0.05)


def test_cell_outline_matches_corner_grid_of_the_whole_grid():
    lat2d, lon2d, _, _ = lattice(27.5, 40.0, n=6, m=7)
    clat, clon = cells.corner_grid(lat2d, lon2d)
    for i, j in [(0, 0), (2, 3), (5, 6), (5, 0)]:
        lat4, lon4 = cells.cell_outline(lat2d, lon2d, i, j)
        np.testing.assert_allclose(lat4, clat[[i, i, i + 1, i + 1], [j, j + 1, j + 1, j]], atol=1e-9)
        np.testing.assert_allclose(lon4, clon[[i, i, i + 1, i + 1], [j, j + 1, j + 1, j]], atol=1e-9)


def test_corner_grid_needs_two_by_two():
    with pytest.raises(ValueError):
        cells.corner_grid(np.zeros((1, 3)), np.zeros((1, 3)))


def test_cells_near_and_cell_size():
    lat2d, lon2d, x, y = lattice(10.0, 15.0)
    near = cells.cells_near(lat2d, lon2d, LAT0, LON0, 10.5)
    assert near == [(3, 5), (4, 4), (4, 5), (4, 6), (5, 5)]  # the centre cell and its four neighbours
    assert cells.cell_size_km(lat2d, lon2d, 4, 5) == pytest.approx(10.0, rel=2e-3)
    # ~60 km from the centre of the lattice, which is regular in local km, the great circle spacing differs by ~1 %
    assert cells.cell_size_km(lat2d, lon2d, 0, 0) == pytest.approx(10.0, rel=2e-2)


def test_format_km():
    assert cells.format_km(6.697) == "6.70 km"
    assert cells.format_km(15.989) == "16.0 km"
    assert cells.format_km(11.222) == "11.2 km"


# --- drawing (smoke tests) --------------------------------------------------------------------------------------------


def test_drawing_functions_run():
    lat2d, lon2d, x, y = lattice(11.0, 20.0)
    fig, ax = plt.subplots()
    centres = cells.draw_cells(ax, lat2d, lon2d, (LAT0, LON0), 30.0, {(4, 5): style.CURRENT, (4, 6): style.ALTERNATIVE})
    assert set(centres) == {(4, 5), (4, 6)}
    assert centres[(4, 6)] == (pytest.approx(x[4, 6], abs=0.05), pytest.approx(y[4, 6], abs=0.05))
    assert len(ax.collections) == 1 and len(ax.collections[0].get_segments()) > 10
    assert [line.get_linestyle() for line in ax.lines] == ["-", "--"]

    other_lat, other_lon, _, _ = lattice(27.5, -10.0, n=4, m=4)
    mx, my = cells.draw_outline(ax, other_lat, other_lon, 2, 2, (LAT0, LON0))
    cells.draw_centre(ax, mx, my, marker=cells.OTHER_CENTRE_MARKER, label="MAR 2, 2")
    cells.draw_site(ax, 1.0, 2.0, "Site A")
    cells.draw_centre(ax, *centres[(4, 5)], label="4, 5\n1.0 km")
    cells.distance_label(ax, (1.0, 2.0), (mx, my), "20.0 km")
    cells.map_axes(ax, 22.5, view_offset_km=(0, -5))
    cells.cells_legend(fig, cells.legend_entries(other_rcm="MAR"), loc="outside lower center", ncol=4)
    assert ax.get_ylim() == (-27.5, 17.5)
    assert len(fig.legends[0].get_texts()) == 7
    fig.canvas.draw()
    plt.close(fig)


def test_site_map_and_overlay_on_a_synthetic_grid(monkeypatch):
    """site_map and overlay_cell with the grid and the sites replaced, so no config path or RCM file is used."""
    grids = {"RCM": lattice(11.0, 20.0)[:2], "OTHER": lattice(27.5, -10.0, n=4, m=4)[:2]}
    monkeypatch.setattr(cells, "rcm_grid", lambda rcm_name, data_path=None: grids[rcm_name])
    sites = {"A": cfm_config.Site("A", LAT0, LON0), "B": cfm_config.Site("B", LAT0 - 0.05, LON0 + 0.1)}
    monkeypatch.setattr(cfm_config, "get_site", lambda name: sites[name])

    fig, (ax1, ax2) = plt.subplots(1, 2)
    frame = cells.site_map(ax1, "RCM", ["A", "B"], {(4, 5): style.CURRENT, (3, 5): style.ALTERNATIVE})
    assert frame.sites["A"] == (pytest.approx(0.0), pytest.approx(0.0))
    texts = [t.get_text() for t in ax1.texts]
    assert "4, 5\n0.00 km to A" in texts and "B" in texts
    cells.overlay_cell(ax1, frame, "OTHER", (2, 2), distances_to=["A", "B"], distance_text="line")
    cells.overlay_cell(ax1, frame, "OTHER", (1, 1), distances_to=["A"], distance_text="centre", label="other")

    frame2 = cells.site_map(ax2, "RCM", ["A"], {(4, 5): style.CURRENT}, distance_text="line", view_offset_km=(3, 0))
    assert "0.00 km" in [t.get_text() for t in ax2.texts]
    assert ax2.get_xlim() == (-19.5, 25.5) and frame2.centres
    fig.canvas.draw()
    plt.close(fig)
