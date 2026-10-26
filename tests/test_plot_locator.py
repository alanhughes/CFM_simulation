"""Tests of plot.locator on a synthetic BedMachine file (FEAT-007). No file on the network share is read."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import cfm_config
from plot import cells, locator

LAT0, LON0 = -67.5, -63.3
EXTENT = (-60.0, 60.0, 2420.0, 2540.0)  # map km around (LAT0, LON0)


@pytest.fixture
def bedmachine(tmp_path):
    """A BedMachine-like file on a 2 km EPSG:3031 grid that covers EXTENT: x increasing, y decreasing, mask int8.
    The west half is grounded ice with one point of Lake Vostok (4), the east half floating ice, and the east edge
    ocean."""
    from pyproj import Transformer

    to_3031 = Transformer.from_crs(locator.MAP_CRS, "EPSG:3031", always_xy=True)
    bx, by = to_3031.transform(np.array([-100.0, 100.0, 100.0, -100.0]), np.array([2380.0, 2380.0, 2580.0, 2580.0]))
    x = np.arange(np.floor(bx.min() / 2e3) * 2e3, bx.max(), 2e3)
    y = np.arange(np.ceil(by.max() / 2e3) * 2e3, by.min(), -2e3)
    mx, my = locator._BEDMACHINE_TO_MAP.transform(*np.meshgrid(x, y))
    mask = np.where(mx < 0, locator.GROUNDED, locator.FLOATING).astype(np.int8)
    mask[mx > 40] = locator.OCEAN
    mask[0, 0] = 4
    path = tmp_path / "bedmachine.nc"
    xr.Dataset({"mask": (("y", "x"), mask)}, coords={"x": x, "y": y}).to_netcdf(path)
    return str(path)


def test_map_km_is_north_up_on_the_central_meridian():
    x, y = locator.map_km(np.array([-66.0, -68.0]), locator.CENTRAL_LON)
    assert x == pytest.approx([0.0, 0.0], abs=1e-6) and y[0] > y[1]
    # one degree of latitude is about 111.6 km; at 67 S the scale of the projection (true at 71 S) is ~1 % larger
    assert y[0] - y[1] == pytest.approx(2 * 111.6, rel=0.03)
    x_east, _ = locator.map_km(-67.0, locator.CENTRAL_LON + 1.0)
    assert x_east > 0


def test_read_mask_covers_the_extent(bedmachine):
    x, y, mask = locator.read_mask(EXTENT, path=bedmachine)
    assert x.shape == y.shape == mask.shape
    assert x.min() <= EXTENT[0] and x.max() >= EXTENT[1] and y.min() <= EXTENT[2] and y.max() >= EXTENT[3]
    assert set(np.unique(mask)) <= {locator.OCEAN, locator.GROUNDED, locator.FLOATING}  # Lake Vostok is grounded
    assert np.all(mask[(x < -5) & (np.abs(y - 2480) < 50)] == locator.GROUNDED)
    assert np.all(mask[(x > 45) & (np.abs(y - 2480) < 50)] == locator.OCEAN)

    xs, ys, strided = locator.read_mask(EXTENT, stride=3, path=bedmachine)
    assert np.array_equal(strided, mask[::3, ::3]) and np.allclose(xs, x[::3, ::3])


def test_view_box_outlines_the_panel():
    fig, (ax, panel) = plt.subplots(1, 2)
    panel.set_xlim(-22.5, 22.5)
    panel.set_ylim(-26.5, 18.5)
    locator.view_box(ax, panel, cells.MapFrame(LAT0, LON0), "(b)")
    x, y = ax.lines[0].get_data()
    cx, cy = locator.map_km(LAT0, LON0)
    assert x.max() - x.min() == pytest.approx(45.0, rel=0.03) and y.max() - y.min() == pytest.approx(45.0, rel=0.03)
    assert np.mean(x[:4]) == pytest.approx(cx, abs=1.0) and np.mean(y[:4]) == pytest.approx(cy - 4.0, abs=1.0)
    assert [t.get_text() for t in ax.texts] == ["(b)"]
    plt.close(fig)


def test_locator_map_and_inset_run(bedmachine, monkeypatch):
    monkeypatch.setattr(cfm_config, "get_site", lambda name: cfm_config.Site(name, LAT0, LON0))
    fig, ax = plt.subplots()
    positions = locator.locator_map(ax, EXTENT, ["A"], lats=[-67, -68], lons=[-64, -62], stride=1, path=bedmachine)
    assert positions["A"] == pytest.approx(tuple(float(v) for v in locator.map_km(LAT0, LON0)))
    assert ax.get_xlim() == EXTENT[:2] and ax.get_ylim() == EXTENT[2:]
    assert {"67°S", "64°W", "62°W"} <= {t.get_text() for t in ax.texts}
    locator.place_name(ax, LAT0, LON0, "Larsen C")
    locator.antarctica_inset(ax, EXTENT, stride=1, path=bedmachine)
    ax.legend(handles=locator.surface_legend_entries())
    fig.canvas.draw()
    plt.close(fig)


def test_panel_surface_draws_in_local_km_and_keeps_the_limits(bedmachine):
    fig, panel = plt.subplots()
    cells.map_axes(panel, 22.5, view_offset_km=(0, -4))
    locator.panel_surface(panel, cells.MapFrame(LAT0, LON0), path=bedmachine)
    assert panel.get_xlim() == (-22.5, 22.5) and panel.get_ylim() == (-26.5, 18.5)
    mesh = panel.collections[0]
    assert mesh.get_array().size > 45 * 45 / 4  # the 2 km test grid covers the 45 km panel
    fig.canvas.draw()
    plt.close(fig)
