"""Locator map of the Antarctic Peninsula and Larsen C, with the sites and the areas of the cell maps (FEAT-007,
Figure 1).

The map is polar stereographic (true scale at 71 S, as EPSG:3031) with the central meridian at CENTRAL_LON, so north
is up at the sites. Positions are in km of that projection (map_km). The surface types are the BedMachine Antarctica v2
mask (Morlighem et al., 2020; 500 m posting, EPSG:3031): ocean, ice-free land, grounded ice and floating ice. The file is
read from BedMachine_data_path in config.yaml when a function is called, never at import time.

A cell map of plot.cells (site_map) is in local km around a site. view_box draws the area that such a panel shows, so
the locator shows where each panel is.
"""

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from pyproj import Transformer

from plot import cells, style

CENTRAL_LON = -63.0
MAP_CRS = f"+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0={CENTRAL_LON} +datum=WGS84 +units=km"

# BedMachine mask values and their colours; Lake Vostok (4) is drawn as grounded ice
OCEAN, ROCK, GROUNDED, FLOATING = 0, 1, 2, 3
SURFACE_COLOURS = {OCEAN: "#c3d5e6", ROCK: "#a9a8a3", GROUNDED: "#fbfbfa", FLOATING: "#e4ecf3"}
SURFACE_LABELS = {OCEAN: "Ocean", ROCK: "Ice-free land", GROUNDED: "Grounded ice", FLOATING: "Floating ice"}
COAST_LINE = {"colors": style.TEXT_SECONDARY, "linewidths": 0.5}  # ice front and coast: ocean against anything else
GROUNDING_LINE = {"colors": "#8a8984", "linewidths": 0.4}  # grounded against floating ice
GRATICULE = {"color": "#8a8984", "linewidth": 0.4, "linestyle": (0, (2, 2))}
VIEW_LINE = {"color": style.TEXT, "linewidth": 1.0}  # the area of a cell map panel
SITE_MARKER = {**cells.SITE_MARKER, "markersize": 4}

_TO_MAP = Transformer.from_crs("EPSG:4326", MAP_CRS, always_xy=True)
_BEDMACHINE_TO_MAP = Transformer.from_crs("EPSG:3031", MAP_CRS, always_xy=True)
_MAP_TO_BEDMACHINE = Transformer.from_crs(MAP_CRS, "EPSG:3031", always_xy=True)
_FROM_MAP = Transformer.from_crs(MAP_CRS, "EPSG:4326", always_xy=True)


def map_km(lat, lon) -> Tuple[np.ndarray, np.ndarray]:
    """(x, y) in km of the locator projection (MAP_CRS). Works on scalars and arrays, broadcast against each other."""
    lat, lon = np.broadcast_arrays(np.asarray(lat, dtype=float), np.asarray(lon, dtype=float))
    x, y = _TO_MAP.transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def bedmachine_path() -> str:
    from cfm_config import config

    return config["BedMachine_data_path"]


def read_mask(extent_km: Tuple[float, float, float, float], stride: int = 1, path: Optional[str] = None):
    """The BedMachine mask that covers extent_km (x0, x1, y0, y1 in map km), every stride-th point (stride 1 is 500 m).

    Returns x, y (2D, map km, of each mask point) and the mask (int8). The mask is read on its own EPSG:3031 grid,
    so x and y are the rotated grid, for pcolormesh.
    """
    import xarray as xr

    x0, x1, y0, y1 = extent_km
    edge = np.linspace(0.0, 1.0, 21)
    xs = np.concatenate([x0 + (x1 - x0) * edge, np.full(21, x1), x1 - (x1 - x0) * edge, np.full(21, x0)])
    ys = np.concatenate([np.full(21, y0), y0 + (y1 - y0) * edge, np.full(21, y1), y1 - (y1 - y0) * edge])
    bx, by = _MAP_TO_BEDMACHINE.transform(xs, ys)
    with xr.open_dataset(path or bedmachine_path()) as ds:
        # y decreases along the BedMachine grid. The file is chunked and compressed, so a strided read is far slower
        # than reading the whole block and taking every stride-th point after
        block = ds["mask"].sel(x=slice(bx.min(), bx.max()), y=slice(by.max(), by.min()))
        mask = block.values[::stride, ::stride]
        bx2, by2 = np.meshgrid(block["x"].values[::stride], block["y"].values[::stride])
    x, y = _BEDMACHINE_TO_MAP.transform(bx2.astype(float), by2.astype(float))
    return x, y, np.where(mask == 4, GROUNDED, mask)


def draw_surface(ax: plt.Axes, x, y, mask, lines: bool = True) -> None:
    """The surface types of the mask (read_mask) in SURFACE_COLOURS, and with lines, the coast and ice fronts and the
    grounding line."""
    colours = ListedColormap([SURFACE_COLOURS[k] for k in (OCEAN, ROCK, GROUNDED, FLOATING)])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], colours.N)
    ax.pcolormesh(x, y, mask, cmap=colours, norm=norm, shading="nearest", rasterized=True, zorder=0)
    if lines:
        ax.contour(x, y, (mask != OCEAN).astype(float), levels=[0.5], zorder=1, **COAST_LINE)
        grounded = np.where(mask == OCEAN, np.nan, (mask != FLOATING).astype(float))
        ax.contour(x, y, grounded, levels=[0.5], zorder=1, **GROUNDING_LINE)


def surface_legend_entries() -> list:
    """Legend handles of the surface types, for ax.legend."""
    return [
        Patch(facecolor=SURFACE_COLOURS[k], edgecolor="#8a8984", linewidth=0.4, label=SURFACE_LABELS[k])
        for k in (FLOATING, GROUNDED, ROCK, OCEAN)
    ]


def draw_graticule(
    ax: plt.Axes,
    lats: Sequence[float],
    lons: Sequence[float],
    extent_km: Tuple[float, float, float, float],
) -> None:
    """Parallels and meridians as thin dashed lines, with the latitude labelled at the left edge of extent_km and the
    longitude at the bottom edge."""
    x0, x1, y0, y1 = extent_km
    t = np.linspace(-90.0, 0.0, 721)
    for lat in lats:
        lon = np.linspace(-180.0, 180.0, 1441)
        x, y = map_km(np.full_like(lon, lat), lon)
        ax.plot(x, y, zorder=2, **GRATICULE)
        _edge_label(ax, x, y, x0, "x", f"{abs(lat):g}°S", (0.0, 0.0), extent_km)
    for lon in lons:
        x, y = map_km(t, np.full_like(t, lon))
        ax.plot(x, y, zorder=2, **GRATICULE)
        _edge_label(ax, x, y, y0, "y", f"{abs(lon):g}°W", (0.0, 0.0), extent_km)


def _edge_label(ax, x, y, edge, along, text, offset, extent_km) -> None:
    # the point where the line crosses the edge (x == edge for along "x", y == edge for along "y"), inside the extent
    x0, x1, y0, y1 = extent_km
    a = x if along == "x" else y
    b = y if along == "x" else x
    lo, hi = (y0, y1) if along == "x" else (x0, x1)
    crossings = np.nonzero(np.diff(np.sign(a - edge)) != 0)[0]
    for k in crossings:
        f = (edge - a[k]) / (a[k + 1] - a[k])
        p = b[k] + f * (b[k + 1] - b[k])
        if lo <= p <= hi:
            xy = (edge, p) if along == "x" else (p, edge)
            ax.annotate(
                text,
                xy,
                xytext=(-3, 0) if along == "x" else (0, -3),
                textcoords="offset points",
                ha="right" if along == "x" else "center",
                va="center" if along == "x" else "top",
                fontsize=7,
                color=style.TEXT_SECONDARY,
                annotation_clip=False,
            )
            return


def draw_sites(ax: plt.Axes, site_names: Sequence[str]) -> dict:
    """The sites as small black triangles. Returns their positions in map km, {name: (x, y)}."""
    from cfm_config import get_site

    positions = {}
    for name in site_names:
        site = get_site(name)
        x, y = map_km(site.lat, site.lon)
        ax.plot(x, y, **SITE_MARKER)
        positions[name] = (float(x), float(y))
    return positions


def place_name(ax: plt.Axes, lat: float, lon: float, text: str, rotation: float = 0.0, **kwargs) -> None:
    """A place name (for example "Larsen C\nIce Shelf") in italic secondary text, centred on (lat, lon)."""
    x, y = map_km(lat, lon)
    kw = {"ha": "center", "va": "center", "fontsize": 7, "style": "italic", "color": style.TEXT_SECONDARY}
    ax.text(x, y, text, rotation=rotation, zorder=4, linespacing=1.1, **{**kw, **kwargs})


def _panel_outline(panel: plt.Axes, frame: "cells.MapFrame") -> Tuple[np.ndarray, np.ndarray]:
    # the corners of a cell map panel (its axis limits, in local km around frame.lat0, frame.lon0), closed, in map km
    (px0, px1), (py0, py1) = panel.get_xlim(), panel.get_ylim()
    px = np.array([px0, px1, px1, px0, px0])
    py = np.array([py0, py0, py1, py1, py0])
    lat = frame.lat0 + np.degrees(py / cells.EARTH_RADIUS_KM)
    lon = frame.lon0 + np.degrees(px / (cells.EARTH_RADIUS_KM * np.cos(np.radians(frame.lat0))))
    return map_km(lat, lon)


def panel_surface(panel: plt.Axes, frame: "cells.MapFrame", margin_km: float = 2.0, path: Optional[str] = None) -> None:
    """The surface types, coast and grounding line under a cell map panel of plot.cells (site_map), in its local km.
    Call it after site_map, which sets the axis limits. The mask is read at its full 500 m posting."""
    x, y = _panel_outline(panel, frame)
    extent = (x.min() - margin_km, x.max() + margin_km, y.min() - margin_km, y.max() + margin_km)
    mx, my, mask = read_mask(extent, path=path)
    lon, lat = _FROM_MAP.transform(mx, my)
    lx, ly = cells.local_km(lat, lon, frame.lat0, frame.lon0)
    limits = panel.get_xlim(), panel.get_ylim()
    draw_surface(panel, lx, ly, mask)
    panel.set_xlim(limits[0])
    panel.set_ylim(limits[1])


def view_box(
    ax: plt.Axes,
    panel: plt.Axes,
    frame: "cells.MapFrame",
    label: Optional[str] = None,
    offset: Tuple[float, float] = (4, 0),
    ha: str = "left",
    va: str = "center",
) -> None:
    """The area that a cell map panel shows (its axis limits, in local km around frame.lat0, frame.lon0) as a black
    outline, with an optional label such as "(b)" next to it."""
    x, y = _panel_outline(panel, frame)
    ax.plot(x, y, zorder=5, **VIEW_LINE)
    if label:
        right = {"left": x.max(), "right": x.min(), "center": x.mean()}[ha]
        ax.annotate(
            label,
            (right, y.mean()),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=8,
            fontweight="bold",
            color=style.TEXT,
            bbox=cells.LABEL_BOX,
            zorder=7,
        )


def locator_map(
    ax: plt.Axes,
    extent_km: Tuple[float, float, float, float],
    site_names: Sequence[str] = (),
    lats: Sequence[float] = (),
    lons: Sequence[float] = (),
    stride: int = 2,
    path: Optional[str] = None,
) -> dict:
    """The locator panel: the surface types in extent_km (x0, x1, y0, y1, map km), the graticule and the sites. The
    mask is read every stride-th point (2 is 1 km, plenty for a panel of some hundred km). The axes have no ticks: the
    graticule gives the position. Returns the site positions in map km."""
    x, y, mask = read_mask(extent_km, stride=stride, path=path)
    draw_surface(ax, x, y, mask)
    draw_graticule(ax, lats, lons, extent_km)
    positions = draw_sites(ax, site_names)
    x0, x1, y0, y1 = extent_km
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    return positions


def antarctica_inset(
    ax: plt.Axes,
    extent_km: Tuple[float, float, float, float],
    bounds: Sequence[float] = (0.0, 0.0, 0.3, 0.3),
    stride: int = 40,
    path: Optional[str] = None,
) -> plt.Axes:
    """A small map of Antarctica in the corner of ax (bounds in axes fraction), with extent_km of ax as a box."""
    inset = ax.inset_axes(bounds)
    whole = (-3400.0, 3400.0, -3400.0, 3400.0)
    x, y, mask = read_mask(whole, stride=stride, path=path)
    draw_surface(inset, x, y, np.where(mask == ROCK, GROUNDED, mask), lines=False)
    x0, x1, y0, y1 = extent_km
    inset.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], zorder=3, **VIEW_LINE)
    land = np.nonzero(mask != OCEAN)
    pad = 150.0
    inset.set_xlim(x[land].min() - pad, x[land].max() + pad)
    inset.set_ylim(y[land].min() - pad, y[land].max() + pad)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.grid(False)
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    return inset
