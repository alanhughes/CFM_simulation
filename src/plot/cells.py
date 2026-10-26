"""Map of the sites and the RCM grid cells around them, in local km (FEAT-007, Figure 1).

The functions draw on a matplotlib Axes, so a figure can combine them. Positions are in km east (x) and north (y) of a
centre point, by an equirectangular projection (local_km). That is accurate to well under 1 % over a panel of ~50 km.

The RCM grids are curvilinear (RACMO is a rotated pole grid, MAR is polar stereographic), so a cell is not a lat/lon
box. Each corner of a cell is the mean of the four cell centres around it (corner_grid), and a cell outline joins its
four corners.

Index convention (as force.model_point): cell (i, j) is lat2d[i, j], lon2d[i, j]; i is the row (rlat / Y) and j the
column (rlon / X). The model point tag RACMO_261_126 is i=261, j=126.

Line style follows plot.style: a solid outline is the current (corrected) cell, a dashed outline the alternative (old)
cell. Outlines and text are black or grey, never a scheme colour. Nothing here reads config.yaml at import time:
site_map and overlay_cell read the RCM grid and sites.yaml when they are called.
"""

import importlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from plot import style

EARTH_RADIUS_KM = 6371.0  # as force.grid.EARTH_RADIUS_KM

Cell = Tuple[int, int]
Point = Tuple[float, float]

GRID_LINE = {"color": "#c9c8c3", "linewidth": 0.5}  # the other cells of the grid
CELL_LINE = {"color": "#2b2a28", "linewidth": 1.4}  # a highlighted cell; the line style says which side of a pair
OTHER_GRID_LINE = {"color": style.TEXT_SECONDARY, "linewidth": 1.0, "linestyle": "-."}  # a cell of a second RCM
SITE_MARKER = {"marker": "^", "markersize": 6, "color": "black", "linestyle": "none", "zorder": 6}
CENTRE_MARKER = {"marker": "+", "markersize": 7, "markeredgewidth": 1.3, "color": "#2b2a28", "linestyle": "none"}
OTHER_CENTRE_MARKER = {"marker": "x", "markersize": 7, "markeredgewidth": 1.3, "color": "#2b2a28", "linestyle": "none"}
CONNECTOR = {"color": "#8a8984", "linewidth": 0.6, "linestyle": "-"}
LABEL_BOX = {"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.85}


# --- geometry ---------------------------------------------------------------------------------------------------------


def _great_circle_km(lat1, lon1, lat2, lon2):
    # force.grid is imported here: importing the force package reads config.yaml (force/__init__ -> force.process)
    from force.grid import great_circle_km

    return great_circle_km(lat1, lon1, lat2, lon2)


def local_km(lat, lon, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    """(x, y) in km east and north of (lat0, lon0), by an equirectangular projection. Works on scalars and arrays.

    The longitude difference is wrapped to [-180, 180), so a point across the date line stays near the centre.
    """
    lat = np.asarray(lat, dtype=float)
    dlon = (np.asarray(lon, dtype=float) - lon0 + 180.0) % 360.0 - 180.0
    x = EARTH_RADIUS_KM * np.radians(dlon) * np.cos(np.radians(lat0))
    y = EARTH_RADIUS_KM * np.radians(lat - lat0)
    return x, y


def _xyz(lat, lon) -> np.ndarray:
    lat, lon = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def _latlon(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    return np.degrees(np.arctan2(z, np.hypot(x, y))), np.degrees(np.arctan2(y, x))


def corner_grid(lat2d, lon2d) -> Tuple[np.ndarray, np.ndarray]:
    """Latitude and longitude of the cell corners of a 2D grid of cell centres, each of shape (n + 1, m + 1).

    Corner [k, l] is the mean of the (up to) four centres [k-1:k+1, l-1:l+1], taken as unit vectors on the sphere, so
    the result does not depend on the longitude convention. At the edges of the array, the centres are first extended
    by one row or column by linear extrapolation. Cell (i, j) has the corners [i, j], [i, j+1], [i+1, j+1], [i+1, j].
    The grid must be at least 2 x 2.
    """
    p = _xyz(np.asarray(lat2d, dtype=float), np.asarray(lon2d, dtype=float))
    if p.shape[0] < 2 or p.shape[1] < 2:
        raise ValueError(f"corner_grid needs at least 2 x 2 cell centres, got {p.shape[:2]}")
    p = np.concatenate([2 * p[:1] - p[1:2], p, 2 * p[-1:] - p[-2:-1]], axis=0)
    p = np.concatenate([2 * p[:, :1] - p[:, 1:2], p, 2 * p[:, -1:] - p[:, -2:-1]], axis=1)
    return _latlon(p[:-1, :-1] + p[1:, :-1] + p[:-1, 1:] + p[1:, 1:])


def _window(shape: Tuple[int, int], i0: int, i1: int, j0: int, j1: int) -> Tuple[slice, slice]:
    """Rows i0..i1 and columns j0..j1 with one neighbour on each side (inside the grid), at least 2 x 2."""
    n, m = shape
    rows = slice(max(min(i0 - 1, n - 2), 0), min(max(i1 + 2, 2), n))
    cols = slice(max(min(j0 - 1, m - 2), 0), min(max(j1 + 2, 2), m))
    return rows, cols


def cell_outline(lat2d, lon2d, i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
    """Latitude and longitude of the four corners of cell (i, j), in the order [i, j], [i, j+1], [i+1, j+1], [i+1, j].

    Uses the neighbours of the cell in the full grid, so the result is the same as corner_grid on the whole grid.
    """
    lat2d, lon2d = np.asarray(lat2d), np.asarray(lon2d)
    rows, cols = _window(lat2d.shape, i, i, j, j)
    clat, clon = corner_grid(lat2d[rows, cols], lon2d[rows, cols])
    a, b = i - rows.start, j - cols.start
    rows_k = np.array([a, a, a + 1, a + 1])
    cols_k = np.array([b, b + 1, b + 1, b])
    return clat[rows_k, cols_k], clon[rows_k, cols_k]


def cell_size_km(lat2d, lon2d, i: int, j: int) -> float:
    """The mean distance from the centre of cell (i, j) to the centres of its row and column neighbours, in km."""
    lat2d, lon2d = np.asarray(lat2d), np.asarray(lon2d)
    n, m = lat2d.shape
    near = [(a, b) for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)) if 0 <= a < n and 0 <= b < m]
    rows, cols = zip(*near)
    distances = _great_circle_km(lat2d[i, j], lon2d[i, j], lat2d[rows, cols], lon2d[rows, cols])
    return float(np.mean(distances))


def cells_near(lat2d, lon2d, lat0: float, lon0: float, radius_km: float) -> list:
    """The cells (i, j) whose centre is within radius_km (great circle) of (lat0, lon0), sorted by index."""
    distance = _great_circle_km(lat2d, lon2d, lat0, lon0)
    return [(int(i), int(j)) for i, j in zip(*np.nonzero(distance <= radius_km))]


# --- drawing ----------------------------------------------------------------------------------------------------------


def draw_cells(
    ax: plt.Axes,
    lat2d,
    lon2d,
    centre: Point,
    radius_km: float,
    highlight: Optional[Mapping[Cell, str]] = None,
    grid_line: Optional[dict] = None,
    cell_line: Optional[dict] = None,
) -> Dict[Cell, Point]:
    """The outlines of the cells within radius_km of centre (lat0, lon0) as thin light grey lines, and the cells of
    highlight ({(i, j): line style}) as dark lines in their line style (style.CURRENT, style.ALTERNATIVE, ...).

    A highlighted cell is drawn even if it is not within radius_km. Returns the centre of each highlighted cell in
    local km, {(i, j): (x, y)}. Choose radius_km of about half the panel diagonal plus one cell, so the grey cells fill
    the panel; the axes clip what is outside.
    """
    lat2d, lon2d = np.asarray(lat2d, dtype=float), np.asarray(lon2d, dtype=float)
    lat0, lon0 = centre
    highlight = dict(highlight or {})
    near = set(cells_near(lat2d, lon2d, lat0, lon0, radius_km)) | set(highlight)
    if near:
        i_all, j_all = zip(*near)
        rows, cols = _window(lat2d.shape, min(i_all), max(i_all), min(j_all), max(j_all))
        clat, clon = corner_grid(lat2d[rows, cols], lon2d[rows, cols])
        cx, cy = local_km(clat, clon, lat0, lon0)

        def polygon(i, j):
            a, b = i - rows.start, j - cols.start
            rows_k = [a, a, a + 1, a + 1, a]
            cols_k = [b, b + 1, b + 1, b, b]
            return np.column_stack([cx[rows_k, cols_k], cy[rows_k, cols_k]])

        grey = [polygon(i, j) for i, j in sorted(near) if (i, j) not in highlight]
        ax.add_collection(LineCollection(grey, zorder=1, **{**GRID_LINE, **(grid_line or {})}))
        for (i, j), linestyle in highlight.items():
            xy = polygon(i, j)
            ax.plot(xy[:, 0], xy[:, 1], linestyle=linestyle, zorder=3, **{**CELL_LINE, **(cell_line or {})})
    centres = {}
    for i, j in highlight:
        x, y = local_km(lat2d[i, j], lon2d[i, j], lat0, lon0)
        centres[(i, j)] = (float(x), float(y))
    return centres


def draw_outline(ax: plt.Axes, lat2d, lon2d, i: int, j: int, centre: Point, **line) -> Point:
    """The outline of one cell (for example the cell of a second RCM), with the line properties of line (default
    OTHER_GRID_LINE). Returns the centre of the cell in local km."""
    lat0, lon0 = centre
    clat, clon = cell_outline(lat2d, lon2d, i, j)
    x, y = local_km(np.append(clat, clat[0]), np.append(clon, clon[0]), lat0, lon0)
    ax.plot(x, y, zorder=2, **{**OTHER_GRID_LINE, **line})
    cx, cy = local_km(lat2d[i, j], lon2d[i, j], lat0, lon0)
    return float(cx), float(cy)


def draw_site(
    ax: plt.Axes,
    x: float,
    y: float,
    name: str,
    offset: Tuple[float, float] = (5, 4),
    ha: str = "left",
    va: str = "bottom",
) -> None:
    """A site as a black triangle, with its name in bold black, offset by offset points from the marker."""
    ax.plot(x, y, **SITE_MARKER)
    ax.annotate(
        name,
        (x, y),
        xytext=offset,
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=8,
        fontweight="bold",
        color=style.TEXT,
        bbox=LABEL_BOX,
        zorder=7,
    )


def draw_centre(
    ax: plt.Axes,
    x: float,
    y: float,
    marker: Optional[dict] = None,
    label: Optional[str] = None,
    offset: Tuple[float, float] = (5, -4),
    ha: str = "left",
    va: str = "top",
) -> None:
    """A cell centre (default a dark "+", CENTRE_MARKER), with an optional label, for example the cell index and the
    distance to the site, "261, 126\n6.70 km"."""
    ax.plot(x, y, zorder=5, **{**CENTRE_MARKER, **(marker or {})})
    if label:
        ax.annotate(
            label,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=7,
            color=style.TEXT,
            linespacing=1.1,
            bbox=LABEL_BOX,
            zorder=7,
        )


def connect(ax: plt.Axes, start: Point, end: Point) -> None:
    """A thin grey connector line from start to end (local km), for example from a site to a cell centre."""
    ax.plot([start[0], end[0]], [start[1], end[1]], zorder=2, **CONNECTOR)


def distance_label(
    ax: plt.Axes,
    start: Point,
    end: Point,
    text: str,
    at: float = 0.5,
    offset: Tuple[float, float] = (0, 0),
    line: bool = True,
) -> None:
    """A thin grey connector from start to end (for example a site and a cell centre, in local km), with text on a
    white box at the fraction at of the way along it, moved by offset points."""
    (x0, y0), (x1, y1) = start, end
    if line:
        connect(ax, start, end)
    ax.annotate(
        text,
        (x0 + at * (x1 - x0), y0 + at * (y1 - y0)),
        xytext=offset,
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=7,
        color=style.TEXT,
        bbox=LABEL_BOX,
        zorder=8,
    )


def format_km(distance_km: float) -> str:
    """3 significant figures: "6.70 km", "16.0 km"."""
    return f"{distance_km:#.3g} km" if distance_km < 100 else f"{distance_km:.0f} km"


def map_axes(
    ax: plt.Axes,
    half_width_km: float,
    view_offset_km: Point = (0.0, 0.0),
    xlabel: bool = True,
    ylabel: bool = True,
) -> None:
    """Square axes of 2 * half_width_km around view_offset_km (x, y), equal aspect, labelled in km east and north."""
    dx, dy = view_offset_km
    ax.set_xlim(dx - half_width_km, dx + half_width_km)
    ax.set_ylim(dy - half_width_km, dy + half_width_km)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    if xlabel:
        ax.set_xlabel("Distance east (km)")
    if ylabel:
        ax.set_ylabel("Distance north (km)")


def legend_entries(other_rcm: Optional[str] = None, grid_name: str = "ANT-12") -> list:
    """The (label, Line2D properties) pairs of the map: corrected and old cell, the other cells, the centres, the
    sites, and with other_rcm, the cell and centre of the second RCM (overlay_cell)."""

    def marker(props):
        return {k: v for k, v in props.items() if k != "zorder"}

    entries = [
        ("Corrected cell", {**CELL_LINE, "linestyle": style.CURRENT}),
        ("Old cell", {**CELL_LINE, "linestyle": style.ALTERNATIVE}),
        (f"Other {grid_name} cells", GRID_LINE),
        (f"{grid_name} cell centre", marker(CENTRE_MARKER)),
    ]
    if other_rcm:
        entries += [(f"{other_rcm} cell", OTHER_GRID_LINE), (f"{other_rcm} cell centre", marker(OTHER_CENTRE_MARKER))]
    return entries + [("Site", marker(SITE_MARKER))]


def cells_legend(target, entries: Sequence[Tuple[str, dict]], **kwargs) -> None:
    """A legend on an Axes or a Figure from (label, Line2D properties) pairs (legend_entries), for example
    ("Corrected cell", {**CELL_LINE, "linestyle": "-"})."""
    handles = [Line2D([], [], **props) for _, props in entries]
    target.legend(handles, [label for label, _ in entries], **{"loc": "lower left", "fontsize": 7, **kwargs})


# --- figure 1 ---------------------------------------------------------------------------------------------------------


@dataclass
class MapFrame:
    """The frame of one map panel: the centre (lat0, lon0), and the positions in local km of its sites."""

    lat0: float
    lon0: float
    sites: Dict[str, Point] = field(default_factory=dict)
    site_coords: Dict[str, Point] = field(default_factory=dict)  # (lat, lon) of each site
    centres: Dict[Cell, Point] = field(default_factory=dict)  # local km of the highlighted cells of the main RCM


def rcm_grid(rcm_name: str, data_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """lat2d, lon2d of an RCM ("RACMO", "MAR", ...) from force.model_point._grid (cached; opens one file)."""
    model_point = importlib.import_module("force.model_point")  # the force package exports a function of this name
    _, _, lat2d, lon2d = model_point._grid(rcm_name, data_path or model_point.rcm_data_path(rcm_name))
    return lat2d, lon2d


def _distance_km(frame: MapFrame, site: str, lat2d, lon2d, cell: Cell) -> float:
    lat, lon = frame.site_coords[site]
    return float(_great_circle_km(lat, lon, lat2d[cell], lon2d[cell]))


def _distance_text(distance_km: float, site: str, named: bool) -> str:
    return f"{format_km(distance_km)} to {site}" if named else format_km(distance_km)


def site_map(
    ax: plt.Axes,
    rcm_name: str,
    site_names: Sequence[str],
    highlight: Mapping[Cell, str],
    distances: Optional[Iterable[Tuple[str, Cell]]] = None,
    distance_text: str = "centre",
    half_width_km: float = 22.5,
    view_offset_km: Point = (0.0, 0.0),
    centre: Optional[Point] = None,
    data_path: Optional[str] = None,
    site_labels: Optional[Mapping[str, dict]] = None,
    centre_labels: Optional[Mapping[Cell, dict]] = None,
    distance_labels: Optional[Mapping[Tuple[str, Cell], dict]] = None,
) -> MapFrame:
    """One map panel: the cells of rcm_name around the panel centre, the highlighted cells ({(i, j): line style}),
    their centres, the sites, and a connector with the distance from a site to a cell centre for each (site, cell) of
    distances.

    The coordinates are km from centre (lat, lon), by default the first site. The panel shows 2 * half_width_km
    around view_offset_km (x, y) of that point, so the view can move while the axes stay relative to the site.
    distances defaults to the first site and each highlighted cell. The distances are great circle distances
    (force.grid.great_circle_km). With distance_text "centre", the distances are written under the cell index next to
    the cell centre (best for short connectors); with "line", on the connector (distance_label). The site name is
    added to a distance when the panel has more than one site. site_labels, centre_labels and distance_labels give
    keyword arguments of draw_site, draw_centre and distance_label for single labels, to move them away from each
    other. Returns the MapFrame, for overlay_cell.
    """
    from cfm_config import get_site

    lat2d, lon2d = rcm_grid(rcm_name, data_path)
    sites = {name: get_site(name) for name in site_names}
    first = sites[site_names[0]]
    lat0, lon0 = centre or (first.lat, first.lon)
    frame = MapFrame(lat0, lon0)
    frame.site_coords = {name: (s.lat, s.lon) for name, s in sites.items()}
    for name, (lat, lon) in frame.site_coords.items():
        x, y = local_km(lat, lon, lat0, lon0)
        frame.sites[name] = (float(x), float(y))

    radius = np.hypot(half_width_km + abs(view_offset_km[0]), half_width_km + abs(view_offset_km[1]))
    radius += 1.5 * cell_size_km(lat2d, lon2d, *next(iter(highlight)))
    frame.centres = draw_cells(ax, lat2d, lon2d, (lat0, lon0), radius, highlight)

    named = len(site_names) > 1
    centre_text = {cell: [f"{cell[0]}, {cell[1]}"] for cell in highlight}
    for name, cell in [(site_names[0], cell) for cell in highlight] if distances is None else distances:
        text = _distance_text(_distance_km(frame, name, lat2d, lon2d, cell), name, named)
        if distance_text == "line":
            distance_label(
                ax, frame.sites[name], frame.centres[cell], text, **(distance_labels or {}).get((name, cell), {})
            )
        else:
            connect(ax, frame.sites[name], frame.centres[cell])
            centre_text[cell].append(text)
    for cell, (x, y) in frame.centres.items():
        draw_centre(ax, x, y, label="\n".join(centre_text[cell]), **(centre_labels or {}).get(cell, {}))

    for name, (x, y) in frame.sites.items():
        draw_site(ax, x, y, name, **(site_labels or {}).get(name, {}))
    map_axes(ax, half_width_km, view_offset_km)
    return frame


def overlay_cell(
    ax: plt.Axes,
    frame: MapFrame,
    rcm_name: str,
    cell: Cell,
    distances_to: Iterable[str] = (),
    distance_text: str = "centre",
    data_path: Optional[str] = None,
    label: Optional[str] = None,
    centre_label: Optional[dict] = None,
    distance_labels: Optional[Mapping[str, dict]] = None,
    line: Optional[dict] = None,
) -> Point:
    """One cell of a second RCM on a map panel (for example the MAR cell on the RACMO map): its outline (the part in
    the panel; OTHER_GRID_LINE, or line), its centre as a cross, and a connector with the distance to each site of
    distances_to (distance_text as in site_map). Returns the centre of the cell in local km."""
    lat2d, lon2d = rcm_grid(rcm_name, data_path)
    x, y = draw_outline(ax, lat2d, lon2d, *cell, (frame.lat0, frame.lon0), **(line or {}))
    lines = [label or f"{rcm_name} {cell[0]}, {cell[1]}"]
    named = len(frame.sites) > 1
    for name in distances_to:
        text = _distance_text(_distance_km(frame, name, lat2d, lon2d, cell), name, named)
        if distance_text == "line":
            distance_label(ax, frame.sites[name], (x, y), text, **(distance_labels or {}).get(name, {}))
        else:
            connect(ax, frame.sites[name], (x, y))
            lines.append(text)
    draw_centre(ax, x, y, marker=OTHER_CENTRE_MARKER, label="\n".join(lines), **(centre_label or {}))
    return x, y
