"""Distances on the RCM grids (BUG-002)."""

from typing import Tuple

import numpy as np

EARTH_RADIUS_KM = 6371.0


def great_circle_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Haversine distance in km on a sphere of radius EARTH_RADIUS_KM. Works on scalars and numpy arrays."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def nearest_cell(lat2d, lon2d, lat: float, lon: float) -> Tuple[int, int, float]:
    """Indices (i, j) of the cell centre nearest to (lat, lon) by great circle distance, and that distance in km.

    i is the row index (rlat / Y) and j the column index (rlon / X) of the 2D latitude/longitude arrays. This replaces
    the old search in degrees, which gave a longitude difference ~2.5 times too much weight at 67 S (BUG-002).
    """
    distance = great_circle_km(lat2d, lon2d, lat, lon)
    i, j = np.unravel_index(np.nanargmin(distance), distance.shape)
    return int(i), int(j), float(distance[i, j])
