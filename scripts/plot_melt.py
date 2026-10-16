"""Script to plot average yearly melt across the AIS peninsula for MAR and RACMO, with borehole locations overlaid.

Calculates melt averaged over the period specified in config

Usage:
    - plot_melt.py
"""

from plot import plot_larsen_c_melt
from cfm_config import sites
from force.model_point import model_point_label, shared_model_points

if __name__ == "__main__":
    active_sites = sites()
    borehole_sites = {name: (site.lat, site.lon) for name, site in active_sites.items()}
    # legend labels that show the sites that share a grid cell (FEAT-002)
    labels = {}
    for rcm in ("MAR", "RACMO"):
        groups = shared_model_points(sites(active_only=False), rcm)
        labels[rcm] = {name: model_point_label(name, groups) for name in active_sites}

    plot_larsen_c_melt(
        rcm_name="mar",
        lat_name="LAT",
        lon_name="LON",
        borehole_sites=borehole_sites,
        labels=labels["MAR"],
    )

    plot_larsen_c_melt(
        rcm_name="racmo",
        lat_name="lat",
        lon_name="lon",
        borehole_sites=borehole_sites,
        labels=labels["RACMO"],
    )
