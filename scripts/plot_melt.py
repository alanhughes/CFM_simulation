"""Script to plot average yearly melt across the AIS peninsula for MAR and RACMO, with borehole locations overlaid.

Calculates melt averaged over the period specified in config

Usage:
    - plot_melt.py
"""

from plot import plot_larsen_c_melt
from cfm_config import sites

if __name__ == "__main__":
    borehole_sites = {name: (site.lat, site.lon) for name, site in sites().items()}

    plot_larsen_c_melt(
        rcm_name="mar",
        lat_name="LAT",
        lon_name="LON",
        borehole_sites=borehole_sites,
    )

    plot_larsen_c_melt(
        rcm_name="racmo",
        lat_name="lat",
        lon_name="lon",
        borehole_sites=borehole_sites,
    )
