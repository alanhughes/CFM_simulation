"""Script to plot average yearly melt across the AIS peninsula for MAR and RACMO, with borehole locations overlaid.

Plots the melt maps of scripts/run_melt.py. Each RCM has its own years, as in scripts/run_cfm.py
(force.years.run_years, DISC-001): start_year and end_year of config.yaml, with the end year clipped to the last
complete year of the RCM data (RACMO: 1979-2023), or --start-year/--end-year as given.

Usage:
    - plot_melt.py [--start-year <year>] [--end-year <year>]
"""

import argparse

from cfm_config import sites
from force.model_point import model_point_label, shared_model_points
from force.years import add_year_arguments, run_years, set_config_years
from plot import plot_larsen_c_melt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the average yearly melt maps of MAR and RACMO for Larsen C")
    add_year_arguments(parser)
    args = parser.parse_args()
    # plot_larsen_c_melt reads the years from config
    years = {rcm: run_years(rcm, args.start_year, args.end_year) for rcm in ("MAR", "RACMO")}

    active_sites = sites()
    borehole_sites = {name: (site.lat, site.lon) for name, site in active_sites.items()}
    # legend labels that show the sites that share a grid cell (FEAT-002)
    labels = {}
    for rcm in ("MAR", "RACMO"):
        groups = shared_model_points(sites(active_only=False), rcm)
        labels[rcm] = {name: model_point_label(name, groups) for name in active_sites}

    set_config_years(years["MAR"])
    plot_larsen_c_melt(
        rcm_name="mar",
        lat_name="LAT",
        lon_name="LON",
        borehole_sites=borehole_sites,
        labels=labels["MAR"],
    )

    set_config_years(years["RACMO"])
    plot_larsen_c_melt(
        rcm_name="racmo",
        lat_name="lat",
        lon_name="lon",
        borehole_sites=borehole_sites,
        labels=labels["RACMO"],
    )
