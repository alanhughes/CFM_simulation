"""
Plot the simulated density profiles of each active site against its in situ density profile.

Usage:
    python plot_density_error.py [--cfm <label>] [--start-year <year>] [--end-year <year>] [--spin-end-year <year>|any]

The runs are found from their manifests (sim.runs.find_run), with the CFM version from --cfm (default_cfm_version if
not given). The years are those of scripts/run_cfm.py (force.years.run_years, DISC-001): start_year and end_year of
config.yaml, with the end year clipped to the last complete year of the RCM data (RACMO: 1979-2023), or
--start-year/--end-year as given. The runs have the spin_end_year of config.yaml, or --spin-end-year as given (a year,
or any for the runs of any spin_end_year, FEAT-004).

Plots the active sites in sites.yaml, each with its own model time windows (the "windows" key). A site with no
windows stops the script with an error (BUG-005). The RCM and the melt scheme are those of config.yaml.

For each site and scheme, the model density of each time step in the windows is interpolated to the in situ depths
(plot.metrics.compare_with_profile, FEAT-007). The plot shows the mean of those steps as a line, their range as a
band, and the mean bias in the legend. The in situ profile is drawn raw (grey) and as a 1 m running mean (black).

The figures are saved to {CFM_data_path}/cfm_figures/errors.
"""

import argparse
import logging
import os

import matplotlib.pyplot as plt

from cfm_config import cfm_version, config, sites
from force.model_point import model_point_label, shared_model_points
from force.years import add_year_arguments, run_years
from plot import profiles, style
from plot.output import RunOutput, site_profile
from sim.runs import add_spin_end_year_argument, find_run, spin_end_year_filter

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the model density against the in situ density for each site")
    parser.add_argument("--cfm", help="CFM version (a label in cfm_versions); default_cfm_version if not given")
    add_year_arguments(parser)
    add_spin_end_year_argument(parser)
    args = parser.parse_args()
    cfm = cfm_version(args.cfm)["label"]
    spin_end_year = spin_end_year_filter(args.spin_end_year)

    rcm_name = config["rcm_name"]
    melt_scheme = config["cfm_config"]["liquid"]
    years = run_years(rcm_name, args.start_year, args.end_year)

    # model point of each site; also the inactive sites, to show all the sites that share a grid cell (FEAT-002)
    model_points = shared_model_points(sites(active_only=False), rcm_name)

    save_dir = f"{config['CFM_data_path']}/cfm_figures/errors"
    style.apply()

    for site, info in sites().items():
        # model time windows of this site, in decimal years (sites.yaml) - stops with an error if it has none (BUG-005)
        windows = info.decimal_year_windows()
        profile = site_profile(site)

        fig, ax = plt.subplots(figsize=(4.2, 6.0))
        profiles.draw_observed(ax, profile)
        for scheme in style.SCHEMES:
            run = find_run(
                site=site,
                rcm_name=rcm_name,
                physRho=scheme,
                liquid=melt_scheme,
                start_year=years.start_year,
                end_year=years.end_year,
                spin_end_year=spin_end_year,
                cfm=cfm,
            )
            comparison = RunOutput.from_run(run).compare_with_profile(profile, windows)
            profiles.draw_comparison(
                ax, comparison, scheme, band=True, label=f"{scheme} ({comparison.bias:+.0f} kg/m³)"
            )
        profiles.reference_value(ax, 830, "830")
        profiles.depth_axes(ax, profile.depth.max(), "Density (kg/m³)")
        profiles.scheme_legend(ax)
        ax.set_title(f"{site}, {rcm_name}, {cfm}\n{model_point_label(site, model_points)}", fontsize=8)
        path = style.save_figure(fig, f"{save_dir}/{site}_{melt_scheme}_{rcm_name}_{cfm}.png")
        logging.info(f"Saved {path}")
