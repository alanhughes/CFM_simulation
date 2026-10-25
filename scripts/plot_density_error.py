"""
Working script for plotting simulated density profiles against in situ density profiles
could be integrated into src/plot/results.py if we decide to keep these plots

Usage:
    python plot_density_error.py [--cfm <label>] [--start-year <year>] [--end-year <year>] [--spin-end-year <year>|any]

The runs are found from their manifests (sim.runs.find_run), with the CFM version from --cfm (default_cfm_version if
not given). The years are those of scripts/run_cfm.py (force.years.run_years, DISC-001): start_year and end_year of
config.yaml, with the end year clipped to the last complete year of the RCM data (RACMO: 1979-2023), or
--start-year/--end-year as given. The runs have the spin_end_year of config.yaml, or --spin-end-year as given (a year,
or any for the runs of any spin_end_year, FEAT-004).

Plots the active sites in sites.yaml, each with its own model time windows (the "windows" key). A site with no
windows stops the script with an error (BUG-005). Requires the physical densification schemes
to plot in physrho_values list, and the RCM and melt scheme to use for loading the correct simulation output.

The script will
- load in situ density data for each site
- load the corresponding CFM simulation output for each physical densification scheme
- interpolate the model density to the in situ depth grid
- calculate the mean model density across all time steps in the windows of the site
- and plot the difference between the model and in situ density profiles with depth.

The resulting plots will be saved to the specified figure path.

"""

import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
import argparse
import logging
import os

from cfm_config import cfm_version, config, sites
from force.model_point import model_point_label, shared_model_points
from force.years import add_year_arguments, run_years
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

    borehole_sites = sites()
    physrho_values = [
        "GSFC2020",
        "HLdynamic",
        "Crocus",
        "Barnola1991",
        "Ligtenberg2011",
    ]
    rcm_name = config["rcm_name"]
    melt_scheme = config["cfm_config"]["liquid"]
    years = run_years(rcm_name, args.start_year, args.end_year)

    # model point of each site; also the inactive sites, to show all the sites that share a grid cell (FEAT-002)
    model_points = shared_model_points(sites(active_only=False), rcm_name)

    save_dir = f"{config['CFM_data_path']}/cfm_figures/errors"
    os.makedirs(save_dir, exist_ok=True)

    for site, info in borehole_sites.items():
        # model time windows of this site, in decimal years (sites.yaml) - stops with an error if it has none (BUG-005)
        windows = info.decimal_year_windows()

        # load in in situ density data for this site
        in_situ_path = f"{config['in_situ_data_path']}/{info.profile_file}"
        in_situ_data = pd.read_csv(in_situ_path)
        # put in situ data on same depth grid as model output
        in_situ_depth = pd.to_numeric(in_situ_data["Depth"].values)
        in_situ_density = pd.to_numeric(in_situ_data["Density"].values)
        plt.figure(figsize=(6, 8))
        # plt.plot(in_situ_density, in_situ_depth, label="In situ", color="k", linewidth=1)
        for physrho in physrho_values:
            # load simulation
            output_path = find_run(
                site=site,
                rcm_name=rcm_name,
                physRho=physrho,
                liquid=melt_scheme,
                start_year=years.start_year,
                end_year=years.end_year,
                spin_end_year=spin_end_year,
                cfm=cfm,
            ).results_file
            with xr.open_dataset(output_path, engine="h5netcdf", phony_dims="sort") as ds:
                results_dict = {
                    "model_time_matrix": ds["density"][1:, 0],
                    "model_time_vector": ds["DIP"][1:, 0],
                    "depth": ds["depth"][1:],
                    "density": ds["density"][1:, 1:],
                    "temperature": ds["temperature"][1:, 1:],
                    "DIP": ds["DIP"][1:, 1:],
                }

            # only keep data from the measurement windows of this site
            model_time = results_dict["model_time_vector"]
            summer_mask = np.zeros(model_time.shape, dtype=bool)
            for start, end in windows:
                summer_mask |= ((model_time >= start) & (model_time <= end)).values

            # only keep density data from astrochronological summers
            density_summer = results_dict["density"][summer_mask, :]

            # only keep model data from depths where we have in situ data
            depth_mask = (results_dict["depth"] >= in_situ_depth.min()) & (results_dict["depth"] <= in_situ_depth.max())

            # interpolate in model data to in situ depth grid
            density_summer_interp = np.empty((density_summer.shape[0], len(in_situ_depth)))
            for i in range(density_summer.shape[0]):
                density_summer_interp[i, :] = np.interp(
                    in_situ_depth,
                    results_dict["depth"].values[depth_mask],
                    density_summer[i, depth_mask],
                )

            # get mean model density at each depth across all model time steps in astrochronological summers
            density_summer_mean = np.mean(density_summer_interp, axis=0)
            density_summer_lower = np.percentile(density_summer_interp, 25, axis=0)
            density_summer_upper = np.percentile(density_summer_interp, 75, axis=0)

            # plot rmse vs depth
            plt.plot(
                density_summer_mean,
                in_situ_depth,
                label=f"{physrho}",
                linestyle="--",
            )
            plt.fill_betweenx(
                in_situ_depth,
                density_summer_lower,
                density_summer_upper,
                alpha=0.3,
                color=plt.gca().lines[-1].get_color(),
            )
        # also plot in situ density profile for reference
        # add vertical line at 0 to indicate perfect agreement between model and in situ
        plt.plot(in_situ_density, in_situ_depth, label="In situ", color="k", linewidth=1)
        plt.gca().invert_yaxis()
        plt.xlabel("Density (kg/m³)")
        plt.ylabel("Depth (m)")
        plt.title(f"Density vs Depth for {site}\n{model_point_label(site, model_points)}", fontsize=9)
        plt.legend()
        plt.grid()
        plt.savefig(f"{save_dir}/{site}_{melt_scheme}_{rcm_name}_{cfm}.png")
