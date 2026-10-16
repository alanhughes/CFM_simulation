"""
Working script for plotting simulated density profiles against in situ density profiles
could be integrated into src/plot/results.py if we decide to keep these plots

Usage:
    python density_error.py

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
import logging
import os

from cfm_config import cfm_output_folder, config, sites

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
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

    save_dir = f"{config['CFM_data_path']}/cfm_figures/errors"
    os.makedirs(save_dir, exist_ok=True)

    for site, info in borehole_sites.items():
        lat, lon = info.lat, info.lon
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
            output_path = f"{cfm_output_folder(lat, lon, physrho, melt_scheme, rcm_name)}/CFMresults.hdf5"
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
        plt.title(f"Density vs Depth for {site}")
        plt.legend()
        plt.grid()
        plt.savefig(f"{save_dir}/{site}_{melt_scheme}_{rcm_name}.png")
