"""Script to calculate the rate of change of ice thickness (dh/dt) from the CFM output data and plot it over time for
different physical density options. The script reads the output data from NetCDF files, extracts the relevant variables,
and creates time series plots of dh/dt for each physical density option. Sites, RCM, melt scheme and period are taken
from config.yaml. The time series are saved to csv files, and the plots and csvs are written to
{CFM_data_path}/cfm_figures/dh_dt

Usage:
    - python calc_dh.py

"""

import os
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from cfm_config import cfm_output_folder, config, sites
from force.cache import forcing_csv_path
from force.model_point import model_point, model_point_label, shared_model_points


def read_output_data(file_path):
    """
    Reads the output data from a NetCDF file and returns it as an xarray Dataset.

    Parameters:
    file_path (str): The path to the NetCDF file containing the output data.

    Returns:
    dict: A dictionary containing the model time matrix, model time vector, depth, density, temperature, and DIP.
    """

    with xr.open_dataset(file_path, engine="h5netcdf", phony_dims="sort") as ds:
        results_dict = {
            "model_time_matrix": ds["density"][1:, 0],
            "model_time_vector": ds["DIP"][1:, 0],
            "depth": ds["depth"][1:],
            "density": ds["density"][1:, 1:],
            "temperature": ds["temperature"][1:, 1:],
            "DIP": ds["DIP"][1:, 1:],
            # "compaction": ds["compaction"][1:, 1:],
            # dHOut from CFM update_dH: column index 2 in full DIP array,
            # units are meters of surface-height change per model step.
            "dH_step": ds["DIP"][1:, 2],
        }

    return results_dict


if __name__ == "__main__":
    borehole_sites = sites()
    phys_rho_options = ("GSFC2020", "HLdynamic", "Crocus", "Barnola1991", "Ligtenberg2011")
    start, end = config["start_year"], config["end_year"]
    rcm_name = config["rcm_name"]
    liquid = config["cfm_config"]["liquid"]

    # model point of each site; also the inactive sites, to show all the sites that share a grid cell (FEAT-002)
    model_points = shared_model_points(sites(active_only=False), rcm_name)

    save_dir = f"{config['CFM_data_path']}/cfm_figures/dh_dt"
    os.makedirs(save_dir, exist_ok=True)

    for site, info in borehole_sites.items():
        lat, lon = info.lat, info.lon
        input_path = forcing_csv_path(model_point(info, rcm_name), start, end)
        input_df = pd.read_csv(input_path, parse_dates=["TIME"])
        input_df = input_df[input_df["TIME"] >= pd.Timestamp("1980-01-01")]

        plt.figure(figsize=(10, 6))
        for phys_rho in phys_rho_options:
            output_path = f"{cfm_output_folder(lat, lon, phys_rho, liquid, rcm_name)}/CFMresults.hdf5"
            results_dict = read_output_data(output_path)

            # Align lengths so time and DIP-derived series are paired safely.
            dH_step = results_dict["dH_step"].values

            # plot
            plt.plot(input_df["TIME"], dH_step, label=phys_rho, alpha=0.5)

            # save
            output_df = pd.DataFrame({"TIME": input_df["TIME"], "dh_dt": dH_step})
            output_df.to_csv(f"{save_dir}/dh_dt_time_series_{site}_{phys_rho}_{liquid}_{rcm_name}.csv", index=False)

        plt.xlabel("Time")
        plt.ylabel("dh/dt (m/day)")
        plt.title(
            f"Rate of change of ice thickness (dh/dt) over time at {site}\n{model_point_label(site, model_points)}"
        )
        plt.legend()
        plt.grid()
        plt.savefig(f"{save_dir}/dh_dt_time_series_{site}_{liquid}_{rcm_name}.png", dpi=300)
        plt.close()
