"""Script to calculate the rate of change of ice thickness (dh/dt) from the CFM output data and plot it over time for
different physical density options. The script reads the output data from NetCDF files, extracts the relevant variables,
and creates time series plots of dh/dt for each physical density option. Sites, RCM, melt scheme and period are taken
from config.yaml. The time series are saved to csv files, and the plots and csvs are written to
{CFM_data_path}/cfm_figures/dh_dt

The runs are found from their manifests (sim.runs.find_run), with the CFM version from --cfm (default_cfm_version if
not given).

The years are those of scripts/run_cfm.py (force.years.run_years, DISC-001): start_year and end_year of config.yaml,
with the end year clipped to the last complete year of the RCM data (RACMO: 1979-2023), or --start-year/--end-year as
given. The runs have the spin_end_year of config.yaml, or --spin-end-year as given (a year, or any for the runs of any
spin_end_year, FEAT-004).

Usage:
    - python calc_dh.py [--cfm <label>] [--start-year <year>] [--end-year <year>] [--spin-end-year <year>|any]

"""

import argparse
import os
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from cfm_config import cfm_version, config, sites
from force.cache import find_forcing_csv
from force.model_point import model_point, model_point_label, shared_model_points
from force.years import add_year_arguments, run_years
from sim.runs import add_spin_end_year_argument, find_run, spin_end_year_filter


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
    parser = argparse.ArgumentParser(description="Calculate and plot dh/dt for the sites in sites.yaml")
    parser.add_argument("--cfm", help="CFM version (a label in cfm_versions); default_cfm_version if not given")
    add_year_arguments(parser)
    add_spin_end_year_argument(parser)
    args = parser.parse_args()
    cfm = cfm_version(args.cfm)["label"]
    spin_end_year = spin_end_year_filter(args.spin_end_year)

    borehole_sites = sites()
    phys_rho_options = ("GSFC2020", "HLdynamic", "Crocus", "Barnola1991", "Ligtenberg2011")
    rcm_name = config["rcm_name"]
    years = run_years(rcm_name, args.start_year, args.end_year)
    start, end = years.start_year, years.end_year
    liquid = config["cfm_config"]["liquid"]

    # model point of each site; also the inactive sites, to show all the sites that share a grid cell (FEAT-002)
    model_points = shared_model_points(sites(active_only=False), rcm_name)

    save_dir = f"{config['CFM_data_path']}/cfm_figures/dh_dt"
    os.makedirs(save_dir, exist_ok=True)

    for site, info in borehole_sites.items():
        input_path = find_forcing_csv(model_point(info, rcm_name), start, end)
        input_df = pd.read_csv(input_path, parse_dates=["TIME"])
        input_df = input_df[input_df["TIME"] >= pd.Timestamp("1980-01-01")]

        plt.figure(figsize=(10, 6))
        for phys_rho in phys_rho_options:
            output_path = find_run(
                site=site,
                rcm_name=rcm_name,
                physRho=phys_rho,
                liquid=liquid,
                start_year=start,
                end_year=end,
                spin_end_year=spin_end_year,
                cfm=cfm,
                cfm_config={"SEB": True},  # the CFM's own surface energy balance (FEAT-009)
            ).results_file
            results_dict = read_output_data(output_path)

            # Align lengths so time and DIP-derived series are paired safely.
            dH_step = results_dict["dH_step"].values

            # plot
            plt.plot(input_df["TIME"], dH_step, label=phys_rho, alpha=0.5)

            # save
            output_df = pd.DataFrame({"TIME": input_df["TIME"], "dh_dt": dH_step})
            output_df.to_csv(
                f"{save_dir}/dh_dt_time_series_{site}_{phys_rho}_{liquid}_{rcm_name}_{cfm}.csv", index=False
            )

        plt.xlabel("Time")
        plt.ylabel("dh/dt (m/day)")
        plt.title(
            f"Rate of change of ice thickness (dh/dt) over time at {site}\n{model_point_label(site, model_points)}"
        )
        plt.legend()
        plt.grid()
        plt.savefig(f"{save_dir}/dh_dt_time_series_{site}_{liquid}_{rcm_name}_{cfm}.png", dpi=300)
        plt.close()
