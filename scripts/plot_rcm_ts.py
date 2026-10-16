"""
Script to generate plots of time series for MAR and RACMO forcing data at specified borehole sites. The script will:
- Load the MAR and RACMO forcing data for each borehole site from the processed csv
- Generate time series plots of the forcing variables (e.g., temperature, accumulation) for each site and each RCM.
The plots will show the temporal variability in the forcing data at the borehole locations.
- Overlay the time series for MAR and RACMO on the same plot for each variable to compare the two RCMs at each site.
The resulting plots will be saved to the specified figure path.

Usage:
    python rcm_ts_plots.py

"""

import os

import matplotlib.pyplot as plt
import pandas as pd

import logging

logging.basicConfig(level=logging.INFO)

from cfm_config import config, sites

if __name__ == "__main__":
    borehole_sites = sites()
    mar_data_dict: dict[str, pd.DataFrame] = {}
    racmo_data_dict: dict[str, pd.DataFrame] = {}

    for site, info in borehole_sites.items():
        lat, lon = info.lat, info.lon
        # Load MAR data for this site
        mar_data_path = (
            f"{config['CFM_data_path']}/cfm_input/MAR_{lat}_{lon}_{config['start_year']}_{config['end_year']}.csv"
        )
        mar_data_dict[site] = pd.read_csv(mar_data_path)

        # Load RACMO data for this site
        racmo_data_path = (
            f"{config['CFM_data_path']}/cfm_input/RACMO_{lat}_{lon}_{config['start_year']}_{config['end_year']}.csv"
        )
        racmo_data_dict[site] = pd.read_csv(racmo_data_path)

    save_dir = f"{config['CFM_data_path']}/cfm_figures/rcm_comp"
    os.makedirs(save_dir, exist_ok=True)

    # Generate time series plots for each site and variable
    for site in borehole_sites.keys():
        mar_data = mar_data_dict[site]
        racmo_data = racmo_data_dict[site]

        # Example variables to plot (these should match the columns in your data)
        variables_to_plot = mar_data.columns.difference(["TIME"])  # Assuming 'TIME' is the time column

        for var in variables_to_plot:
            # get monthly moving average for smoother plots
            mar_data[var] = mar_data[var].rolling(window=30, center=True).mean()
            racmo_data[var] = racmo_data[var].rolling(window=30, center=True).mean()

            plt.figure(figsize=(10, 6))
            plt.plot(mar_data[var], label="MAR", color="blue", linestyle="--", alpha=0.5)
            plt.plot(
                racmo_data[var],
                label="RACMO",
                color="orange",
                linestyle="--",
                alpha=0.5,
            )
            plt.title(f"{var.capitalize()} Time Series at {site}")
            plt.xlabel("Time")
            plt.ylabel(var.capitalize())
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{site}_{var}_timeseries.png")
            plt.close()

            logging.info(f"Saved time series plot for {var} at {site} to {save_dir}/{site}_{var}_timeseries.png")
