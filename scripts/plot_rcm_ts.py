"""
Script to generate plots of time series for MAR and RACMO forcing data at specified borehole sites. The script will:
- Load the MAR and RACMO forcing data for each borehole site from the processed csv
- Generate time series plots of the forcing variables (e.g., temperature, accumulation) for each site and each RCM.
The plots will show the temporal variability in the forcing data at the borehole locations.
- Overlay the time series for MAR and RACMO on the same plot for each variable to compare the two RCMs at each site.
The resulting plots will be saved to the specified figure path.

The forcing CSV of each RCM has the years of scripts/run_cfm.py for that RCM (force.years.run_years, DISC-001): the
MAR CSVs are 1979-2024 and the RACMO CSVs 1979-2023, if end_year in config.yaml is 2024. --start-year and --end-year
give the years of both, as they are.

Usage:
    python plot_rcm_ts.py [--start-year <year>] [--end-year <year>]

"""

import argparse
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd

from cfm_config import config, sites
from force.cache import forcing_csv_path
from force.model_point import model_point, model_point_label, shared_model_points
from force.years import add_year_arguments, run_years

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the MAR and RACMO forcing time series at each site")
    add_year_arguments(parser)
    args = parser.parse_args()
    years = {rcm: run_years(rcm, args.start_year, args.end_year) for rcm in ("MAR", "RACMO")}

    borehole_sites = sites()
    mar_data_dict: dict[str, pd.DataFrame] = {}
    racmo_data_dict: dict[str, pd.DataFrame] = {}

    for site, info in borehole_sites.items():
        # Load MAR and RACMO data for the model point of this site from the forcing cache
        mar = years["MAR"]
        racmo = years["RACMO"]
        mar_data_dict[site] = pd.read_csv(forcing_csv_path(model_point(info, "MAR"), mar.start_year, mar.end_year))
        racmo_data_dict[site] = pd.read_csv(
            forcing_csv_path(model_point(info, "RACMO"), racmo.start_year, racmo.end_year)
        )

    # model points of each site; also the inactive sites, to show all the sites that share a grid cell (FEAT-002)
    all_sites = sites(active_only=False)
    model_points = {rcm: shared_model_points(all_sites, rcm) for rcm in ("MAR", "RACMO")}

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
            plt.title(
                f"{var.capitalize()} Time Series at {site}\n"
                f"MAR {model_point_label(site, model_points['MAR'])}\n"
                f"RACMO {model_point_label(site, model_points['RACMO'])}",
                fontsize=9,
            )
            plt.xlabel("Time")
            plt.ylabel(var.capitalize())
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{site}_{var}_timeseries.png")
            plt.close()

            logging.info(f"Saved time series plot for {var} at {site} to {save_dir}/{site}_{var}_timeseries.png")
