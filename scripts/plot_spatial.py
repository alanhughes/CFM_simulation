"""Script to plot spatial maps of variables from the RCM simulations, typically restricted to AIS peninsula.
The bbox for this restriction can be set in config.yaml.

Usage:
    - plot_spatial.py
"""

from plot import SpatialPlotter
import logging

from cfm_config import config

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    mar_variables = config["MAR_to_CFM_column_map"].keys()
    racmo_variables = config["RACMO_to_CFM_column_map"].keys()
    # rm 'rf' and 'alb' from racmo variables, as these are calculated from other variables and not directly available in the RACMO output
    racmo_variables = [var for var in racmo_variables if var not in ["rf", "alb"]]

    # faster to plot each variable separately, as this avoids having to load in all the data for all variables at once (which can be very large and cause memory issues)
    for var in racmo_variables:
        logging.info(f"Plotting RACMO variable: {var}")
        SpatialPlotter(rcm_name="RACMO", variables=[var], plot_type="mean").plot()
    for var in mar_variables:
        logging.info(f"Plotting MAR variable: {var}")
        SpatialPlotter(rcm_name="MAR", variables=[var], plot_type="mean").plot()
