import logging
import os
from typing import List, Literal

import matplotlib.pyplot as plt
import xarray as xr

from force import ProcessMAR, ProcessRACMO

from cfm_config import config


class SpatialPlotter:
    """Class to plot spatial maps of the average yearly melt across AIS peninsula for MAR and RACMO"""

    def __init__(self, rcm_name: Literal["MAR", "RACMO"], variables: List[str], plot_type: str) -> None:
        """Initializes the PlotSpatial class.

        Args:
            rcm_name (Literal["MAR", "RACMO"]): name of the RCM to plot melt for - either 'MAR' or 'RACMO'
            variables (List[str]): list of variable names to plot
            plot_type (str): type of plot to create (e.g., "mean", "max", "min", or a time point)
        """

        self.rcm_name = rcm_name
        self.variables = variables

        self.processor = {"MAR": ProcessMAR, "RACMO": ProcessRACMO}[rcm_name]
        self.time_name: str = {"MAR": "TIME", "RACMO": "time"}[rcm_name]
        self.lat_name: str = {"MAR": "LAT", "RACMO": "lat"}[rcm_name]
        self.lon_name: str = {"MAR": "LON", "RACMO": "lon"}[rcm_name]
        self.plot_type: str = plot_type

        self.save_dir: str = f"{config['CFM_data_path']}/cfm_figures/spatial_maps"
        os.makedirs(self.save_dir, exist_ok=True)

    def plot(self) -> None:
        """Plots the spatial map of the specified variable(s) for the specified RCM and plot type. This method will
        read in the data, aggregate it according to the specified plot type, and then plot the spatial map.
        The resulting plot will be saved to the specified directory.
        """

        logging.info("Reading in data...")
        self._read_data()
        logging.info("...Data read.")

        logging.info("Aggregating data...")
        self._aggregate_data()
        logging.info("...Data aggregated.")

        logging.info("Plotting spatial map...")
        self._plot_spatial_map()
        logging.info("...All spatial maps plotted.")

    def _read_data(self) -> None:
        """Reads in the data for the specified RCM and variables, and concatenates it into a single xarray Dataset."""

        # initialise the processor class to read in the data
        # can use generic borehole location, as this class is just for plotting spatial maps of melt across the AIS
        # peninsula, so doesn't need specific borehole location
        processor = self.processor(borehole_lat=0.0, borehole_lon=0.0)

        # use the processer to read in data
        if self.rcm_name == "RACMO":
            processor._var_to_read = self.variables  # do this to speed up reading for racmo data

        ds = processor._read_data()
        ds = xr.concat(ds, dim=self.time_name)

        # save lat lon variables for later
        if self.rcm_name == "MAR":
            ds = ds.sel(OUTLAY=0.0)
            ds = ds.sel(SECTOR=1.0)
            self.lat_values, self.lon_values = (
                ds[self.lat_name].isel({self.time_name: 0}),
                ds[self.lon_name].isel({self.time_name: 0}),
            )
        elif (
            self.rcm_name == "RACMO"
        ):  # racmo doesnt store lat lon as variables, but as coordinates so have no time dim
            self.lat_values, self.lon_values = ds[self.lat_name], ds[self.lon_name]

        # only keep variables that are needed
        self.ds = ds[self.variables]

        self._convert_units()
        self.ds.compute()

    def _convert_units(self) -> None:
        """Converts smb variables in RACMO data from kg/m2/s to mmWE/day, to match the units of the MAR data.

        Converts temperature variables in MAR data from C to K, to match the units of the RACMO data.

        This method should be called after reading in the data, but before aggregating it.
        """

        if self.rcm_name == "RACMO":
            # convert smb variables from kg/m2/s to mmWE/day
            for var in self.variables:
                if var in config["RACMO_var_to_multiply"]:
                    self.ds[var] = self.ds[var] * config["RACMO_to_MAR_smb_factor"]
                    self.ds[var].attrs["units"] = "mmWE/day"  # update units attribute to reflect conversion

        elif self.rcm_name == "MAR":
            # convert temperature variables from C to K
            for var in self.variables:
                if var in config["MAR_temp_vars"]:
                    self.ds[var] = self.ds[var] + 273.15  # convert from C to K
                    self.ds[var].attrs["units"] = "K"  # update units attribute to reflect conversion

    def _aggregate_data(self) -> None:
        """Aggregates the data according to the specified plot type (e.g., "mean", "max", "min", or a time point)."""

        if self.plot_type == "mean":
            self.ds = self.ds.mean(dim=self.time_name, keep_attrs=True)
        elif self.plot_type == "max":
            self.ds = self.ds.max(dim=self.time_name, keep_attrs=True)
        elif self.plot_type == "min":
            self.ds = self.ds.min(dim=self.time_name, keep_attrs=True)
        else:
            # assume plot_type is a time point, so select the data for that time point
            self.ds = self.ds.sel({self.time_name: self.plot_type}, keep_attrs=True)

        if self.rcm_name == "MAR":
            # reassign lat lon to MAR data, as these are needed for plotting later
            self.ds = self.ds.assign_coords(LON=self.lon_values, LAT=self.lat_values)

    def _plot_spatial_map(self) -> None:
        """Plots the spatial map of the specified variable(s) for the specified RCM and plot type."""

        # restrict to larsenC_box
        self.ds = self.ds.where(
            (self.ds[self.lat_name] >= config["larsenC_box"]["lat_min"])
            & (self.ds[self.lat_name] <= config["larsenC_box"]["lat_max"])
            & (self.ds[self.lon_name] >= config["larsenC_box"]["lon_min"])
            & (self.ds[self.lon_name] <= config["larsenC_box"]["lon_max"]),
        )

        for var in self.variables:
            plt.figure(figsize=(10, 8))
            self.ds[var].plot(
                x=self.lon_name,
                y=self.lat_name,
                cmap="Blues",
                vmin=config["rcm_plot_cbar_limits"][var][0],
                vmax=config["rcm_plot_cbar_limits"][var][1],
            )
            plt.xlim(config["larsenC_box"]["lon_min"], config["larsenC_box"]["lon_max"])
            plt.ylim(config["larsenC_box"]["lat_min"], config["larsenC_box"]["lat_max"])
            plt.title(f"{self.rcm_name} {var} ({self.plot_type})")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.savefig(f"{self.save_dir}/{self.rcm_name}_{var}_{self.plot_type}.png")
            plt.close()

            logging.info(
                f"Spatial map for {self.rcm_name} {var} ({self.plot_type}) saved to "
                f"{self.save_dir}/{self.rcm_name}_{var}_{self.plot_type}.png"
            )
