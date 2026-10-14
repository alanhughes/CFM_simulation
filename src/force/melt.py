"""Classes which find the average yearly total melt across the AIS, for specified RCM"""

import logging

import xarray as xr
import os

from cfm_config import config
from .process import ProcessMAR, ProcessRACMO

logging.basicConfig(level=logging.INFO)


class MeltMAR(ProcessMAR):
    """Class to find the average yearly total melt across the AIS for MAR

    Inherits from ProcessMAR so that it can use the _read_data method.
    """

    def __init__(self, borehole_lat: float = 0.0, borehole_lon: float = 0.0) -> None:
        """Initialise with generic borehole_lat and borehole_lon. This class calculates melt over entire AIS, so doesn't need borehole location.

        Args:
            borehole_lat (float, optional): borehole location latitude (not needed). Defaults to 0.0 as a placeholder.
            borehole_lon (float, optional): borehole location longitude (not needed). Defaults to 0.0 as a placeholder.
        """
        super().__init__(borehole_lat, borehole_lon)

        self._save_location: str = f"{config['CFM_data_path']}/melt"
        self._file_name: str = f"mar_avg_yearly_melt_map_{config['start_year']}-{config['end_year']}.nc"
        os.makedirs(self._save_location, exist_ok=True)

    def get_melt_map(self) -> None:
        """Get the average yearly total melt across the AIS for MAR. This method will load in the MAR data for the entire
        AIS, calculate the yearly sums of melt, and then average these yearly sums to get a single spatial map of how 'wet' firn is across the AIS.
        """

        logging.info("Reading in MAR data for the entire AIS over the entire time period...")
        # Load the MAR data for the entire AIS over the entire time period
        mar_ds = self._read_data()
        logging.info("MAR data loaded successfully.")

        logging.info("Calculating the average yearly total melt across the AIS...")
        # joint list of xrarray together
        mar_ds = xr.concat(mar_ds, dim="TIME")

        # Calculate the yearly sums of melt
        yearly_melt = mar_ds["ME"].groupby("TIME.year").sum(dim="TIME")

        # Average these yearly sums to get a single spatial map of how 'wet' firn is across the AIS
        avg_yearly_melt = yearly_melt.mean(dim="year")

        # add back in LON LAT variables to the avg_yearly_melt DataArray, as these are needed for plotting later
        # only use LON LAT variables from a single time step, as these don't change over time
        avg_yearly_melt = avg_yearly_melt.assign_coords(LON=mar_ds["LON"].isel(TIME=0), LAT=mar_ds["LAT"].isel(TIME=0))

        # Compute the data to force evaluation before saving
        avg_yearly_melt = avg_yearly_melt.compute()
        logging.info("Average yearly total melt calculated successfully.")

        # save the average yearly melt map to a netcdf file
        avg_yearly_melt.to_netcdf(self._save_location + "/" + self._file_name, engine="netcdf4", mode="w")
        logging.info(f"Average yearly total melt saved to {self._save_location}/{self._file_name}")


class MeltRACMO(ProcessRACMO):
    """Class to find the average yearly total melt across the AIS for RACMO

    Inherits from ProcessRACMO so that it can use the _read_data method.
    """

    def __init__(self, borehole_lat: float = 0.0, borehole_lon: float = 0.0) -> None:
        """Initialise with generic borehole_lat and borehole_lon. This class calculates melt over entire AIS, so doesn't need borehole location.

        Args:
            borehole_lat (float, optional): borehole location latitude (not needed). Defaults to 0.0 as a placeholder.
            borehole_lon (float, optional): borehole location longitude (not needed). Defaults to 0.0 as a placeholder.
        """
        super().__init__(borehole_lat, borehole_lon)

        self._save_location: str = f"{config['CFM_data_path']}/melt"
        self._file_name: str = f"racmo_avg_yearly_melt_map_{config['start_year']}-{config['end_year']}.nc"
        os.makedirs(self._save_location, exist_ok=True)

    def get_melt_map(self) -> None:
        """Get the average yearly total melt across the AIS for RACMO.

        This method will load in the RACMO data for the entire AIS, calculate
        the yearly sums of melt, and then average these yearly sums to get a
        single spatial map of how 'wet' firn is across the AIS.
        """

        logging.info("Reading in RACMO data for the entire AIS over the entire time period...")
        # Load the RACMO data for the entire AIS over the entire time period
        racmo_ds = self._read_data()
        logging.info("RACMO data loaded successfully.")

        logging.info("Calculating the average yearly total melt across the AIS...")
        # joint list of xrarray together
        racmo_ds = xr.concat(racmo_ds, dim="time")

        # Calculate the yearly sums of melt
        yearly_melt = racmo_ds["mltgl"].groupby("time.year").sum(dim="time")

        # Average these yearly sums to get a single spatial map of how 'wet' firn is across the AIS
        avg_yearly_melt = yearly_melt.mean(dim="year")

        # multiply by 60*60*24 to account for the fact that original units of melt in RACMO are kg/m2/s, and we want to convert to kg/m2/day
        avg_yearly_melt = avg_yearly_melt * 60 * 60 * 24

        # Compute the data to force evaluation before saving
        avg_yearly_melt = avg_yearly_melt.compute()
        logging.info("Average yearly total melt calculated successfully.")

        # save the average yearly melt map to a netcdf file
        avg_yearly_melt.to_netcdf(self._save_location + "/" + self._file_name, engine="netcdf4", mode="w")
        logging.info(f"Average yearly total melt saved to {self._save_location}/{self._file_name}")
