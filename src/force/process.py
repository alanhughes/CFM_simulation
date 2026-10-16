import logging
import os
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod

from cfm_config import config

from .cache import check_date_coverage, forcing_csv_path, forcing_json_path, write_forcing_json
from .grid import nearest_cell
from .model_point import ModelPoint, model_point_at

logging.basicConfig(level=logging.INFO)


def _time_values(datasets: List[xr.Dataset], start_year: int, end_year: int) -> pd.DatetimeIndex:
    """The time coordinates of the RCM datasets in the year range. Reads the time coordinates only."""
    times = []
    for ds in datasets:
        name = next((name for name in ("time", "TIME") if name in ds.coords), None)
        if name is not None:
            times.append(pd.DatetimeIndex(ds[name].values))
    if not times:
        return pd.DatetimeIndex([])
    index = times[0].append(times[1:])
    return index[(index.year >= start_year) & (index.year <= end_year)]


def _source_files(datasets: List[xr.Dataset]) -> List[str]:
    """The files that the RCM datasets were read from, from the encoding of their variables."""
    sources = set()
    for ds in datasets:
        if "source" in ds.encoding:
            sources.add(ds.encoding["source"])
        for var in ds.variables.values():
            if "source" in var.encoding:
                sources.add(var.encoding["source"])
    return sorted(sources)


class ProcessBase(ABC):
    # name of the RCM, the config key of its column map and the unit changes that _xr_to_input_dataframe makes, for the
    # JSON file of the forcing cache - set by each subclass
    _rcm_name: str = None
    _map_name: str = None
    _unit_factors: Dict[str, str] = {}

    def __init__(self, borehole_lat: float, borehole_lon: float) -> None:
        """Base class for processing data at specified borehole coordinates. Subclasses should implement the process
        method to read in and process data to their relevant save directory.

        Args:
            borehole_lat (float): Latitude of borehole location.
            borehole_lon (float): Longitude of borehole location.
        """

        self._borehole_lat: float = borehole_lat
        self._borehole_lon: float = borehole_lon

        self._x_idx: int = 0
        self._y_idx: int = 0
        self._daily_xr: List[xr.Dataset] = None
        self._borehole_data: xr.Dataset = None
        self._model_point: Optional[ModelPoint] = None

    @property
    def model_point(self) -> ModelPoint:
        """The RCM grid cell of the borehole location (FEAT-002). Found when it is first used."""
        if self._model_point is None:
            self._model_point = model_point_at(self._borehole_lat, self._borehole_lon, self._rcm_name)
        return self._model_point

    @property
    def _save_path(self) -> str:
        """The forcing CSV of the model point in the forcing cache (FEAT-002)."""
        return forcing_csv_path(self.model_point)

    def _check_cell(self, y_idx: int, x_idx: int) -> None:
        """Make sure that the data comes from the cell that names the forcing CSV."""
        mp = self.model_point
        if (y_idx, x_idx) != (mp.i, mp.j):
            raise ValueError(
                f"{self._rcm_name}: the nearest cell in the data is ({y_idx}, {x_idx}), but the model point from "
                f"the coordinate file is ({mp.i}, {mp.j}). The files in the data path do not use one grid."
            )

    def process(self, save_path: str) -> None:
        """
        Process the RCM dataset to extract data at the borehole location, and save it with a JSON file that records
        its source. Stops with ForcingCoverageError if the data does not include all the years from start_year to
        end_year (BUG-003).
        """

        start_year, end_year = config["start_year"], config["end_year"]
        dataset = f"{self._rcm_name} ({self.model_point.release})"

        if os.path.exists(save_path):
            logging.info(f"Processed RCM data already exists at {save_path}. Skipping processing.")
            self._check_existing(save_path, start_year, end_year, dataset)

        else:
            logging.info(
                f"Model point {self.model_point.tag}: cell centre ({self.model_point.cell_lat:.4f}, "
                f"{self.model_point.cell_lon:.4f}), {self.model_point.distance_km:.1f} km from the borehole"
            )
            logging.info("Reading RCM data...")
            self._daily_xr = self._read_data()
            logging.info("...RCM data read successfully.")

            # stop before the long extraction if a year is not in the files (BUG-003)
            check_date_coverage(_time_values(self._daily_xr, start_year, end_year), start_year, end_year, dataset)

            logging.info("Processing borehole data...")
            borehole_dataframes = [self._xr_to_input_dataframe(xr_data) for xr_data in self._daily_xr]
            self._input_dataframe = pd.concat(borehole_dataframes)
            logging.info("...borehole data processed successfully.")

            self.save_forcing(save_path, self._input_dataframe, _source_files(self._daily_xr))

    def save_forcing(self, save_path: str, df: pd.DataFrame, source_files: List[str], extra: Optional[dict] = None):
        """Check the dates of the forcing data (BUG-003), then save it as a CSV with its JSON file."""
        start_year, end_year = config["start_year"], config["end_year"]
        dataset = f"{self._rcm_name} ({self.model_point.release})"
        check_date_coverage(df.index, start_year, end_year, dataset)

        # write to a temporary name first, so that a stopped run leaves no CSV that looks complete
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tmp_path = f"{save_path}.tmp"
        df.to_csv(tmp_path)
        json_path = write_forcing_json(
            tmp_path,
            self.model_point,
            df,
            source_files=source_files,
            column_map=config[self._map_name],
            unit_factors=self._unit_factors,
            start_year=start_year,
            end_year=end_year,
            site_lat=self._borehole_lat,
            site_lon=self._borehole_lon,
            extra=extra,
        )
        os.replace(json_path, forcing_json_path(save_path))
        os.replace(tmp_path, save_path)
        logging.info(f"Borehole RCM data saved to {save_path}")

    @staticmethod
    def _check_existing(save_path: str, start_year: int, end_year: int, dataset: str) -> None:
        """Check the dates of a forcing CSV in the cache: from its JSON file, or from the CSV if it has no JSON."""
        json_path = forcing_json_path(save_path)
        if os.path.exists(json_path):
            with open(json_path) as file:
                record = json.load(file)
            dates = pd.date_range(record["first_date"], record["last_date"], freq="D")
            check_date_coverage(dates, start_year, end_year, dataset)
        else:
            logging.warning(f"{save_path} has no JSON file, so its source is not known.")
            check_date_coverage(pd.read_csv(save_path, index_col=0).index, start_year, end_year, dataset)

    @abstractmethod
    def _read_data(self) -> List[xr.Dataset]:
        """Abstract method to read in data, to be implemented by subclasses."""
        pass

    @abstractmethod
    def _xr_to_input_dataframe(self, xr_data: xr.Dataset) -> pd.DataFrame:
        """Abstract method to convert xarray Dataset to pandas DataFrame, to be implemented by subclasses."""
        pass


class ProcessMAR(ProcessBase):
    def __init__(self, borehole_lat: float, borehole_lon: float) -> None:
        """Initialize with daily MAR dataset at specified borehole coordinates. Reads in .nc files from MAR data path
        specified in config.

        Args:
            borehole_lat (float): Latitude of borehole location.
            borehole_lon (float): Longitude of borehole location.
        """

        super().__init__(borehole_lat, borehole_lon)

        self._rcm_name: str = "MAR"
        self._map_name = "MAR_to_CFM_column_map"
        self._unit_factors = {"TSKIN": "+273.15 (degC to K)", "T2m": "+273.15 (degC to K)"}

    def process(self) -> None:
        """
        Process the MAR dataset to extract data at the borehole location.

        Returns:
            xr.Dataset: MAR data at the borehole location.
        """

        super().process(self._save_path)

    def _read_data(self) -> List[xr.Dataset]:
        """
        Read in the MAR .nc files from the specified data path in config.

        Returns:
            List[xr.Dataset]: List of xarray Datasets for each year of MAR data.
        """
        mar_data_path = config["MAR_data_path"]
        year_files = [
            f"{mar_data_path}/MARv3.14.3-27.5km-daily-ERA5-{year}.nc"
            for year in range(config["start_year"], config["end_year"] + 1)
        ]

        # Use on-disk chunk structure to avoid splitting stored chunks (better Dask performance).
        datasets = [xr.open_dataset(file, engine="netcdf4", chunks="auto") for file in year_files]

        return datasets

    def _xr_to_input_dataframe(self, xr_data: xr.Dataset) -> pd.DataFrame:
        """
        Extract daily data for borehole location, then convert the borehole MAR data to the pandas DataFrame needed for
        CFM input. Also only takes the sector specified in config.

        Args:
            xr_data (xr.Dataset): MAR data at the borehole location.
        Returns:
            pd.DataFrame: DataFrame containing MAR data at the borehole location, with columns named as required by CFM.
        """
        logging.info(f"Processing XR data at year {xr_data['TIME'].dt.year.values[0]}")

        # get indices of closest grid point to borehole
        y_idx, x_idx, _ = nearest_cell(xr_data["LAT"], xr_data["LON"], self._borehole_lat, self._borehole_lon)
        self._check_cell(y_idx, x_idx)

        # find coordinate names for X/Y (prefer MAR defaults, fallback to X/Y)
        x_coord_candidates = ["X18_215", "X"]
        y_coord_candidates = ["Y15_176", "Y"]
        x_coord = next((name for name in x_coord_candidates if name in xr_data.coords), None)
        y_coord = next((name for name in y_coord_candidates if name in xr_data.coords), None)
        if x_coord is None or y_coord is None:
            raise ValueError(f"Could not find appropriate X/Y coordinate names. Found coords: {list(xr_data.coords)}")
        logging.info(
            "Using coordinates: %s for x and %s for y for year %s",
            x_coord,
            y_coord,
            xr_data["TIME"].dt.year.values[0],
        )

        # find data at borehole location
        borehole_data = xr_data.isel(**{y_coord: y_idx, x_coord: x_idx})

        # drop everything not in correct sector
        borehole_data = borehole_data.sel(SECTOR=config["sector"])

        # convert to dataframe, filtering for valid data variables only
        mapping = config["MAR_to_CFM_column_map"]
        vars_to_include = list(mapping.keys())

        # drop any variables not in vars_to_include
        borehole_data = borehole_data.drop_vars([var for var in borehole_data.data_vars if var not in vars_to_include])

        # convert to dataframe
        borehole_df = borehole_data.to_dataframe().reset_index()

        # drop everything apart from outlay 0.0 (surface layer)
        borehole_df = borehole_df[borehole_df["OUTLAY"] == 0.0]

        # rename columns to match CFM input column names
        borehole_df.rename(columns=mapping, inplace=True)
        borehole_df.set_index("TIME", inplace=True)

        # drop unneeded columns - should just be coordinates now
        borehole_df = borehole_df[list(mapping.values())]

        # convert temperature to Kelvin
        borehole_df["TSKIN"] = borehole_df["TSKIN"] + 273.15
        borehole_df["T2m"] = borehole_df["T2m"] + 273.15

        # remove duplicate times
        borehole_df = borehole_df[~borehole_df.index.duplicated(keep="first")]

        return borehole_df


class ProcessRACMO(ProcessBase):
    def __init__(self, borehole_lat: float, borehole_lon: float) -> None:
        """Initialize with daily RACMO dataset at specified borehole coordinates. Reads in .nc files from RACMO data
        path
        specified in config.

        Args:
            borehole_lat (float): Latitude of borehole location.
            borehole_lon (float): Longitude of borehole location.
        """

        super().__init__(borehole_lat, borehole_lon)

        self._rcm_name: str = "RACMO"
        self._var_to_read = list(config["RACMO_to_CFM_column_map"].keys())
        self._var_to_read += config["RACMO_additional_vars"]
        self._map_name = "RACMO_to_CFM_column_map"
        self._unit_factors = {
            "SUBLIM": "*86400 (kg m-2 s-1 to mm w.e. day-1)",
            "RAIN": "*86400 (kg m-2 s-1 to mm w.e. day-1); RAIN = pr - snowfall, pr and snowfall clipped at 0",
            "BDOT": "*86400 (kg m-2 s-1 to mm w.e. day-1); clipped at 0",
            "SMELT": "*86400 (kg m-2 s-1 to mm w.e. day-1); clipped at 0",
            "ALBEDO": "rsusgl / rsds",
        }

        self._snowfall_name = "prsn"  # this changes between RACMO versions

    def process(self) -> None:
        """
        Process the RACMO dataset to extract data at the borehole location.

        Returns:
            xr.Dataset: RACMO data at the borehole location.
        """

        if config.get("forcing_fast_extraction", False):
            # FEAT-003: h5py extraction in one parallel pass, through the local cache around the sites
            from .extract import process_fast

            process_fast([self])
            return

        super().process(self._save_path)

    def _read_data(self) -> List[xr.Dataset]:
        """Read in the RACMO .nc files from the specified data path in config, filtering for files which contain the
        variables we want to read and which contain a year between start and end year in config.

        Returns:
            List[xr.Dataset]: List of xarray Datasets for each batch of years of RACMO data
        """

        racmo_data_path = config["RACMO_data_path"]
        all_files = os.listdir(racmo_data_path)
        datasets = []

        for year in range(config["start_year"], config["end_year"] + 1):
            year_files = [f"{config['RACMO_data_path']}/{file}" for file in all_files if f"{year}0101-" in file]
            if year_files:
                datasets.append(self._read_data_by_year_all_vars(year_files))

        return datasets

    def _read_data_by_year_all_vars(self, year_files: List[str]) -> xr.Dataset:
        """
        For a given year, read all RACMO files containing any of the variables we want to read, then merge these into a
        single xarray Dataset for that year.

        Args:
            year_files (List[str]): List of all files in the RACMO data directory with given year in name.

        Returns:
            xr.Dataset: Merged xarray Dataset containing all variables we want to read for the given year.

        """

        var_files = []
        for var in self._var_to_read:
            var_file = next((file for file in year_files if f"{var}_" in file), None)
            if var_file is not None:
                var_files.append(var_file)
        # Use on-disk chunk structure to avoid splitting stored chunks (better Dask performance).
        datasets = [xr.open_dataset(file, engine="netcdf4", chunks="auto") for file in var_files]
        merged_dataset = xr.merge(datasets)

        return merged_dataset

    def _xr_to_input_dataframe(self, xr_data: xr.Dataset) -> pd.DataFrame:
        """Takes xarray Data for a given batch of years of RACMO data

        Args:
            xr_data (xr.Dataset): RACMO data for a given batch of years, containing all variables we want to read.
        """

        # get indices of closest grid point to borehole
        y_idx, x_idx, _ = nearest_cell(xr_data["lat"], xr_data["lon"], self._borehole_lat, self._borehole_lon)
        self._check_cell(y_idx, x_idx)

        # find data at borehole location
        # can use any coordinate as the indexing is the same for all
        borehole_data = xr_data.isel(rlat=y_idx, rlon=x_idx)

        # drop anything not in var_to_read
        borehole_data = borehole_data.drop_vars(
            [var for var in borehole_data.data_vars if var not in self._var_to_read]
        )

        # convert to dataframe
        borehole_df = borehole_data.to_dataframe().reset_index()

        # ensure mass fluxes are positive
        borehole_df["pr"] = np.clip(borehole_df["pr"], a_min=0, a_max=None)
        borehole_df[self._snowfall_name] = np.clip(borehole_df[self._snowfall_name], a_min=0, a_max=None)
        borehole_df["mltgl"] = np.clip(borehole_df["mltgl"], a_min=0, a_max=None)

        # get rf and alb
        borehole_df["rf"] = borehole_df["pr"] - borehole_df[self._snowfall_name]
        borehole_df["alb"] = borehole_df["rsusgl"] / borehole_df["rsds"]

        # rename columns to match CFM input column names
        mapping = config[self._map_name]
        borehole_df.rename(columns=mapping, inplace=True)
        borehole_df.set_index("time", inplace=True)

        # drop unneeded columns
        borehole_df = borehole_df[list(mapping.values())]

        # only keep dates in the time range
        borehole_df = borehole_df[
            (borehole_df.index.year >= config["start_year"]) & (borehole_df.index.year <= config["end_year"])
        ]

        # multiply all mass fluxes by 60*60*24 to convert from kg/m2/s to mmWE/day
        # don't include melt here because it seems to already be in daily totals
        mass_flux_columns = ["SUBLIM", "RAIN", "BDOT", "SMELT"]

        for col in mass_flux_columns:
            if col in borehole_df.columns:
                borehole_df[col] = borehole_df[col] * 60 * 60 * 24

        # make sure rain is positive
        borehole_df["RAIN"] = np.clip(borehole_df["RAIN"], a_min=0, a_max=None)

        # remove duplicate times
        borehole_df = borehole_df[~borehole_df.index.duplicated(keep="first")]

        return borehole_df


class ProcessRACMO23(ProcessRACMO):
    def __init__(self, borehole_lat: float, borehole_lon: float) -> None:
        """Initialize with daily RACMO 2.3 dataset at specified borehole coordinates. Reads in .nc files from higher
        resolution RACMO 2.3 data path specified in config.

        Args:
            borehole_lat (float): Latitude of borehole location.
            borehole_lon (float): Longitude of borehole location.
        """

        super().__init__(borehole_lat, borehole_lon)

        self._rcm_name: str = "RACMO23"
        self._var_to_read = list(config["RACMO23_to_CFM_column_map"].keys())
        self._var_to_read += config["RACMO_additional_vars"]

        self._snowfall_name = "sf"
        self._map_name = "RACMO23_to_CFM_column_map"

    def _read_data(self) -> List[xr.Dataset]:
        """Read in the RACMO2.3 .nc files from the specified data path in config, filtering for files which contain the
        variables we want to read and which contain a year between start and end year in config.

        Returns:
            List[xr.Dataset]: List of xarray Datasets for each batch of years of RACMO data
        """

        racmo_data_path = config["RACMO23_data_path"]
        all_files = os.listdir(racmo_data_path)
        datasets = []

        for year in range(config["start_year"], config["end_year"] + 1):
            year_files = [f"{config['RACMO23_data_path']}/{file}" for file in all_files if f"{year}" in file]
            if year_files:
                datasets.append(self._read_data_by_year_all_vars(year_files))

        return datasets

    def _read_data_by_year_all_vars(self, year_files: List[str]) -> xr.Dataset:
        """
        For a given year, read all RACMO files containing any of the variables we want to read, then merge these into a
        single xarray Dataset for that year.

        Args:
            year_files (List[str]): List of all files in the RACMO2.3 data directory with given year in name.

        Returns:
            xr.Dataset: Merged xarray Dataset containing all variables we want to read for the given year.

        """

        var_files = []
        for var in self._var_to_read:
            var_file = next((file for file in year_files if f"{var}." in file), None)
            if var_file is not None:
                var_files.append(var_file)
        # Use on-disk chunk structure to avoid splitting stored chunks (better Dask performance).
        datasets = [xr.open_dataset(file, chunks="auto") for file in var_files]
        merged_dataset = xr.merge(datasets, compat="override")

        return merged_dataset
