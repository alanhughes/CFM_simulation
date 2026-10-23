import logging
import os
import json
from typing import Dict, List, Optional

import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod

from cfm_config import config

from .cache import check_date_coverage, forcing_csv_path, forcing_json_path, write_forcing_json
from .model_point import ModelPoint, model_point_at

logging.basicConfig(level=logging.INFO)


class ProcessBase(ABC):
    # name of the RCM, the config key of its column map and the unit changes of the forcing, for the JSON file of the
    # forcing cache - set by each subclass
    _rcm_name: str = None
    _map_name: str = None
    _unit_factors: Dict[str, str] = {}

    def __init__(self, borehole_lat: float, borehole_lon: float) -> None:
        """Base class for processing data at specified borehole coordinates. Subclasses implement a process method
        that writes the forcing CSV of the borehole location with process_fast (FEAT-003), and _read_data, which reads
        the full maps for MeltRACMO, MeltMAR and the spatial plots.

        Args:
            borehole_lat (float): Latitude of borehole location.
            borehole_lon (float): Longitude of borehole location.
        """

        self._borehole_lat: float = borehole_lat
        self._borehole_lon: float = borehole_lon

        self._x_idx: int = 0
        self._y_idx: int = 0
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
        """Write the forcing CSV and JSON file of the borehole location, with the h5py extraction (FEAT-003).

        The extraction reads the MAR files in one parallel pass, through the local cache around the sites. It stops
        with ForcingCoverageError if the files do not include all the years from start_year to end_year (BUG-003).
        """

        from .extract import process_fast

        process_fast([self])

    def _read_data(self) -> List[xr.Dataset]:
        """
        Read in the MAR .nc files from the specified data path in config.

        This reads the full maps. MeltMAR and the spatial plots use it. The forcing for a borehole does not use it,
        because process uses the h5py extraction (FEAT-003).

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
        """Write the forcing CSV and JSON file of the borehole location, with the h5py extraction (FEAT-003).

        The extraction reads the RACMO files in one parallel pass, through the local cache around the sites. It stops
        with ForcingCoverageError if the files do not include all the years from start_year to end_year (BUG-003).
        """

        from .extract import process_fast

        process_fast([self])

    def _read_data(self) -> List[xr.Dataset]:
        """Read in the RACMO .nc files from the specified data path in config, filtering for files which contain the
        variables we want to read and which contain a year between start and end year in config.

        This reads the full maps. MeltRACMO and the spatial plots use it. The forcing for a borehole does not use it,
        because process uses the h5py extraction (FEAT-003).

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
