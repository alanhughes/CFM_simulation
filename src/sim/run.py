import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from typing import Literal

from cfm_config import cfm_output_folder, config, run_id

logging.basicConfig(level=logging.INFO)

sys.path.append(config["CFM_repo_path"])
from RCMpkl_to_spin import makeSpinFiles  # noqa: E402 # type: ignore
from firn_density_nospin import FirnDensityNoSpin  # noqa: E402 # type: ignore


class CFMRun:
    def __init__(
        self,
        borehole_lat: float,
        borehole_lon: float,
        physRho: str,
        rcm_name: Literal["MAR", "RACMO"],
        liquid: Literal["bucket"],
    ) -> None:
        """Initialize CFM Run with configuration from config.yaml.

        Sets up paths and variables needed for the CFM run. Including:
        - Forcing data path from which to read MAR or RACMO data
        - Output path to save CFM results
        - JSON configuration name saving of CFM configuration

        Args:
            borehole_lat (float): Latitude of borehole location.
            borehole_lon (float): Longitude of borehole location.
            physRho (str): Physical densification scheme to use in CFM model.
            rcm_name (Literal["MAR", "RACMO"]): RCM name to use for forcing data (e.g., MAR or RACMO).
            liquid (Literal["bucket"]): meltwater scheme to use (currently only bucket)
        """

        # all these need to be passed in because they vary by run
        self._rcm_name: Literal["MAR", "RACMO"] = rcm_name
        self._cfm_input_path: str = f"{config['CFM_data_path']}/cfm_input"
        self._cfm_output_path: str = f"{config['CFM_data_path']}/cfm_output"
        self._borehole_lat: float = borehole_lat
        self._borehole_lon: float = borehole_lon

        # load cfm config
        self._cfm_config: dict = config["cfm_config"]

        # set physRho and borehole loc from argument in case we've used command line to override config
        self._cfm_config["physRho"] = physRho
        self._force_data: pd.DataFrame = None

        # also set the liquid scheme in the config
        self._cfm_config["liquid"] = liquid

        # set json config name
        self._json_config_name = f"CFMconfig_{run_id(borehole_lat, borehole_lon, physRho, liquid, rcm_name)}.json"

        # flag to indicate if output folder exists
        self._output_exists: bool = False

        # set output folder in config
        self._cfm_config["resultsFolder"] = cfm_output_folder(borehole_lat, borehole_lon, physRho, liquid, rcm_name)

        if not os.path.exists(self._cfm_config["resultsFolder"]):
            os.makedirs(self._cfm_config["resultsFolder"])
            logging.info(f"CFM output will be saved to {self._cfm_config['resultsFolder']}.")
        else:
            logging.info(
                f"CFM output path exists at {self._cfm_config['resultsFolder']}. Delete this folder to rerun simulation."
            )
            self._output_exists = True

    def run(self) -> None:
        """Run the CFM model with the specified configuration and forcing data."""

        if not self._output_exists:
            self._read_force_data()
            self._run_cfm()

    def _read_force_data(self) -> None:
        """
        Read in the forcing data from the MAR data CSV file.
        Raises FileNotFoundError if the file does not exist.
        """

        # Read in forcing data
        force_data_path = f"{self._cfm_input_path}/{self._rcm_name}_{self._borehole_lat}_{self._borehole_lon}_{config['start_year']}_{config['end_year']}.csv"
        if not os.path.exists(force_data_path):
            raise FileNotFoundError(
                f"Forcing data file not found: {force_data_path}. Run read_force_data.py to generate the file."
            )
        self._force_data = pd.read_csv(force_data_path, index_col=0, parse_dates=True)
        logging.info(f"Forcing data read from {force_data_path}.")

    def _run_cfm(self) -> None:
        """Run the CFM model using the forcing data and configuration.
        This code is mostly copied from main.py in the CFM repository.
        """

        # format the CFM forcing data (including creating the spin up)
        # climateTS is a dictionary with the various climate fields needed, in the correct units.
        climateTS, StpsPerYr, depth_S1, depth_S2, grid_bottom, SEBfluxes = makeSpinFiles(
            self._force_data,
            timeres=self._cfm_config["DFresample"],
            Tinterp="mean",
            spin_date_st=config["start_year"],
            spin_date_end=config["spin_end_year"],
            melt=self._cfm_config["MELT"],
            desired_depth=None,
            SEB=self._cfm_config["SEB"],
            rho_bottom=config["rho_bottom"],
        )
        climateTS["SUBLIM"] = (
            config["sublim_sf"] * climateTS["SUBLIM"]
        )  # ADDED THIS FOR MERRA2 TO GET THE SIGN CORRECT.
        climateTS["forcing_data_start"] = config["start_year"]

        # unsure why these are redefined here, but keeping consistent with previous code
        self._cfm_config["stpsPerYear"] = float("%.2f" % (StpsPerYr))
        self._cfm_config["stpsPerYearSpin"] = float("%.2f" % (StpsPerYr))
        self._cfm_config["grid1bottom"] = float("%.1f" % (depth_S1))
        self._cfm_config["grid2bottom"] = float("%.1f" % (depth_S2))
        self._cfm_config["HbaseSpin"] = float("%.1f" % (3000 - grid_bottom))
        self._cfm_config["DIPhorizon"] = np.floor(0.8 * grid_bottom)  # firn air content, depth integrated porosity
        self._cfm_config["keep_firnthickness"] = True
        self._cfm_config["grid_outputs"] = True

        # write updated config to json
        with open(
            f"{self._cfm_output_path}/{self._json_config_name}",
            "w",
        ) as fp:
            fp.write(json.dumps(self._cfm_config, sort_keys=False, indent=4, separators=(",", ": ")))

        ### Create CFM instance by passing config file and forcing data, then run the model
        firn = FirnDensityNoSpin(
            f"{self._cfm_output_path}/{self._json_config_name}",
            climateTS=climateTS,
            NewSpin=self._cfm_config["NewSpin"],
            SEBfluxes=SEBfluxes,
        )
        firn.time_evolve()
