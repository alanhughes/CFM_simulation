import json
import logging
import os
import socket
import sys
import time
import traceback

import numpy as np
import pandas as pd
from typing import Literal

from cfm_config import cfm_output_folder, config, run_id
from cfm_config import cfm_version as lookup_cfm_version
from force.cache import forcing_csv_path
from force.model_point import model_point_at
from sim.identity import check_cfm_identity, measure_cfm_identity, wrapper_identity
from sim.manifest import COMPLETE, FAILED, RUNNING, is_alive, now, read_manifest, write_manifest

logging.basicConfig(level=logging.INFO)


def import_cfm(repo: str) -> tuple:
    """Import makeSpinFiles and FirnDensityNoSpin from the CFM in repo.

    The CFM's modules are flat top-level ones, so one process can only import one CFM. scripts/run_cfm.py starts each
    run in its own process (sim.worker) for this reason.
    """
    loaded = sys.modules.get("firn_density_nospin")
    if loaded is not None and os.path.realpath(os.path.dirname(loaded.__file__)) != os.path.realpath(repo):
        raise RuntimeError(f"This process already imported the CFM from {os.path.dirname(loaded.__file__)}, not {repo}.")
    sys.path.append(repo)
    from RCMpkl_to_spin import makeSpinFiles  # type: ignore
    from firn_density_nospin import FirnDensityNoSpin  # type: ignore

    return makeSpinFiles, FirnDensityNoSpin


class CFMRun:
    def __init__(
        self,
        borehole_lat: float,
        borehole_lon: float,
        physRho: str,
        rcm_name: Literal["MAR", "RACMO"],
        liquid: Literal["bucket"],
        cfm_version: dict | None = None,
        force: bool = False,
        site: str | None = None,
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
            cfm_version (dict | None): CFM version from cfm_config.cfm_version(), default_cfm_version if None. This
                process must be running that version's python.
            force (bool): rerun even if a complete run exists in the output folder.
            site (str | None): name in sites.yaml of the site at borehole_lat/borehole_lon, for the manifest.
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
        self._force_data_path: str = None

        # also set the liquid scheme in the config
        self._cfm_config["liquid"] = liquid

        # set json config name
        self._json_config_name = f"CFMconfig_{run_id(borehole_lat, borehole_lon, physRho, liquid, rcm_name)}.json"

        # set output folder in config. The folder is made in run(), once the forcing data has been read
        self._cfm_config["resultsFolder"] = cfm_output_folder(borehole_lat, borehole_lon, physRho, liquid, rcm_name)
        self._force: bool = force
        self._site: str | None = site
        self._cfm_version: dict = cfm_version or lookup_cfm_version()
        self._manifest: dict = {}

    def run(self) -> None:
        """Run the CFM model with the specified configuration and forcing data, unless a complete run exists.

        The run's state is kept in manifest.json in the output folder: "running" while the CFM runs, then "complete"
        or "failed". Only a "complete" run is skipped, and --force reruns even that.
        """

        results_folder = self._cfm_config["resultsFolder"]
        if not self._should_run(results_folder):
            return

        if os.path.realpath(sys.executable) != os.path.realpath(self._cfm_version["python"]):
            raise RuntimeError(
                f"CFM version {self._cfm_version['label']!r} runs with {self._cfm_version['python']}, "
                f"but this is {sys.executable}. Use scripts/run_cfm.py, which starts the right one."
            )
        cfm_identity = measure_cfm_identity(self._cfm_version)
        check_cfm_identity(cfm_identity)

        # read forcing before making the folder, so a missing CSV leaves nothing behind
        self._read_force_data()
        os.makedirs(results_folder, exist_ok=True)

        # run.log also stops the CFM deleting the folder at spin-up: it rmtree's resultsFolder if every file in it is
        # .json or .hdf5 (firn_density_spin.py), which would take manifest.json with it. With run.log present it only
        # deletes the .hdf5 files.
        log_handler = logging.FileHandler(os.path.join(results_folder, "run.log"))
        log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(log_handler)

        self._manifest = self._new_manifest(cfm_identity)
        write_manifest(results_folder, self._manifest)
        logging.info(f"CFM output will be saved to {results_folder}.")

        start = time.monotonic()
        try:
            self._run_cfm()
        except BaseException as error:  # also KeyboardInterrupt, so a ctrl-C run is marked failed, not running
            self._manifest.update(
                status=FAILED,
                finished=now(),
                runtime_s=round(time.monotonic() - start, 1),
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            )
            write_manifest(results_folder, self._manifest)
            logging.error(f"CFM run failed, see {results_folder}/manifest.json")
            raise
        else:
            self._manifest.update(status=COMPLETE, finished=now(), runtime_s=round(time.monotonic() - start, 1))
            write_manifest(results_folder, self._manifest)
            logging.info(f"CFM run complete in {self._manifest['runtime_s']} s.")
        finally:
            logging.getLogger().removeHandler(log_handler)
            log_handler.close()

    def _should_run(self, results_folder: str) -> bool:
        """Whether to run the CFM, from the manifest (if any) of an earlier run in results_folder."""

        if not os.path.exists(results_folder):
            return True

        manifest = read_manifest(results_folder)

        if manifest is None:
            # made before manifests existed. Keep the old skip only if it holds results, as that is the best we can do
            if os.path.exists(os.path.join(results_folder, self._cfm_config["resultsFileName"])) and not self._force:
                logging.warning(
                    f"{results_folder} has results but no manifest, so it is assumed complete. "
                    "Use --force to rerun it."
                )
                return False
            logging.info(f"{results_folder} has no manifest and no results. Running again.")
            return True

        status = manifest["status"]
        if self._force:
            logging.info(f"--force: rerunning {results_folder} (status was {status}).")
            if status == RUNNING and is_alive(manifest):
                raise RuntimeError(f"{results_folder} is being run by pid {manifest['pid']} on {manifest['host']}.")
            return True

        if status == COMPLETE:
            logging.info(f"Complete run found at {results_folder}, skipping. Use --force to rerun.")
            return False
        if status == RUNNING:
            alive = is_alive(manifest)
            if alive is None:
                raise RuntimeError(
                    f"{results_folder} is marked running on {manifest['host']} (pid {manifest['pid']}). "
                    "If that run is dead, use --force."
                )
            if alive:
                raise RuntimeError(f"{results_folder} is being run by pid {manifest['pid']} on this host.")
            logging.warning(f"{results_folder} was left running by a dead process (pid {manifest['pid']}). Rerunning.")
            return True
        logging.warning(f"{results_folder} has status {status} ({manifest.get('error')}). Rerunning.")
        return True

    def _new_manifest(self, cfm_identity: dict) -> dict:
        """Manifest for a run that is about to start. The final cfm_config is added once _run_cfm has made it."""
        return {
            "status": RUNNING,
            "run": {
                "site": self._site,
                "borehole_lat": self._borehole_lat,
                "borehole_lon": self._borehole_lon,
                "physRho": self._cfm_config["physRho"],
                "liquid": self._cfm_config["liquid"],
                "rcm_name": self._rcm_name,
                "start_year": config["start_year"],
                "end_year": config["end_year"],
                "spin_end_year": config["spin_end_year"],
                "sublim_sf": config["sublim_sf"],
                "rho_bottom": config["rho_bottom"],
                "forcing_csv": self._force_data_path,
            },
            "cfm": cfm_identity,
            "wrapper": wrapper_identity(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "started": now(),
        }

    def _read_force_data(self) -> None:
        """
        Read in the forcing data from the MAR data CSV file.
        Raises FileNotFoundError if the file does not exist.
        """

        # Read in forcing data
        # the forcing CSV of the model point in the forcing cache (FEAT-002)
        self._force_data_path = force_data_path = forcing_csv_path(
            model_point_at(self._borehole_lat, self._borehole_lon, self._rcm_name)
        )
        if not os.path.exists(force_data_path):
            raise FileNotFoundError(
                f"Forcing data file not found: {force_data_path}. Run scripts/run_cfm.py to generate the file."
            )
        self._force_data = pd.read_csv(force_data_path, index_col=0, parse_dates=True)
        logging.info(f"Forcing data read from {force_data_path}.")

    def _run_cfm(self) -> None:
        """Run the CFM model using the forcing data and configuration.
        This code is mostly copied from main.py in the CFM repository.
        """

        makeSpinFiles, FirnDensityNoSpin = import_cfm(self._cfm_version["repo"])

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
        self._manifest["cfm_config"] = self._cfm_config
        write_manifest(self._cfm_config["resultsFolder"], self._manifest)

        ### Create CFM instance by passing config file and forcing data, then run the model
        firn = FirnDensityNoSpin(
            f"{self._cfm_output_path}/{self._json_config_name}",
            climateTS=climateTS,
            NewSpin=self._cfm_config["NewSpin"],
            SEBfluxes=SEBfluxes,
        )
        firn.time_evolve()
