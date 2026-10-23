import copy
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

from cfm_config import config
from cfm_config import cfm_version as lookup_cfm_version
from force.cache import forcing_csv_path, forcing_json_path, sha256_file
from force.model_point import ModelPoint, model_point_at
from sim.identity import check_cfm_identity, check_wrapper_identity, measure_cfm_identity, wrapper_identity
from sim.manifest import (
    COMPLETE,
    FAILED,
    RUNNING,
    WRAPPER_PARAMS,
    is_alive,
    now,
    read_manifest,
    run_folder,
    run_hash,
    run_hash_inputs,
    write_manifest,
)

CFM_CONFIG_NAME = "CFMconfig.json"

logging.basicConfig(level=logging.INFO)


def import_cfm(repo: str) -> tuple:
    """Import makeSpinFiles and FirnDensityNoSpin from the CFM in repo.

    The CFM's modules are flat top-level ones, so one process can only import one CFM. scripts/run_cfm.py starts each
    run in its own process (sim.worker) for this reason.
    """
    loaded = sys.modules.get("firn_density_nospin")
    if loaded is not None and os.path.realpath(os.path.dirname(loaded.__file__)) != os.path.realpath(repo):
        raise RuntimeError(
            f"This process already imported the CFM from {os.path.dirname(loaded.__file__)}, not {repo}."
        )
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

        The run folder is not known until run(): it contains the run hash, which needs the forcing CSV and the
        measured CFM identity (FEAT-001).

        Args:
            borehole_lat (float): Latitude of borehole location.
            borehole_lon (float): Longitude of borehole location.
            physRho (str): Physical densification scheme to use in CFM model.
            rcm_name (Literal["MAR", "RACMO"]): RCM name to use for forcing data (e.g., MAR or RACMO).
            liquid (Literal["bucket"]): meltwater scheme to use (currently only bucket)
            cfm_version (dict | None): CFM version from cfm_config.cfm_version(), default_cfm_version if None. This
                process must be running that version's python.
            force (bool): rerun even if a complete run exists in the output folder.
            site (str | None): name in sites.yaml of the site at borehole_lat/borehole_lon, for the manifest. Two sites
                in one grid cell share a run; the manifest lists all the sites that asked for it.
        """

        # all these need to be passed in because they vary by run
        self._rcm_name: Literal["MAR", "RACMO"] = rcm_name
        self._borehole_lat: float = borehole_lat
        self._borehole_lon: float = borehole_lon

        # a copy of the cfm config, as _run_cfm adds keys to it
        self._cfm_config: dict = copy.deepcopy(config["cfm_config"])

        # set physRho and borehole loc from argument in case we've used command line to override config
        self._cfm_config["physRho"] = physRho
        self._force_data: pd.DataFrame = None
        self._force_data_path: str = None

        # also set the liquid scheme in the config
        self._cfm_config["liquid"] = liquid

        self._force: bool = force
        self._site: str | None = site
        self._cfm_version: dict = cfm_version or lookup_cfm_version()
        self._manifest: dict = {}
        self._model_point: ModelPoint | None = None
        self._hash_inputs: dict = {}
        self._run_hash: str | None = None
        self._cfm_identity: dict = {}
        self._wrapper_identity: dict = {}

    def run(self) -> None:
        """Run the CFM model with the specified configuration and forcing data, unless a complete run exists.

        The run folder is runs/<ModelPoint.tag>/<physRho>_<liquid>_<start>-<end>/<cfm-label>/<hash8>/ (see
        sim.manifest). The run's state is kept in manifest.json in that folder: "running" while the CFM runs, then
        "complete" or "failed". Only a "complete" run with the same run hash is skipped, and --force reruns even that.
        """

        results_folder = self.prepare()
        if not self._should_run(results_folder):
            return

        self._read_force_data()
        os.makedirs(results_folder, exist_ok=True)

        # run.log also stops the CFM deleting the folder at spin-up: it rmtree's resultsFolder if every file in it is
        # .json or .hdf5 (firn_density_spin.py), which would take manifest.json with it. With run.log present it only
        # deletes the .hdf5 files.
        log_handler = logging.FileHandler(os.path.join(results_folder, "run.log"))
        log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(log_handler)

        self._manifest = self._new_manifest(self._cfm_identity, self._wrapper_identity)
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

    def prepare(self) -> str:
        """Measure the CFM and the wrapper, find the forcing CSV, and make the run hash. Returns the run folder.

        Makes no folder and runs nothing. Stops with an error if the interpreter, the CFM checkout or src/ is not
        correct, or if the forcing CSV does not exist.
        """
        if os.path.realpath(sys.executable) != os.path.realpath(self._cfm_version["python"]):
            raise RuntimeError(
                f"CFM version {self._cfm_version['label']!r} runs with {self._cfm_version['python']}, "
                f"but this is {sys.executable}. Use scripts/run_cfm.py, which starts the right one."
            )
        self._cfm_identity = cfm_identity = measure_cfm_identity(self._cfm_version)
        check_cfm_identity(cfm_identity)
        self._wrapper_identity = wrapper = wrapper_identity()
        check_wrapper_identity(wrapper)

        # the forcing CSV must exist before the folder is made, so a missing CSV leaves nothing behind (BUG-004)
        self._model_point = model_point_at(self._borehole_lat, self._borehole_lon, self._rcm_name)
        self._force_data_path = forcing_csv_path(self._model_point)
        forcing_sha256 = self._forcing_sha256()

        self._hash_inputs = run_hash_inputs(
            self._cfm_config,
            {key: config[key] for key in WRAPPER_PARAMS},
            self._model_point.tag,
            self._model_point.release,
            forcing_sha256,
            cfm_identity,
            wrapper,
        )
        self._run_hash = run_hash(self._hash_inputs)
        results_folder = run_folder(
            self._model_point.tag,
            self._cfm_config["physRho"],
            self._cfm_config["liquid"],
            config["start_year"],
            config["end_year"],
            self._cfm_version["label"],
            self._run_hash,
        )
        self._cfm_config["resultsFolder"] = results_folder
        return results_folder

    def _should_run(self, results_folder: str) -> bool:
        """Whether to run the CFM, from the manifest (if any) of an earlier run in results_folder.

        Only a "complete" manifest with the same run hash is skipped (BUG-004). A manifest with a different hash is an
        error, also with --force: it is a different run (hash8 is the same by chance), or a folder that was copied.
        """

        if not os.path.exists(results_folder):
            return True

        manifest = read_manifest(results_folder)

        if manifest is None:
            # a folder with results but no manifest is not from sim.run; do not delete its results without --force
            if os.path.exists(os.path.join(results_folder, self._cfm_config["resultsFileName"])) and not self._force:
                raise RuntimeError(
                    f"{results_folder} has results but no manifest, so its run is not known. "
                    "Use --force to run again and replace the results."
                )
            logging.info(f"{results_folder} has no manifest. Running.")
            return True

        if manifest.get("run_hash") != self._run_hash:
            raise RuntimeError(
                f"{results_folder} contains the run {manifest.get('run_hash')}, not {self._run_hash}. "
                "Move that folder, and do the run again."
            )

        status = manifest["status"]
        if self._force:
            logging.info(f"--force: rerunning {results_folder} (status was {status}).")
            if status == RUNNING and is_alive(manifest):
                raise RuntimeError(f"{results_folder} is being run by pid {manifest['pid']} on {manifest['host']}.")
            return True

        if status == COMPLETE:
            logging.info(f"Complete run found at {results_folder}, skipping. Use --force to rerun.")
            self._add_site(results_folder, manifest)
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

    def _add_site(self, results_folder: str, manifest: dict) -> None:
        """Add this site to the sites of a complete run. Two sites in one grid cell share a run (FEAT-002)."""
        sites = manifest["run"].setdefault("sites", [])
        if self._site is not None and self._site not in sites:
            sites.append(self._site)
            write_manifest(results_folder, manifest)
            logging.info(f"Added site {self._site} to the sites of {results_folder}: {', '.join(sites)}.")

    def _forcing_sha256(self) -> str:
        """SHA-256 of the forcing CSV. If the CSV has a JSON file (FEAT-002), its csv_sha256 must agree."""
        if not os.path.exists(self._force_data_path):
            raise FileNotFoundError(
                f"Forcing data file not found: {self._force_data_path}. Run scripts/run_cfm.py to generate the file."
            )
        sha256 = sha256_file(self._force_data_path)
        json_path = forcing_json_path(self._force_data_path)
        if os.path.exists(json_path):
            with open(json_path) as file:
                recorded = json.load(file).get("csv_sha256")
            if recorded is not None and recorded != sha256:
                raise RuntimeError(
                    f"{self._force_data_path} has SHA-256 {sha256}, but {json_path} records {recorded}. "
                    "The CSV changed after it was made. Make the forcing data again."
                )
        return sha256

    def _new_manifest(self, cfm_identity: dict, wrapper: dict) -> dict:
        """Manifest for a run that is about to start. The final cfm_config is added once _run_cfm has made it."""
        mp = self._model_point
        return {
            "status": RUNNING,
            "run_hash": self._run_hash,
            "run": {
                "sites": [self._site] if self._site is not None else [],
                "model_point": {
                    "tag": mp.tag,
                    "rcm_name": mp.rcm_name,
                    "release": mp.release,
                    "i": mp.i,
                    "j": mp.j,
                    "cell_lat": mp.cell_lat,
                    "cell_lon": mp.cell_lon,
                    "distance_km": mp.distance_km,
                    "method": mp.method,
                },
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
            "wrapper": wrapper,
            "hash_inputs": self._hash_inputs,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "started": now(),
        }

    def _read_force_data(self) -> None:
        """Read the forcing CSV of the model point in the forcing cache (FEAT-002). run() found its path."""
        self._force_data = pd.read_csv(self._force_data_path, index_col=0, parse_dates=True)
        logging.info(f"Forcing data read from {self._force_data_path}.")

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

        # write updated config to json, in the run folder. run.log is in the folder, so the CFM does not delete it
        config_path = os.path.join(self._cfm_config["resultsFolder"], CFM_CONFIG_NAME)
        with open(config_path, "w") as fp:
            fp.write(json.dumps(self._cfm_config, sort_keys=False, indent=4, separators=(",", ": ")))
        self._manifest["cfm_config"] = self._cfm_config
        write_manifest(self._cfm_config["resultsFolder"], self._manifest)

        ### Create CFM instance by passing config file and forcing data, then run the model
        firn = FirnDensityNoSpin(
            config_path,
            climateTS=climateTS,
            NewSpin=self._cfm_config["NewSpin"],
            SEBfluxes=SEBfluxes,
        )
        firn.time_evolve()
