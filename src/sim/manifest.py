"""The run manifest, the run hash and the run folder (FEAT-001).

Each run has a manifest.json in its folder that records what the run was and whether it completed. A run is only
treated as done when its manifest says "complete" with the same run hash. An empty folder, or one left by a run that
crashed or was killed, does not block a new run (BUG-004).

The run hash is the SHA-256 of a canonical JSON document of everything that changes the results (run_hash_inputs). Its
first 8 characters name the run folder:

    <CFM_data_path>/runs/<ModelPoint.tag>/<physRho>_<liquid>_<start>-<end>/<cfm-label>/<hash8>/

The run is for a model point, not for a site: two sites in one grid cell share it (FEAT-002). The manifest records the
sites that asked for the run in run.sites, and the coordinates of each site in run.site_coords, {name: [lat, lon]}.
The site names and coordinates are not in the run hash. A manifest from before run.site_coords has the coordinates of
its first site only, in run.borehole_lat and run.borehole_lon.
"""

import hashlib
import json
import os
import socket
from datetime import datetime, timezone

from cfm_config import config

MANIFEST_NAME = "manifest.json"

RUNNING = "running"
COMPLETE = "complete"
FAILED = "failed"

# a change to the content of run_hash_inputs() must change this number, so that the old and new hashes differ
# 2: the run code (src/ without src/plot/) in place of the src/ tree. 3: RUN_CODE_VERSION in place of the hash of the
# run code (FEAT-005)
HASH_VERSION = 3

# the wrapper parameters in config.yaml that change the results
WRAPPER_PARAMS = ("start_year", "end_year", "spin_end_year", "sublim_sf", "rho_bottom")

# cfm_config keys that are paths, not physics. resultsFolder is the run folder, which contains the hash
_UNHASHED_CFM_CONFIG_KEYS = ("resultsFolder",)

# the keys of the measured CFM identity that change the results. Not the label, repo or python paths: they are names
_HASHED_CFM_IDENTITY_KEYS = ("commit", "code_diff_sha256", "python_version", "pip_freeze")

HASH_LENGTH = 8


def run_hash_inputs(
    cfm_config: dict,
    wrapper_params: dict,
    model_point_tag: str,
    rcm_release: str,
    forcing_csv_sha256: str,
    cfm_identity: dict,
    wrapper_identity: dict,
) -> dict:
    """The document that the run hash is made from. It has no times, host, pid, paths, site names or coordinates.

    Of this repo, it has only the run code version (RUN_CODE_VERSION, FEAT-005). The commit, the status, src_tree and
    run_code_sha256 of wrapper_identity are in the manifest only.
    """
    return {
        "hash_version": HASH_VERSION,
        "cfm_config": {key: value for key, value in cfm_config.items() if key not in _UNHASHED_CFM_CONFIG_KEYS},
        "wrapper_params": {key: wrapper_params[key] for key in WRAPPER_PARAMS},
        "model_point": {"tag": model_point_tag, "release": rcm_release},
        "forcing_csv_sha256": forcing_csv_sha256,
        "cfm": {key: cfm_identity[key] for key in _HASHED_CFM_IDENTITY_KEYS},
        "wrapper_run_code_version": wrapper_identity["run_code_version"],
    }


def run_hash(inputs: dict) -> str:
    """SHA-256 of the canonical JSON of run_hash_inputs(): sorted keys, no spaces."""
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def runs_root() -> str:
    """The root folder of all the runs."""
    return os.path.join(config["CFM_data_path"], "runs")


def run_folder(
    model_point_tag: str, phys_rho: str, liquid: str, start_year: int, end_year: int, cfm_label: str, hash_: str
) -> str:
    """Folder of one run: runs/<tag>/<physRho>_<liquid>_<start>-<end>/<cfm-label>/<hash8>/."""
    return os.path.join(
        runs_root(),
        model_point_tag,
        f"{phys_rho}_{liquid}_{start_year}-{end_year}",
        cfm_label,
        hash_[:HASH_LENGTH],
    )


def now() -> str:
    """Current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_manifest(folder: str) -> dict | None:
    """Manifest of the run in folder, or None if the folder has no manifest."""
    path = os.path.join(folder, MANIFEST_NAME)
    if not os.path.exists(path):
        return None
    with open(path) as file:
        return json.load(file)


def write_manifest(folder: str, manifest: dict) -> None:
    """Write the manifest via a temporary file, so a crash mid-write cannot leave a half-written manifest."""
    path = os.path.join(folder, MANIFEST_NAME)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as file:
        json.dump(manifest, file, indent=4, default=str)
        file.write("\n")
    os.replace(tmp_path, path)


def is_alive(manifest: dict) -> bool | None:
    """Whether the process that wrote a "running" manifest is still alive. None if it ran on another host."""
    if manifest.get("host") != socket.gethostname():
        return None
    try:
        os.kill(manifest["pid"], 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # the pid exists but belongs to another user
        return True
    return True
