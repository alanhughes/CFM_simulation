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

import glob
import hashlib
import json
import logging
import os
import socket
import stat
from datetime import datetime, timezone

from cfm_config import config, share_data_path

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


def share_runs_root() -> str | None:
    """The root folder of the copy of the runs on the share, {CFM_share_data_path}/runs (FEAT-008), or None if
    CFM_share_data_path is not set."""
    share = share_data_path()
    return None if share is None else os.path.join(share, "runs")


# the result of the check of each share run root in this process (FEAT-008): True if it is a folder, False if it does
# not exist or the share is not available. See run_roots()
_share_root_checks: dict[str, bool] = {}


def run_roots() -> list[str]:
    """The run roots that exist, local first: [runs_root(), share_runs_root()] (FEAT-008).

    A root that is not set or does not exist is left out. The share is mounted "soft": if it is not available, the
    check of the share root raises OSError after a timeout, not an empty folder. Then this gives a warning and the
    local root only.

    The share root is checked once in each process, and the result is used again: a share that is not available
    gives one timeout and one warning, not one for each call. reset_share_check() clears the result (for the tests).
    """
    roots = [runs_root()] if os.path.isdir(runs_root()) else []
    share = share_runs_root()
    if share is not None:
        if share not in _share_root_checks:
            _share_root_checks[share] = _is_share_root(share)
        if _share_root_checks[share]:
            roots.append(share)
    return roots


def _is_share_root(share: str) -> bool:
    """Whether the share run root is a folder. False, with a warning, if the share is not available."""
    try:
        # os.stat, not os.path.isdir, which hides the OSError of a share that is not available
        return stat.S_ISDIR(os.stat(share).st_mode)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError as error:
        logging.warning(
            f"The share run root {share} is not available ({error}). Using the local runs only in this process."
        )
        return False


def reset_share_check() -> None:
    """Forget the result of the check of the share root, so that the next run_roots() checks it again."""
    _share_root_checks.clear()


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


def run_code_drift(wrapper: dict, root: str | None = None) -> dict | None:
    """The wrapper (with its "finished") of the complete new run of the same run code version that finished last, if
    its run_code_sha256 is different. None if it is the same, or if the version has no complete run.

    A run code version stands for one result of the run code (FEAT-005), so a different run_code_sha256 for one
    version means that the run code changed without a new RUN_CODE_VERSION. Only the last run counts (FEAT-005 open
    question 3): after a change, the first run warns, and once a run of the new code is complete, the next runs do not.
    With all the runs of the version, a change that needs no new version would warn for ever.

    This reads the manifests in root, or in each root of run_roots() if root is None, as sim.runs.find_runs does,
    but without sim.runs, which a CFM run does not import (RUN_CODE_EXCLUDED). The last run is the last of all the
    roots, the local root and the share (FEAT-008): after its local copy is removed, the last run is on the share
    only. A share that is not available is left out (run_roots). It ignores the runs that are not complete, the
    migrated runs, as their wrapper code is not known, and the manifests with no wrapper.run_code_version
    (hash_version 2, from before FEAT-005): their run_code_sha256 includes src/sim/runs.py, so it cannot be compared.
    A manifest that cannot be read is ignored too.
    """
    roots = [root] if root else run_roots()
    pattern = os.path.join("*", "*", "*", "*", MANIFEST_NAME)
    paths = [path for one_root in roots for path in glob.glob(os.path.join(one_root, pattern))]
    last, last_key = None, None
    for path in sorted(paths):
        try:
            manifest = read_manifest(os.path.dirname(path))
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, dict) or "migrated" in manifest or manifest.get("status") != COMPLETE:
            continue
        other = manifest.get("wrapper")
        if not isinstance(other, dict) or other.get("run_code_version") != wrapper["run_code_version"]:
            continue
        key = (_finished_time(manifest.get("finished")), path)
        if last_key is None or key > last_key:
            last, last_key = {**other, "finished": manifest.get("finished")}, key
    if last is None or last.get("run_code_sha256") == wrapper["run_code_sha256"]:
        return None
    return last


def _finished_time(finished) -> datetime:
    """The time of a "finished" value (ISO 8601), for the order of the runs. The earliest time if it is not known."""
    try:
        time = datetime.fromisoformat(finished)
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=timezone.utc)
    return time if time.tzinfo is not None else time.replace(tzinfo=timezone.utc)


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
