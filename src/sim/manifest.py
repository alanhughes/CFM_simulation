"""The run manifest: a manifest.json in each CFM output folder that records what the run was and whether it completed.

A run is only treated as done when its manifest says "complete". An empty folder, or one left by a run that crashed or
was killed, no longer blocks a new run (BUG-004).
"""

import json
import os
import socket
from datetime import datetime, timezone

MANIFEST_NAME = "manifest.json"

RUNNING = "running"
COMPLETE = "complete"
FAILED = "failed"


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
