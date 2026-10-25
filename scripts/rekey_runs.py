"""Give the new runs of hash_version 2 the run hash of hash_version 3, without doing them again (FEAT-005, MIG-003).

hash_version 3 has the run code version (wrapper_run_code_version, RUN_CODE_VERSION) in place of the hash of the run
code (wrapper_run_code_sha256). The user gives the run code version of the runs with --run-code-version. The script
selects the manifests in {CFM_data_path}/runs with no "migrated" key, status "complete" and hash_version 2. For each
one it:

1. Checks the manifest: its run_hash is the hash of its hash_inputs, and its wrapper.run_code_sha256 is the run code of
   its own wrapper.commit under the rule of hash_version 2 (src/ without src/plot/). Thus, a run gets a version only if
   its recorded code is the code of that commit.
2. Makes the new hash_inputs: hash_version 3, and wrapper_run_code_version in place of wrapper_run_code_sha256. All the
   other items stay the same, so the new hash is the hash that sim.run gives for the same run now.
3. Renames the folder <hash8> to the new <hash8>, in the same parent folder (os.rename). It refuses if that exists.
4. Writes the manifest again (atomic): the new run_hash and hash_inputs, wrapper.run_code_version,
   wrapper.run_code_sha256 of the commit under the rule of now (RUN_CODE_EXCLUDED; the old value would give a false
   drift warning), cfm_config.resultsFolder with the new folder, and "rekeyed" with the old values.

The rename comes first and the manifest last. If the script stops between the two, the folder has the new name and
the manifest of hash_version 2. A second run of the script selects it again (FINISH) and writes the manifest only. After
a complete re-key the manifest has hash_version 3, so a second run selects nothing.

The script does not change CFMconfig.json (it keeps the old resultsFolder; no code reads it after the run), run.log or
any .hdf5 file. The site links in sites/ go to the model point folders, so they do not change.

The default is a dry run: it prints each run, and writes nothing. If a run is refused, --apply changes nothing.

Usage:
    python scripts/rekey_runs.py --run-code-version 1            # dry run (the default)
    python scripts/rekey_runs.py --run-code-version 1 --apply    # rename the folders and write the manifests
"""

import argparse
import copy
import datetime
import functools
import glob
import os
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from sim.identity import RUN_CODE_EXCLUDED, RUN_CODE_VERSION, run_code_sha256
from sim.manifest import (
    COMPLETE,
    HASH_LENGTH,
    HASH_VERSION,
    MANIFEST_NAME,
    read_manifest,
    run_hash,
    runs_root,
    write_manifest,
)

FROM_HASH_VERSION = 2
TO_HASH_VERSION = 3

# the run code of hash_version 2: src/ without src/plot/ (src/sim/runs.py was run code then)
OLD_RUN_CODE_EXCLUDED = ("src/plot/",)

REASON = (
    "FEAT-005: the run hash has the run code version in place of the hash of the run code (hash_version 2 to 3). "
    "The results did not change, so the run was not done again."
)


@functools.lru_cache(maxsize=None)
def _run_code_sha256(commit: str, excluded: tuple) -> Optional[str]:
    """run_code_sha256 of commit under a rule, or None if this repo does not have the commit."""
    try:
        return run_code_sha256(commit, excluded)
    except subprocess.CalledProcessError:
        return None


@dataclass
class Rekey:
    """The re-key of one run. action: REKEY (rename and write the manifest), FINISH (the folder has the new name, write
    the manifest) or REFUSE."""

    folder: str
    manifest: dict
    action: str = "REFUSE"
    notes: List[str] = field(default_factory=list)
    old_folder: str = ""
    new_folder: str = ""
    new_hash: str = ""
    new_hash_inputs: dict = field(default_factory=dict)
    new_run_code_sha256: Optional[str] = None

    @property
    def old_hash(self) -> str:
        return self.manifest.get("run_hash", "")


def new_hash_inputs(hash_inputs: dict, run_code_version: int) -> dict:
    """The hash_inputs of hash_version 3 for those of hash_version 2: only the item of the wrapper code changes."""
    new = {key: value for key, value in copy.deepcopy(hash_inputs).items() if key != "wrapper_run_code_sha256"}
    new["hash_version"] = TO_HASH_VERSION
    new["wrapper_run_code_version"] = run_code_version
    return new


def is_selected(manifest: Optional[dict]) -> bool:
    return (
        isinstance(manifest, dict)
        and "migrated" not in manifest
        and manifest.get("status") == COMPLETE
        and (manifest.get("hash_inputs") or {}).get("hash_version") == FROM_HASH_VERSION
    )


def plan_rekey(folder: str, manifest: dict, run_code_version: int) -> Rekey:
    plan = Rekey(folder, manifest)
    inputs = manifest.get("hash_inputs") or {}
    wrapper = manifest.get("wrapper") or {}
    commit, recorded = wrapper.get("commit"), wrapper.get("run_code_sha256")

    if run_hash(inputs) != plan.old_hash:
        plan.notes.append(f"its run_hash is not the hash of its hash_inputs ({run_hash(inputs)})")
    if "wrapper_run_code_sha256" not in inputs or inputs["wrapper_run_code_sha256"] != recorded:
        plan.notes.append("hash_inputs.wrapper_run_code_sha256 is missing or is not wrapper.run_code_sha256")
    old_rule = _run_code_sha256(commit, OLD_RUN_CODE_EXCLUDED) if commit else None
    if old_rule is None:
        plan.notes.append(f"this repo does not have its wrapper.commit {commit}")
    elif old_rule != recorded:
        plan.notes.append(
            f"its wrapper.run_code_sha256 {recorded} is not the run code of its commit {commit} ({old_rule})"
        )
    plan.new_run_code_sha256 = _run_code_sha256(commit, RUN_CODE_EXCLUDED) if commit else None

    plan.new_hash_inputs = new_hash_inputs(inputs, run_code_version)
    plan.new_hash = run_hash(plan.new_hash_inputs)
    parent, name = os.path.split(folder)
    plan.old_folder = os.path.join(parent, plan.old_hash[:HASH_LENGTH])
    plan.new_folder = os.path.join(parent, plan.new_hash[:HASH_LENGTH])
    if name == plan.old_hash[:HASH_LENGTH]:
        if os.path.lexists(plan.new_folder) and plan.new_folder != folder:
            plan.notes.append(f"the new folder {plan.new_folder} exists")
        else:
            plan.action = "REKEY"
    elif name == plan.new_hash[:HASH_LENGTH]:
        plan.action = "FINISH"  # an earlier --apply renamed the folder and stopped before the manifest
    else:
        plan.notes.append(f"the folder name {name} is not the start of its run_hash or of the new run hash")

    if plan.notes:
        plan.action = "REFUSE"
    return plan


def build_plans(run_code_version: int, root: Optional[str] = None) -> List[Rekey]:
    plans = []
    for path in sorted(glob.glob(os.path.join(root or runs_root(), "*", "*", "*", "*", MANIFEST_NAME))):
        folder = os.path.dirname(path)
        manifest = read_manifest(folder)
        if is_selected(manifest):
            plans.append(plan_rekey(folder, manifest, run_code_version))
    return plans


def rekeyed_manifest(plan: Rekey, run_code_version: int) -> dict:
    manifest = copy.deepcopy(plan.manifest)
    old_wrapper = plan.manifest["wrapper"]
    manifest["run_hash"] = plan.new_hash
    manifest["hash_inputs"] = plan.new_hash_inputs
    manifest["wrapper"].update(run_code_version=run_code_version, run_code_sha256=plan.new_run_code_sha256)
    if isinstance(manifest.get("cfm_config"), dict):
        manifest["cfm_config"]["resultsFolder"] = plan.new_folder
    manifest["rekeyed"] = {
        "from_run_hash": plan.old_hash,
        "from_hash_version": FROM_HASH_VERSION,
        "from_run_code_sha256": old_wrapper["run_code_sha256"],
        "from_folder": plan.old_folder,
        "run_code_version": run_code_version,
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "reason": REASON,
        "script": "scripts/rekey_runs.py",
    }
    return manifest


def apply(plan: Rekey, run_code_version: int) -> None:
    """Rename the folder (REKEY), then write the manifest. The manifest is last, so a stop between the two leaves a
    folder that the next run of the script finishes (FINISH)."""
    if plan.action == "REKEY" and plan.new_folder != plan.folder:
        if os.path.lexists(plan.new_folder):
            raise RuntimeError(f"{plan.new_folder} exists.")
        os.rename(plan.folder, plan.new_folder)
    write_manifest(plan.new_folder, rekeyed_manifest(plan, run_code_version))


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-code-version", type=int, required=True, help="The run code version of the runs")
    parser.add_argument("--apply", action="store_true", help="Rename the folders and write the manifests")
    args = parser.parse_args(argv)

    if HASH_VERSION != TO_HASH_VERSION:
        print(f"ERROR: this script makes hash_version {TO_HASH_VERSION}, not {HASH_VERSION}.", file=sys.stderr)
        return 1
    if args.run_code_version < 1:
        parser.error("--run-code-version must be 1 or more.")

    version = args.run_code_version
    if args.apply:
        print(f"APPLY in {runs_root()}\n")
    else:
        print(f"DRY RUN: no folder is renamed and no file is written. {runs_root()}\n")
    if version != RUN_CODE_VERSION:
        print(f"Note: RUN_CODE_VERSION of this code is {RUN_CODE_VERSION}, not {version}.\n")
    plans = build_plans(version)
    for plan in plans:
        commit = plan.manifest.get("wrapper", {}).get("commit")
        print(f"{plan.action:7s} {plan.folder}")
        print(f"        -> {plan.new_folder}")
        print(f"        run_hash {plan.old_hash} -> {plan.new_hash}")
        print(
            f"        wrapper.run_code_sha256 {plan.manifest.get('wrapper', {}).get('run_code_sha256')} -> "
            f"{plan.new_run_code_sha256} (commit {commit}, run code version {version})"
        )
        for note in plan.notes:
            print(f"        REFUSED: {note}")
    counts = {action: sum(plan.action == action for plan in plans) for action in ("REKEY", "FINISH", "REFUSE")}
    summary = ", ".join(f"{action} {n}" for action, n in counts.items())
    print(f"\nSummary: {len(plans)} runs of hash_version {FROM_HASH_VERSION}: {summary}")
    if counts["REFUSE"]:
        print("ERROR: a run is refused, so nothing is changed.", file=sys.stderr)
        return 1
    if not args.apply:
        return 0
    for plan in plans:
        apply(plan, version)
        print(f"DONE    {plan.new_folder}")
    print("No CFMconfig.json, run.log or .hdf5 file is changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
