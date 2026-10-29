"""Add migrated.inferred_cfm_config to the manifests of the migrated runs, in each run root (FEAT-009, MIG-005).

A migrated run with no CFMconfig JSON has "cfm_config": null, so find_runs(cfm_config={"SEB": ...}) cannot find it.
infer_cfm_config() of scripts/migrate_runs.py gives SEB from the form of CFMforcing.hdf5. migrate_runs.py writes it for
the runs that it copies now; this script writes it into the manifests of the runs that it copied before.

The script selects the manifests with a "migrated" key and "cfm_config": null in each root of run_roots(): the local
runs, and the copy of the runs on the share (FEAT-008). A run in the two roots has one folder in each, with the same
path relative to the root. For each folder it reads CFMforcing.hdf5 and gives an action:

    ADD     the manifest has no migrated.inferred_cfm_config. --apply writes it, and migrated.inferred_from.
    SAME    the manifest has the same migrated.inferred_cfm_config and migrated.inferred_from. Nothing is written.
    DIFFER  the manifest has a different migrated.inferred_cfm_config or migrated.inferred_from. The script refuses.
    ERROR   the two copies of a run give a different inferred value. The script refuses.

The manifests of the two copies get the same keys, so that sync_runs.py compares them as SAME. The run hash and
hash_inputs do not change. cfm_config stays in migrated.unknown. The script changes no file other than manifest.json,
and it writes each manifest via a temporary file (write_manifest). Before each write, it reads the manifest again and
refuses if it changed after the plan.

The default is a dry run: it prints each folder, and writes nothing. If a folder is refused, --apply changes nothing.
If CFM_share_data_path is set and the share is not available, the script stops: the two copies must change together.

Usage:
    python scripts/infer_cfm_config.py            # dry run (the default)
    python scripts/infer_cfm_config.py --apply    # write the manifests
"""

import argparse
import copy
import glob
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from cfm_config import share_available, share_data_path
from sim.manifest import MANIFEST_NAME, read_manifest, run_roots, share_runs_root, write_manifest

_spec = importlib.util.spec_from_file_location(
    "migrate_runs", os.path.join(os.path.dirname(os.path.abspath(__file__)), "migrate_runs.py")
)
_migrate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_migrate)
infer_cfm_config = _migrate.infer_cfm_config
inferred_from = _migrate.inferred_from

ACTIONS = ("ADD", "SAME", "DIFFER", "ERROR")


@dataclass
class Copy:
    """One folder of a run, in one root."""

    root: str
    folder: str
    manifest: dict
    inferred: dict = field(default_factory=dict)
    action: str = ""


@dataclass
class RunPlan:
    """The folders of one run (one path relative to the roots), and the value to write into each manifest."""

    relative: str
    copies: List[Copy] = field(default_factory=list)
    inferred: Optional[dict] = None
    notes: List[str] = field(default_factory=list)

    @property
    def action(self) -> str:
        """The action of the run: ERROR or DIFFER if a copy has it, else ADD if a copy has it, else SAME."""
        for action in ("ERROR", "DIFFER", "ADD"):
            if any(item.action == action for item in self.copies):
                return action
        return "SAME"


def is_selected(manifest) -> bool:
    """A migrated run with no cfm_config."""
    return isinstance(manifest, dict) and "migrated" in manifest and manifest.get("cfm_config") is None


def build_plans(roots: List[str]) -> List[RunPlan]:
    """One plan for each path relative to the roots that has a selected manifest, sorted by the path."""
    plans: Dict[str, RunPlan] = {}
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "*", "*", "*", "*", MANIFEST_NAME))):
            folder = os.path.dirname(path)
            manifest = read_manifest(folder)
            if not is_selected(manifest):
                continue
            relative = os.path.relpath(folder, root)
            plan = plans.setdefault(relative, RunPlan(relative))
            plan.copies.append(Copy(root, folder, manifest, infer_cfm_config(folder)))
    for plan in plans.values():
        values = [item.inferred for item in plan.copies]
        if any(value != values[0] for value in values[1:]):
            for item in plan.copies:
                item.action = "ERROR"
            plan.notes.append(f"the copies give different values: {values}")
            continue
        plan.inferred = values[0]
        for item in plan.copies:
            migrated = item.manifest["migrated"]
            if "inferred_cfm_config" not in migrated and "inferred_from" not in migrated:
                item.action = "ADD"
            elif (migrated.get("inferred_cfm_config"), migrated.get("inferred_from")) == (
                plan.inferred,
                inferred_from(plan.inferred),
            ):
                item.action = "SAME"
            else:
                item.action = "DIFFER"
                plan.notes.append(
                    f"{item.folder} has inferred_cfm_config {migrated.get('inferred_cfm_config')}, not {plan.inferred}"
                )
    return [plans[relative] for relative in sorted(plans)]


def new_manifest(manifest: dict, inferred: dict) -> dict:
    """The manifest with migrated.inferred_cfm_config and migrated.inferred_from."""
    manifest = copy.deepcopy(manifest)
    manifest["migrated"]["inferred_cfm_config"] = inferred
    manifest["migrated"]["inferred_from"] = inferred_from(inferred)
    return manifest


def apply(plan: RunPlan) -> None:
    """Write the manifest of each ADD copy. Raise RuntimeError if a manifest changed after the plan."""
    for item in plan.copies:
        if item.action != "ADD":
            continue
        if read_manifest(item.folder) != item.manifest:
            raise RuntimeError(f"{item.folder}: the manifest changed after the plan. Do the dry run again.")
        write_manifest(item.folder, new_manifest(item.manifest, plan.inferred))


def _root_name(root: str) -> str:
    return "share" if root == share_runs_root() else "local"


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Write the manifests")
    args = parser.parse_args(argv)

    if share_data_path() and not share_available():
        print(
            f"ERROR: the share {share_data_path()} is not available. The local and the share manifests must change "
            "together, so nothing is done.",
            file=sys.stderr,
        )
        return 1
    roots = run_roots()
    print(("APPLY" if args.apply else "DRY RUN: no file is written.") + " Roots: " + ", ".join(roots) + "\n")
    plans = build_plans(roots)
    for plan in plans:
        where = ", ".join(f"{_root_name(item.root)} {item.action}" for item in plan.copies)
        print(f"{plan.action:7s} {plan.relative}  {plan.inferred if plan.inferred is not None else ''}  ({where})")
        for note in plan.notes:
            print(f"        REFUSED: {note}")

    counts = {action: sum(plan.action == action for plan in plans) for action in ACTIONS}
    values: Dict[str, int] = {}
    for plan in plans:
        if plan.inferred is not None:
            values[str(plan.inferred)] = values.get(str(plan.inferred), 0) + 1
    folders = {
        name: sum(_root_name(item.root) == name and item.action == "ADD" for plan in plans for item in plan.copies)
        for name in ("local", "share")
    }
    print(f"\nSummary: {len(plans)} migrated runs with no cfm_config: " + ", ".join(f"{a} {n}" for a, n in counts.items()))
    print("Inferred values: " + (", ".join(f"{value} {n}" for value, n in sorted(values.items())) or "none"))
    print(f"Manifests to write: local {folders['local']}, share {folders['share']}")
    if counts["DIFFER"] or counts["ERROR"]:
        print("ERROR: a run is refused, so nothing is changed.", file=sys.stderr)
        return 1
    if not args.apply:
        return 0
    for plan in plans:
        if plan.action == "ADD":
            apply(plan)
            print(f"DONE    {plan.relative}")
    print("No file other than manifest.json is changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
