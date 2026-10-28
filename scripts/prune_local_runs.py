"""Remove the local runs that have a correct copy on the share (FEAT-008). Run it after sync_runs.py.

The runs are in {CFM_data_path}/runs. The copy is in {CFM_share_data_path}/runs, with the same relative path. The script
checks each local run with classify_run of sync_runs.py, so the two scripts use one rule. It selects a run only if:

    - the local status is "complete", and
    - the class is SAME: the share copy has the same run hash, status "complete", the same manifest.json (without
      run.sites and run.site_coords) and the same files (size and SHA-256). The script reads every file of the two
      copies again, and drops each share file from the page cache before it reads it. A file in the share copy only
      blocks the removal, and so does a symbolic link in the share copy (it can go to the local file).

The other runs are not selected, and the script tells the reason: the local status is not "complete", the local
manifest has no run hash, the local folder has a manifest.json.tmp (a failed write of the manifest, which is not
compared), NEW (no share copy), SITES (the local manifest has a site that the share manifest does not have: do
sync_runs.py --apply first), DIFFER, CONFLICT, or ERROR (an OSError on a copy). A damaged local manifest stops
find_runs, so the script stops before the first check. The script does not ask for a confirmation: the dry run is the
default, and the user reads it first.

    python scripts/prune_local_runs.py                      dry run: the selected runs, their sizes and the reasons
    python scripts/prune_local_runs.py --apply              check one run, remove it, then the next run
    python scripts/prune_local_runs.py --cfm LABEL --model-point TAG     only these runs

The script stops, and removes nothing, if CFM_share_data_path is not set, if the share is not available, or if the two
run roots are the same folder or one is in the other.

--apply checks one run and removes it before the next run. Immediately before the removal, it reads the local manifest
and the list of files with their sizes and modification times again. If they are not the same as before the check, the
run is not removed.
The script removes a folder only if it is inside the local run root, is not a symbolic link, and is not the same folder
as the share copy. The empty parent folders stay. The script never writes or removes anything on the share, and does not
remove a forcing CSV or a melt map. --apply writes a log in {CFM_data_path}/sync-logs/ with one line for each local run:
the path, the class ("-" if the run is not checked), the size in bytes and the outcome ("removed", "not selected:
<reason>" or "FAILED: <error>"). The dry run writes nothing.

The exit code is 1 for a stop at the start, an ERROR, or a failed removal. A run that is not selected for the other
reasons gives 0 (sync_runs.py gives the errors of these runs).
"""

import argparse
import importlib.util
import os
import shutil
import stat
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

from cfm_config import config, share_available, share_data_path
from sim.manifest import COMPLETE, MANIFEST_NAME, read_manifest, runs_root, share_runs_root
from sim.runs import find_runs

_spec = importlib.util.spec_from_file_location(
    "sync_runs", os.path.join(os.path.dirname(os.path.abspath(__file__)), "sync_runs.py")
)
sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync)


@dataclass
class Candidate:
    """One local run, the result of its check and what the script does with it."""

    rel: str
    folder: str
    target: str
    cls: str = "-"  # the class of classify_run; "-" if the run is not checked
    size: int = 0  # bytes, all the local files with manifest.json
    selected: bool = False
    reason: str = ""  # why the run is not selected
    outcome: str = ""
    manifest: Optional[dict] = None  # the local manifest and the files, as they were before the check
    files: Optional[Dict[str, tuple]] = None


def human(size: int) -> str:
    if size >= 10**9:
        return f"{size / 10**9:.2f} GB"
    if size >= 10**6:
        return f"{size / 10**6:.2f} MB"
    return f"{size} bytes"


def snapshot(folder: str) -> tuple:
    """(the local manifest, {name: (size, mtime_ns)} of all the files with manifest.json). An OSError or ValueError is
    raised. The modification time finds a file that is written again with the same size."""
    files = {}
    for name in sorted(sync.all_files(folder)):
        info = os.stat(os.path.join(folder, name))
        files[name] = (info.st_size, info.st_mtime_ns)
    return read_manifest(folder), files


def share_links(target: str) -> List[str]:
    """The symbolic links in the share copy (the folder, a subfolder or a file), relative to target. A link can go to
    the local file, so the check would compare the local file with itself. An OSError is raised."""
    links = [target] if stat.S_ISLNK(os.lstat(target).st_mode) else []
    for directory, folders, names in os.walk(target, onerror=sync._raise):
        for name in folders + names:
            path = os.path.join(directory, name)
            if stat.S_ISLNK(os.lstat(path).st_mode):
                links.append(path)
    return [os.path.relpath(path, target) for path in links]


def inside(path: str, root: str) -> bool:
    """Whether the real path is root or is in root."""
    return os.path.commonpath([os.path.realpath(path), os.path.realpath(root)]) == os.path.realpath(root)


def check(item: Candidate, status: str) -> None:
    """Fill the class, the size and the reason of the run. An OSError or ValueError gives the class ERROR."""
    try:
        item.manifest, item.files = snapshot(item.folder)
        item.size = sum(size for size, _ in item.files.values())
        if status != COMPLETE:
            item.reason = f"the local status is '{status}', not 'complete'"
            return
        # classify_run accepts two manifests with no run hash (None == None)
        if not isinstance(item.manifest, dict) or not item.manifest.get("run_hash"):
            item.reason = "the local manifest has no run hash"
            return
        # classify_run does not compare manifest.json.tmp: it can have a site that manifest.json does not have
        if sync.MANIFEST_TMP in item.files:
            item.reason = f"the local folder has {sync.MANIFEST_TMP} (a failed write of the manifest)"
            return
        run = sync.classify_run(item.folder, item.target, item.rel, drop_share_cache=True)
        links = share_links(item.target) if run.cls == sync.SAME else []
    except (OSError, ValueError) as error:
        item.cls, item.reason = sync.ERROR, f"{sync.ERROR}: {error}"
        return
    item.cls = run.cls
    if links:
        item.reason = "the share copy has a symbolic link: " + ", ".join(links)
    elif run.cls == sync.SAME:
        item.selected = True
    elif run.cls == sync.NEW:
        item.reason = "no share copy"
    elif run.cls == sync.SITES:
        item.reason = ("the local manifest has a site that the share manifest does not have; "
                       "do sync_runs.py --apply first")
    else:
        item.reason = f"{run.cls}: " + "; ".join(run.notes)


def remove(item: Candidate) -> str:
    """Remove the local folder if it is safe, and give the outcome: "removed" or "FAILED: <reason>"."""
    try:
        manifest, files = snapshot(item.folder)
        if manifest != item.manifest or files != item.files:
            return "FAILED: the local run changed during the check"
        root = os.path.realpath(runs_root())
        real = os.path.realpath(item.folder)
        if real == root or not inside(item.folder, root):
            return "FAILED: the folder is not inside the local run root"
        if os.path.islink(item.folder):
            return "FAILED: the folder is a symbolic link"
        if os.path.samefile(item.folder, item.target):
            return "FAILED: the folder is the share copy"
        try:
            shutil.rmtree(item.folder)
        except OSError as error:
            return f"FAILED: the removal stopped; a part of the folder is possibly removed ({error})"
    except Exception as error:  # report the run, and go on with the next
        return f"FAILED: {error}"
    return "removed"


def log_line(log, item: Candidate) -> None:
    log.write(f"{item.rel}\t{item.cls}\t{item.size}\t{item.outcome}\n")
    log.flush()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Remove the local runs that have a correct copy on the share (dry run by default)"
    )
    parser.add_argument("--apply", action="store_true", help="remove the runs and write the log")
    parser.add_argument("--cfm", help="only the runs of this CFM label")
    parser.add_argument("--model-point", help="only the runs of this model point (ModelPoint.tag)")
    args = parser.parse_args(argv)

    share = share_data_path()
    if share is None:
        print("ERROR: CFM_share_data_path is not set in config.yaml.", file=sys.stderr)
        return 1
    if not share_available():
        print(f"ERROR: the share {share} is not available.", file=sys.stderr)
        return 1
    if inside(runs_root(), share_runs_root()) or inside(share_runs_root(), runs_root()):
        print(f"ERROR: the run roots {runs_root()} and {share_runs_root()} are the same folder, or one is in the "
              "other. Nothing is removed.", file=sys.stderr)
        return 1

    try:
        runs = find_runs(root=runs_root(), status=None, cfm=args.cfm, model_point=args.model_point) \
            if os.path.isdir(runs_root()) else []
    except (OSError, ValueError) as error:
        print(f"ERROR: the local runs cannot be listed ({error}). Nothing is removed.", file=sys.stderr)
        return 1
    runs = sorted(runs, key=lambda run: run.folder)
    print(f"{len(runs)} local runs in {runs_root()}")

    failed = False
    items: List[Candidate] = []
    log = None
    try:
        if args.apply:
            os.makedirs(os.path.join(config["CFM_data_path"], "sync-logs"), exist_ok=True)
            log = sync._open_log(os.path.join(config["CFM_data_path"], "sync-logs"))
        for number, run in enumerate(runs, 1):
            rel = os.path.relpath(run.folder, runs_root())
            item = Candidate(rel=rel, folder=run.folder, target=os.path.join(share_runs_root(), rel))
            check(item, run.status)
            items.append(item)
            print(f"[{number}/{len(runs)}] {item.cls:8s} {rel}", flush=True)
            if item.cls == sync.ERROR:
                failed = True
                print(f"ERROR: {rel}: {item.reason}", file=sys.stderr)
            if args.apply:
                item.outcome = remove(item) if item.selected else f"not selected: {item.reason}"
                if item.outcome.startswith("FAILED"):
                    failed = True
                    print(f"ERROR: {rel}: {item.outcome}", file=sys.stderr)
                log_line(log, item)
    finally:
        if log is not None:
            log.close()

    selected = [item for item in items if item.selected]
    if not args.apply:
        for item in selected:
            print(f"selected      {human(item.size):>10s}  {item.rel}")
        print(f"{len(selected)} runs selected, {human(sum(item.size for item in selected))}")
        for item in items:
            if not item.selected:
                print(f"not selected  {item.rel}: {item.reason}")
        print("Dry run: nothing is removed. Use --apply to remove.")
        return 1 if failed else 0

    removed = [item for item in items if item.outcome == "removed"]
    count = sum(item.outcome.startswith("FAILED") for item in items)
    print(f"Done. {len(removed)} runs removed ({human(sum(item.size for item in removed))}), {count} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
