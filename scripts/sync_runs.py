"""Copy the complete local runs, the forcing CSVs and the melt maps to the share (FEAT-008).

The runs are in {CFM_data_path}/runs. The copy is in {CFM_share_data_path}/runs, with the same relative path. The script
selects the local runs with status "complete" and gives each one a class:

    NEW       the share has no folder for the run. --apply copies it.
    SAME      the share copy has the same run hash, status "complete", the same manifest.json (without run.sites and
              run.site_coords) and the same files (size and SHA-256). No action.
    SITES     as SAME, but the local manifest has a site, or the coordinates of a site, that the share manifest does not
              have. --apply reads the share manifest again, checks that it has not changed (except the sites), and
              writes it with the union of the sites: the sites and coordinates of the share first, then the new local
              sites. If the share has all the local sites, the class is SAME.
    DIFFER    the same run hash, but a file or the manifest is different, a file is in one copy only, or the status is
              not "complete". The user decides: keep, replace or skip (--on-differ).
    CONFLICT  a different run hash. A share manifest with no run_hash is also CONFLICT. The script refuses.
    ERROR     the script cannot read the run or the share copy (an OSError, for example an I/O error of the soft
              mount). The script does not copy the run. An error never gives NEW.

A share folder with the same path but no manifest, or a manifest that cannot be read, is DIFFER: the path has the
first 8 characters of the run hash, so the folder is a part, or a damage, of this run, not another run. The user can
keep it, or replace it (the old folder is moved to replaced/). A file manifest.json.tmp in the share folder (a failed
write of the sites by sim.run) is ignored.

For each DIFFER run the script shows the finished time and the CFM commit of each copy, the size and SHA-256 of each
file that is different, and whether all the datasets are identical (compare_cfm_runs.summary). Two copies with the same
run hash can have identical datasets, but different bytes in the .hdf5 file. If the share cannot be read for this
report, the script prints "not available" for the run and goes on.

The script classifies all the runs and gets all the decisions first. Then it copies. If a decision is needed ("ask")
and the input is not a terminal (or ends, Ctrl-D), --apply stops before the first copy, with exit code 1.

The copy of a run:
    1. the files, except manifest.json, go to {share}/.incoming/<relative path> (an old folder of the same run in
       .incoming is removed first: it is the only folder that the script removes. The folder of another run in
       .incoming stays);
    2. each file is flushed (fsync, posix_fadvise DONTNEED), read again from the share, and its SHA-256 is compared
       with the local file (the CIFS mount has cache=strict, so without this the read can come from the page cache.
       If posix_fadvise is not available, the script warns once);
    3. manifest.json is copied, and the files in .incoming are compared with the files of the local run: a file that
       is not in the local run stops the run;
    4. the folder is renamed to the target.
With "replace", step 4 is preceded by a rename of the share folder to {share}/replaced/<YYYY-MM-DD>/<relative path>.
The script refuses the run if that folder exists. If the second rename fails, the message gives the two folders: the
old copy is in replaced/ and the new copy is in .incoming. find_runs does not read .incoming, so it never finds a part
copy. If the script stops, the next --apply replaces the folder in .incoming. If another VM made the target before the
rename, the run fails and the script reports it.

The script never deletes a local file, and deletes on the share only the .incoming folder of the run that it copies.
It does not copy a log.

The forcing CSVs and the melt maps (no decisions):

    forcing   each .csv file under forcing_root() (local), with its .json file, goes to
              {CFM_share_data_path}/forcing/<path relative to forcing_root()>, the path that find_forcing_csv reads
              (force.cache.share_forcing_path). Other files in the forcing folder are ignored, and so is a .json file
              with no .csv file.
    melt      each {CFM_data_path}/melt/*.nc goes to {CFM_share_data_path}/melt/<name>. No local melt folder is fine.

Each file has the class NEW (the share has no file: copy it), SAME (the same size and SHA-256: no action) or CONFLICT
(the share has a different file with the same name: refused, never overwritten. A forcing file is the same file for
the same release, so this is an error). A file that cannot be read gives ERROR. A forcing CSV with no local .json
file gives ERROR, unless the share has its .json file. The exit code is 1 for CONFLICT and ERROR.

The .json file of a forcing CSV has metadata of the fetch (created, host, requested_location), so the files of two
VMs differ. A .json file that differs from the share file is SAME, and the share file is kept, if its CSV is SAME and
the two files have the same csv_sha256; else it is CONFLICT. A CSV and its .json file are one unit: if the CSV is
CONFLICT or ERROR, the .json file is not copied either.

A file goes to {share}/.incoming/forcing/<relative path> (or .incoming/melt/<name>), is flushed, read again and
compared by SHA-256 as for a run, and is linked to the target (os.link, which does not replace a file that another VM
made; the folder of the target is made) and the .incoming file is removed. If the mount does not support hard links,
the script checks the target and renames the file: then a small window remains between the check and the rename. If
the target exists, the file fails. Only the .incoming file of this file is removed first. The .json file is copied
before its .csv file, so find_forcing_csv never finds a share CSV with no .json file. If the .json file is refused or
fails, the .csv file is not copied. If the files cannot be listed (an OSError), the script reports it, does not copy
any of them, goes on with the runs, and the exit code is 1. The files are copied after the decisions of the
DIFFER runs, and before the runs. So a run
decision that is missing (ask, no terminal) also stops the copy of the files. The log has one line for each file, with
the path "forcing/<relative path>" or "melt/<name>".

The default is a dry run: it reads the local runs and each file of the share copies (to compare the SHA-256, this is
slow for many runs), and writes nothing, on the share or on the local disk. --apply writes the copies and the log
{CFM_data_path}/sync-logs/<YYYY-MM-DDTHHMMSS>.txt (each run, its class, its decision or result and its notes; a name
that exists gets a suffix), and makes {CFM_share_data_path}/runs if it does not exist. The log is also written when
--apply stops for a missing decision. The exit code is 1 if a run is CONFLICT or ERROR, a copy or a check fails, or
(with --apply) a decision is missing; else 0. The script stops with an error if CFM_share_data_path is not set or the
share is not available.

--runs PATTERN (one or more times) selects only the complete local runs whose path relative to the run root matches a
pattern (fnmatch: "*" also matches "/"; an absolute pattern is made relative to the run root), for a copy in parts.
The files are then only the forcing CSVs of the selected runs (a local CSV whose SHA-256 is the forcing_csv_sha256 of
the run hash inputs; the csv_sha256 of its .json file is used if it has one), each with its .json file. A run whose
forcing CSV is not in the local cache has no file, and a warning. No melt map is selected: no run gives its melt map.
A pattern that selects no run stops the script with an error (exit code 1), before it classifies a run.

Usage:
    python scripts/sync_runs.py                          # dry run (the default)
    python scripts/sync_runs.py --apply                  # copy; ask for each DIFFER run
    python scripts/sync_runs.py --apply --on-differ keep # one decision for all the DIFFER runs
    python scripts/sync_runs.py --runs 'RACMO_259_136/*/ab871c8/*'   # dry run of 5 runs and their forcing CSV
"""

import argparse
import copy
import datetime
import errno
import fnmatch
import glob
import importlib.util
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from cfm_config import config, share_available, share_data_path
from force.cache import forcing_root, sha256_file, share_forcing_path
from sim.manifest import COMPLETE, MANIFEST_NAME, read_manifest, runs_root, share_runs_root, write_manifest

_spec = importlib.util.spec_from_file_location(
    "compare_cfm_runs", os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_cfm_runs.py")
)
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)

NEW, SAME, SITES, DIFFER, CONFLICT, ERROR = "NEW", "SAME", "SITES", "DIFFER", "CONFLICT", "ERROR"
KEEP, REPLACE, SKIP, ASK = "keep", "replace", "skip", "ask"

INCOMING = ".incoming"
REPLACED = "replaced"
# a failed write_manifest leaves this file in a run folder. It is not part of the run
MANIFEST_TMP = f"{MANIFEST_NAME}.tmp"


def _exists(path: str) -> bool:
    """Whether path exists. os.path.exists gives False for an OSError, so on the soft mount an I/O error would look
    like a missing folder (and the class NEW). This gives False only for FileNotFoundError and NotADirectoryError."""
    try:
        os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    return True


def _raise(error: OSError) -> None:
    raise error


@dataclass
class RunSync:
    """One local run, its share target, its class and what the script does with it."""

    rel: str  # path relative to the run root, the same in the two roots
    local: str
    target: str
    cls: str = ""
    notes: List[str] = field(default_factory=list)
    diff_files: List[str] = field(default_factory=list)  # the names of the files that are different
    # {name: {"local": (size, SHA-256) or None, "share": (size, SHA-256) or None}} for the files in diff_files
    file_info: Dict[str, dict] = field(default_factory=dict)
    # {"local": (finished, CFM commit), "share": (finished, CFM commit)}, for the report of a DIFFER run
    copies: Dict[str, tuple] = field(default_factory=dict)
    decision: str = ""  # DIFFER: keep, replace or skip
    outcome: str = ""
    requires: Optional["RunSync"] = None  # a forcing CSV: its .json file, which is copied before it
    json_of: Optional["RunSync"] = None  # a forcing .json file: its CSV
    no_json: bool = False  # a forcing CSV with no .json file on the local disk
    blocked: str = ""  # a .json file that is not copied, and why


def select_local_runs() -> List[str]:
    """The folders of the local runs with status "complete", sorted. A manifest that cannot be read is left out."""
    folders = []
    for path in sorted(glob.glob(os.path.join(runs_root(), "*", "*", "*", "*", MANIFEST_NAME))):
        folder = os.path.dirname(path)
        try:
            manifest = read_manifest(folder)
        except (OSError, ValueError) as error:
            print(f"WARNING: {path} cannot be read ({error}); left out.", file=sys.stderr)
            continue
        if isinstance(manifest, dict) and manifest.get("status") == COMPLETE:
            folders.append(folder)
    return folders


def match_runs(folders: List[str], patterns: List[str]) -> List[str]:
    """The folders whose path relative to the run root matches one of the patterns (fnmatch, case-sensitive). A
    pattern is normalised first, so a trailing "/" (of a tab completion) or a leading "./" also matches, and an
    absolute pattern is made relative to the run root."""
    patterns = [
        os.path.relpath(pattern, runs_root()) if os.path.isabs(pattern) else os.path.normpath(pattern)
        for pattern in patterns
    ]
    return [
        folder
        for folder in folders
        if any(fnmatch.fnmatchcase(os.path.relpath(folder, runs_root()), pattern) for pattern in patterns)
    ]


def all_files(folder: str) -> Set[str]:
    """The names of all the files in folder, relative to folder (manifest.json too). An OSError is raised."""
    return {
        os.path.relpath(os.path.join(directory, name), folder)
        for directory, _, names in os.walk(folder, onerror=_raise)
        for name in names
    }


def list_files(folder: str) -> Dict[str, int]:
    """{name relative to folder: size} of the files of a run, without manifest.json and manifest.json.tmp."""
    return {
        name: os.path.getsize(os.path.join(folder, name))
        for name in sorted(all_files(folder))
        if name not in (MANIFEST_NAME, MANIFEST_TMP)
    }


def core(manifest: dict) -> dict:
    """The manifest without run.sites and run.site_coords: the part that the two copies must have in common."""
    manifest = copy.deepcopy(manifest)
    if isinstance(manifest.get("run"), dict):
        manifest["run"].pop("sites", None)
        manifest["run"].pop("site_coords", None)
    return manifest


def site_coords(manifest: dict) -> Dict[str, list]:
    """{site: [lat, lon]} of a manifest. A manifest from before run.site_coords has the coordinates of its first
    site only, in run.borehole_lat and run.borehole_lon (as sim.runs.Run.site_coords)."""
    run = manifest.get("run") or {}
    coords = {site: list(latlon) for site, latlon in (run.get("site_coords") or {}).items()}
    sites = run.get("sites") or []
    if sites and sites[0] not in coords and None not in (run.get("borehole_lat"), run.get("borehole_lon")):
        coords[sites[0]] = [run["borehole_lat"], run["borehole_lon"]]
    return coords


def merge_sites(local: dict, share: dict) -> Optional[dict]:
    """The share manifest with the union of the sites, or None if the share has all the local sites and coordinates.
    The union keeps the order and the coordinates of the share, then adds the new local sites."""
    local_sites = (local.get("run") or {}).get("sites") or []
    share_sites = (share.get("run") or {}).get("sites") or []
    local_coords, share_coords = site_coords(local), site_coords(share)
    new_sites = [site for site in local_sites if site not in share_sites]
    new_coords = [site for site in local_coords if site not in share_coords]
    if not new_sites and not new_coords:
        return None
    merged = copy.deepcopy(share)
    merged.setdefault("run", {})
    merged["run"]["sites"] = share_sites + new_sites
    merged["run"]["site_coords"] = {**share_coords, **{site: local_coords[site] for site in new_coords}}
    return merged


def _info(manifest: Optional[dict]) -> tuple:
    manifest = manifest if isinstance(manifest, dict) else {}
    return manifest.get("finished"), (manifest.get("cfm") or {}).get("commit")


def _size_sha(path: str) -> tuple:
    return os.path.getsize(path), sha256_file(path)


def classify_run(local: str, target: str, rel: str, drop_share_cache: bool = False) -> RunSync:
    """Give the run the class NEW, SAME, SITES, DIFFER or CONFLICT. Reads every file of an existing share copy, but
    only if the run hash is the same. An OSError is raised (main gives the class ERROR). If drop_share_cache, each
    share file is dropped from the page cache just before it is read (prune_local_runs.py)."""
    run = RunSync(rel=rel, local=local, target=target)
    if not _exists(target):
        run.cls = NEW
        return run
    local_manifest = read_manifest(local)
    run.copies["local"] = _info(local_manifest)
    try:
        share_manifest = read_manifest(target)
    except ValueError as error:  # a damaged JSON file; an OSError is raised
        run.notes.append(f"the share manifest cannot be read ({error})")
        share_manifest = None
    else:
        if share_manifest is None:
            run.notes.append("the share folder has no manifest")
        elif not isinstance(share_manifest, dict):
            run.notes.append("the share manifest is not a JSON object")
            share_manifest = None
    run.copies["share"] = _info(share_manifest)
    if share_manifest is not None:
        # the hash and the status first: a CONFLICT does not need the files
        if share_manifest.get("run_hash") != local_manifest.get("run_hash"):
            run.cls = CONFLICT
            run.notes.append(
                f"different run hash: local {local_manifest.get('run_hash')}, share {share_manifest.get('run_hash')}"
            )
            return run
        if share_manifest.get("status") != COMPLETE:
            run.notes.append(f"the share status is {share_manifest.get('status')!r}, not 'complete'")
        elif core(share_manifest) != core(local_manifest):
            run.notes.append(f"{MANIFEST_NAME} is different")
    local_files, share_files = list_files(local), list_files(target)
    for name in sorted(local_files.keys() | share_files.keys()):
        local_path, share_path = os.path.join(local, name), os.path.join(target, name)
        if name not in share_files:
            run.notes.append(f"{name}: in the local copy only")
            info = {"local": _size_sha(local_path), "share": None}
        elif name not in local_files:
            run.notes.append(f"{name}: in the share copy only")
            if drop_share_cache:
                _drop_cache(share_path)
            info = {"local": None, "share": _size_sha(share_path)}
        else:
            if drop_share_cache:
                _drop_cache(share_path)
            info = {"local": _size_sha(local_path), "share": _size_sha(share_path)}
            if info["local"] == info["share"]:
                continue
            run.notes.append(f"{name}: different")
        run.diff_files.append(name)
        run.file_info[name] = info
    if run.notes:
        run.cls = DIFFER
    else:
        run.cls = SAME if merge_sites(local_manifest, share_manifest) is None else SITES
    return run


def describe_differ(run: RunSync) -> None:
    """Print what the user needs to decide on a DIFFER run. It prints what classify_run stored, and compares the
    datasets (an OSError there gives "not available")."""
    print(f"  DIFFER {run.rel}")
    for note in run.notes:
        print(f"    - {note}")
    for side in ("local", "share"):
        finished, commit = run.copies.get(side, (None, None))
        print(f"    {side}: finished {finished}, CFM commit {commit}")
    for name in run.diff_files:
        print(f"    {name}")
        for side in ("local", "share"):
            value = run.file_info[name][side]
            print(f"      {side}: " + ("(no file)" if value is None else f"{value[0]} bytes, SHA-256 {value[1]}"))
    try:
        identical = bool(compare.summary(run.local, run.target)["bit_for_bit"])
    except OSError as error:
        print(f"    datasets: not available ({error})")
    except Exception as error:  # a missing or damaged .hdf5 file; the comparison is only information
        print(f"    datasets: not compared ({error})")
    else:
        print(f"    all datasets identical: {'yes' if identical else 'no'}")


def ask_decision(run: RunSync) -> str:
    """Ask the user for the decision on a DIFFER run. "" (no decision) if the input ends."""
    while True:
        try:
            answer = input(
                f"  {run.rel}: [k]eep the share copy, [r]eplace it with the local run, or [s]kip? "
            ).strip().lower()
        except EOFError:
            return ""
        for decision in (KEEP, REPLACE, SKIP):
            if answer and decision.startswith(answer):
                return decision


def _copy_file(source: str, target: str) -> None:
    """Copy the content only (no copystat: the CIFS mount does not need it)."""
    shutil.copyfile(source, target)


_warned_fadvise = False


def _forget(fd: int) -> None:
    """Drop the file of fd from the page cache. If posix_fadvise is not available or fails, warn once."""
    global _warned_fadvise
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError) as error:
        if not _warned_fadvise:
            _warned_fadvise = True
            print(f"WARNING: posix_fadvise is not available ({error}); the check may read the page cache.",
                  file=sys.stderr)


def _drop_cache(path: str) -> None:
    """Drop the file from the page cache, so that the next read comes from the share. The file is opened for a read
    only (no fsync). If posix_fadvise is not available or fails, warn once: the check may read the page cache."""
    fd = os.open(path, os.O_RDONLY)
    try:
        _forget(fd)
    finally:
        os.close(fd)


def _flush_and_forget(path: str) -> None:
    """fsync the file, and drop it from the page cache, so that the next read comes from the share. If posix_fadvise
    is not available or fails, warn once: the check may then read the page cache."""
    fd = os.open(path, os.O_RDWR)
    try:
        os.fsync(fd)
        _forget(fd)
    finally:
        os.close(fd)


def _rename(source: str, target: str) -> None:
    os.rename(source, target)


def _copy_checked(source: str, target: str) -> None:
    _copy_file(source, target)
    _flush_and_forget(target)
    if sha256_file(target) != sha256_file(source):
        raise RuntimeError(f"{target}: the copy has a different SHA-256 from {source}")


def copy_run(run: RunSync, share: str, replace: bool, today: str) -> None:
    """Copy the local run to the share by .incoming, and rename it to the target. Raises on a failure. A failed copy
    stays in {share}/.incoming/<rel>. With replace, the old share folder goes to {share}/replaced/<today>/<rel>."""
    incoming = os.path.join(share, INCOMING, run.rel)
    replaced = os.path.join(share, REPLACED, today, run.rel)
    if replace and _exists(replaced):
        raise RuntimeError(f"{replaced} exists; the run is refused")
    if _exists(incoming):
        shutil.rmtree(incoming)  # the only removal: the .incoming folder of this run
    for name in list_files(run.local):
        target = os.path.join(incoming, name)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        _copy_checked(os.path.join(run.local, name), target)
    os.makedirs(incoming, exist_ok=True)
    _copy_checked(os.path.join(run.local, MANIFEST_NAME), os.path.join(incoming, MANIFEST_NAME))
    expected = set(list_files(run.local)) | {MANIFEST_NAME}
    if all_files(incoming) != expected:
        raise RuntimeError(
            f"{incoming} has other files than the local run (extra: {sorted(all_files(incoming) - expected)}, "
            f"missing: {sorted(expected - all_files(incoming))}); the copy stays there"
        )
    if replace:
        if _exists(replaced):
            raise RuntimeError(f"{replaced} exists; the run is refused")
        os.makedirs(os.path.dirname(replaced), exist_ok=True)
        _rename(run.target, replaced)
        try:
            os.makedirs(os.path.dirname(run.target), exist_ok=True)
            _rename(incoming, run.target)
        except OSError as error:
            raise RuntimeError(
                f"the old copy is in {replaced}, and the new copy is in {incoming}; {run.target} does not exist "
                f"now ({error})"
            ) from error
        return
    if _exists(run.target):
        raise RuntimeError(f"{run.target} appeared during the copy (another VM?); the copy stays in {incoming}")
    os.makedirs(os.path.dirname(run.target), exist_ok=True)
    _rename(incoming, run.target)


def write_sites(run: RunSync) -> str:
    """Write the union of the sites into the share manifest. The share manifest is read again just before the write,
    because the class SITES was found a long time before (sim.run on another VM can add a site in that time). Raises
    if the share manifest has changed in another key. Gives the outcome."""
    local_manifest, fresh = read_manifest(run.local), read_manifest(run.target)
    if (
        not isinstance(fresh, dict)
        or fresh.get("run_hash") != local_manifest.get("run_hash")
        or fresh.get("status") != COMPLETE
        or core(fresh) != core(local_manifest)
    ):
        raise RuntimeError("the share manifest has changed since the check; run the script again")
    merged = merge_sites(local_manifest, fresh)
    if merged is None:
        return "no change (the share has the sites now)"
    write_manifest(run.target, merged)
    return "sites written"


def _open_log(directory: str):
    """Open a new log file in directory. A name that exists is not overwritten: it gets a suffix."""
    stamp = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    for number in range(1000):
        name = f"{stamp}.txt" if number == 0 else f"{stamp}-{number}.txt"
        try:
            return open(os.path.join(directory, name), "x")
        except FileExistsError:
            continue
    raise RuntimeError(f"no free log name in {directory}")


def _log_line(log, run: RunSync) -> None:
    notes = "; ".join(run.notes)
    log.write(f"{run.rel}\t{run.cls}\t{run.decision or '-'}\t{run.outcome}\t{notes}\n")
    log.flush()


def select_files() -> List[RunSync]:
    """The forcing CSVs (each one after its .json file) and the melt maps of the local disk, as items to classify.
    rel is "forcing/<path relative to forcing_root()>" or "melt/<name>". The target is not classified yet. An OSError
    is raised (main reports it)."""
    items: List[RunSync] = []
    root = forcing_root()
    if _exists(root):
        for name in sorted(all_files(root)):
            if not name.endswith(".csv"):
                continue
            csv = RunSync(rel=os.path.join("forcing", name), local=os.path.join(root, name), target="")
            csv.target = share_forcing_path(csv.local)
            json_name = os.path.splitext(name)[0] + ".json"
            if _exists(os.path.join(root, json_name)):
                json_file = RunSync(
                    rel=os.path.join("forcing", json_name), local=os.path.join(root, json_name), target=""
                )
                json_file.target = share_forcing_path(json_file.local)
                json_file.json_of = csv
                csv.requires = json_file
                items.append(json_file)  # the .json file is copied before its CSV
            else:
                csv.no_json = True
            items.append(csv)
    melt = os.path.join(config["CFM_data_path"], "melt")
    for path in sorted(glob.glob(os.path.join(melt, "*.nc"))):
        name = os.path.basename(path)
        target = os.path.join(share_data_path(), "melt", name)
        items.append(RunSync(rel=os.path.join("melt", name), local=path, target=target))
    return items


def _csv_sha256(csv: RunSync) -> Optional[str]:
    """The SHA-256 of a forcing CSV: the csv_sha256 of its .json file, so the CSVs that no run uses are not read; else
    (no .json file, no csv_sha256, or a .json file that cannot be read) the SHA-256 of the CSV. None, with a warning,
    if the CSV cannot be read."""
    if csv.requires is not None:
        try:
            with open(csv.requires.local) as json_file:
                metadata = json.load(json_file)
            if isinstance(metadata, dict) and metadata.get("csv_sha256"):
                return metadata["csv_sha256"]
        except (OSError, ValueError):
            pass  # the CSV is read instead
    try:
        return sha256_file(csv.local)
    except (OSError, ValueError) as error:
        print(f"WARNING: {csv.rel} cannot be read ({error}); it is not selected.", file=sys.stderr)
        return None


def files_of_runs(items: List[RunSync], folders: List[str]) -> List[RunSync]:
    """The forcing CSVs of select_files() that the runs in folders used, each after its .json file. A CSV is used by a
    run if its SHA-256 is the forcing_csv_sha256 of the run hash inputs. The melt maps are left out. A run with no
    local forcing CSV (or no forcing_csv_sha256) gives a warning: it is copied without its forcing CSV."""
    used: Dict[str, List[str]] = {}  # SHA-256: the runs that used it
    for folder in folders:
        rel = os.path.relpath(folder, runs_root())
        try:
            manifest = read_manifest(folder)
        except (OSError, ValueError) as error:
            manifest = None
            print(f"WARNING: the manifest of {rel} cannot be read ({error}).", file=sys.stderr)
        inputs = manifest.get("hash_inputs") if isinstance(manifest, dict) else None
        sha = inputs.get("forcing_csv_sha256") if isinstance(inputs, dict) else None
        used.setdefault(sha or "", []).append(rel)
    chosen: List[RunSync] = []
    found = set()
    for csv in items:
        if csv.json_of is None and csv.rel.startswith("forcing" + os.sep):
            sha = _csv_sha256(csv)
            if sha and sha in used:
                chosen += ([csv.requires] if csv.requires is not None else []) + [csv]
                found.add(sha)
    for rel in sorted(rel for sha, rels in used.items() if sha not in found for rel in rels):
        print(f"WARNING: no readable local forcing CSV for {rel}; the run is copied without one.", file=sys.stderr)
    return chosen


def same_forcing_json(local: str, share: str) -> bool:
    """Whether two forcing JSON files describe the same CSV: both parse, and have the same csv_sha256. The other keys
    (created, host, requested_location) are metadata of the fetch, so two VMs that fetch the same forcing differ."""
    try:
        with open(local) as local_file, open(share) as share_file:
            local_json, share_json = json.load(local_file), json.load(share_file)
    except ValueError:
        return False
    return (
        isinstance(local_json, dict)
        and isinstance(share_json, dict)
        and local_json.get("csv_sha256") is not None
        and local_json.get("csv_sha256") == share_json.get("csv_sha256")
    )


def classify_file(item: RunSync) -> None:
    """Give a forcing file or a melt map the class NEW, SAME, CONFLICT or ERROR. An OSError is raised.

    A CSV with no local .json file is ERROR unless the share has its .json file. A .json file that differs from the
    share file is SAME (with a note, and the share file is kept) if its CSV is SAME and the two files have the same
    csv_sha256; else it is CONFLICT, so the CSV of an item must be classified before its .json file.
    """
    if item.no_json and not _exists(os.path.splitext(item.target)[0] + ".json"):
        item.cls = ERROR
        item.notes.append("no .json file next to the CSV, and none on the share")
    elif not _exists(item.target):
        item.cls = NEW
    elif os.path.getsize(item.local) != os.path.getsize(item.target) or sha256_file(item.local) != sha256_file(
        item.target
    ):
        item.cls = CONFLICT
        item.notes.append("the share has a different file with this name; it is never overwritten")
        if item.json_of is not None and item.json_of.cls == SAME and same_forcing_json(item.local, item.target):
            item.cls = SAME
            item.notes[:] = ["the JSON differs only in its metadata; the share JSON is kept"]
    else:
        item.cls = SAME


def classify_files(items: List[RunSync]) -> None:
    """Classify the files (a CSV before its .json file). An item that gets an OSError is ERROR. A .json file whose CSV
    is CONFLICT or ERROR is not copied (item.blocked), so the share never gets a .json file of a refused CSV."""
    for item in sorted(items, key=lambda item: item.json_of is not None):
        try:
            classify_file(item)
        except (OSError, ValueError) as error:
            item.cls = ERROR
            item.notes.append(str(error))
    for item in items:
        if item.json_of is not None and item.json_of.cls in (CONFLICT, ERROR) and item.cls == NEW:
            item.blocked = f"its CSV {item.json_of.rel} is {item.json_of.cls}"


def _publish(incoming: str, target: str) -> None:
    """Make the target from the .incoming file, and remove the .incoming file. os.rename replaces a file that exists,
    so the target is made with os.link, which fails with FileExistsError if another VM made it first. If the mount
    does not support hard links, this checks the target and renames: then a small window remains between the check and
    the rename."""
    try:
        os.link(incoming, target)
    except FileExistsError:
        raise RuntimeError(f"{target} appeared during the copy (another VM?); the copy stays in {incoming}")
    except OSError as error:
        if error.errno not in (errno.EPERM, errno.EOPNOTSUPP, errno.ENOTSUP, errno.EXDEV, errno.ENOSYS):
            raise
        if _exists(target):
            raise RuntimeError(f"{target} appeared during the copy (another VM?); the copy stays in {incoming}")
        _rename(incoming, target)
        return
    os.remove(incoming)


def copy_file_item(item: RunSync, share: str) -> None:
    """Copy a file to the share by {share}/.incoming/<rel>, and make the target from it. Raises on a failure."""
    incoming = os.path.join(share, INCOMING, item.rel)
    if _exists(incoming):
        os.remove(incoming)  # the only removal: the .incoming file of this file
    os.makedirs(os.path.dirname(incoming), exist_ok=True)
    _copy_checked(item.local, incoming)
    os.makedirs(os.path.dirname(item.target), exist_ok=True)
    _publish(incoming, item.target)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Copy the complete local runs, forcing CSVs and melt maps to the share (dry run by default)"
    )
    parser.add_argument("--apply", action="store_true", help="copy the runs and write the log")
    parser.add_argument("--on-differ", choices=[ASK, KEEP, REPLACE, SKIP], default=ASK,
                        help="one decision for all the DIFFER runs (default: ask)")
    parser.add_argument("--runs", action="append", metavar="PATTERN",
                        help="only the runs whose path relative to the run root matches PATTERN (fnmatch), and their "
                             "forcing CSVs; no melt map. Can be given more than once")
    args = parser.parse_args(argv)

    share = share_data_path()
    if share is None:
        print("ERROR: CFM_share_data_path is not set in config.yaml.", file=sys.stderr)
        return 1
    if not share_available():
        print(f"ERROR: the share {share} is not available.", file=sys.stderr)
        return 1

    folders = select_local_runs()
    print(f"{len(folders)} complete local runs in {runs_root()}")
    if args.runs:
        folders = match_runs(folders, args.runs)
        print(f"{len(folders)} of them match --runs {' '.join(args.runs)}")
        if not folders:
            print("ERROR: --runs selects no complete local run. Nothing is done.", file=sys.stderr)
            return 1
    runs: List[RunSync] = []
    for number, folder in enumerate(folders, 1):
        rel = os.path.relpath(folder, runs_root())
        try:
            run = classify_run(folder, os.path.join(share_runs_root(), rel), rel)
        except (OSError, ValueError) as error:
            run = RunSync(rel=rel, local=folder, target=os.path.join(share_runs_root(), rel), cls=ERROR)
            run.notes.append(str(error))
        runs.append(run)
        print(f"[{number}/{len(folders)}] {run.cls:8s} {rel}", flush=True)
    counts = {cls: sum(run.cls == cls for run in runs) for cls in (NEW, SAME, SITES, DIFFER, CONFLICT, ERROR)}
    print("Classes: " + ", ".join(f"{cls} {count}" for cls, count in counts.items()))

    failed = False
    try:
        files = select_files()
        if args.runs:
            files = files_of_runs(files, folders)
    except OSError as error:  # for example an unreadable forcing folder; the runs go on
        print(f"ERROR: the forcing and melt files cannot be listed ({error}); they are not copied.", file=sys.stderr)
        files, failed = [], True
    classify_files(files)
    for number, item in enumerate(files, 1):
        print(f"[file {number}/{len(files)}] {item.cls:8s} {item.rel}", flush=True)
    if files:
        counts = ", ".join(f"{cls} {sum(item.cls == cls for item in files)}" for cls in (NEW, SAME, CONFLICT, ERROR))
        print(f"Files: {counts}")

    for run in runs + files:
        if run.cls in (CONFLICT, ERROR):
            failed = True
            print(f"ERROR: {run.cls} {run.rel}: {'; '.join(run.notes)}", file=sys.stderr)
    differ = [run for run in runs if run.cls == DIFFER]
    for run in differ:
        describe_differ(run)

    if not args.apply:
        print("Dry run: nothing is written. Use --apply to copy.")
        return 1 if failed else 0

    os.makedirs(os.path.join(config["CFM_data_path"], "sync-logs"), exist_ok=True)
    with _open_log(os.path.join(config["CFM_data_path"], "sync-logs")) as log:
        # all the decisions first, then the copies
        for run in differ:
            if args.on_differ != ASK:
                run.decision = args.on_differ
            elif sys.stdin.isatty():
                run.decision = ask_decision(run)
            if not run.decision:
                print(f"ERROR: no decision for {run.rel} (the input is not a terminal, or it ended). "
                      "Use --on-differ. Nothing is copied.", file=sys.stderr)
        if any(not run.decision for run in differ):
            for run in runs + files:
                run.outcome = "no decision: nothing copied" if run.cls == DIFFER else "not done"
                _log_line(log, run)
            return 1

        os.makedirs(share_runs_root(), exist_ok=True)
        for number, item in enumerate(files, 1):
            if item.cls == NEW and item.blocked:
                item.outcome = f"refused: {item.blocked}"
                item.notes.append(item.blocked)
            elif item.cls == NEW:
                print(f"[file {number}/{len(files)}] {item.rel}: writing", flush=True)
                try:
                    if item.requires is not None and item.requires.outcome != "copied" and item.requires.cls != SAME:
                        raise RuntimeError(f"not copied: {item.requires.rel} is not on the share")
                    copy_file_item(item, share)
                    item.outcome = "copied"
                except Exception as error:  # report the file, and go on with the next
                    item.outcome = f"FAILED: {error}"
                    failed = True
                    print(f"ERROR: {item.rel}: {error}", file=sys.stderr)
            else:
                item.outcome = "no change" if item.cls == SAME else "refused"
            _log_line(log, item)
        today = datetime.date.today().isoformat()
        todo = [run for run in runs if run.cls in (NEW, SITES) or run.decision == REPLACE]
        done = 0
        for run in runs:
            if run in todo:
                done += 1
                print(f"[{done}/{len(todo)}] {run.cls} {run.rel}: writing", flush=True)
                try:
                    if run.cls == SITES:
                        run.outcome = write_sites(run)
                    else:
                        copy_run(run, share, run.decision == REPLACE, today)
                        run.outcome = "replaced" if run.decision == REPLACE else "copied"
                except Exception as error:  # report the run, and go on with the next
                    run.outcome = f"FAILED: {error}"
                    failed = True
                    print(f"ERROR: {run.rel}: {error}", file=sys.stderr)
            elif run.cls == DIFFER:
                run.outcome = f"{run.decision}: no change"
            elif run.cls in (CONFLICT, ERROR):
                run.outcome = "refused"
            else:
                run.outcome = "no change"
            _log_line(log, run)
    print(f"Done. {sum(run.outcome.startswith('FAILED') for run in runs + files)} runs or files failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
