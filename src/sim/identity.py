"""Measures what a run actually ran (FEAT-001): the CFM's git commit and local changes, its Python environment, and the
commit of this repo. The cfm_versions label in config.yaml is only a name; this is what goes in the run manifest.
"""

import hashlib
import logging
import os
import re
import subprocess

# this repo's own package in pip freeze. Its line carries this repo's commit, which is recorded separately
_OWN_PACKAGE = re.compile(r"^(-e .*#egg=cfm|cfm( @ |==).*)$")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run(*args: str) -> str:
    # without PYTHONPATH, so pip freeze describes the CFM version's own environment. With this repo's src on it (as
    # sim.worker runs), pip also reads any stale src/cfm.egg-info, and fails on its metadata
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    return subprocess.run(args, capture_output=True, text=True, check=True, env=env).stdout


def changed_paths(status: str) -> list[str]:
    """Paths in the output of git status --porcelain (the new path, for a rename)."""
    return [line[3:].split(" -> ")[-1] for line in status.splitlines()]


def measure_cfm_identity(version: dict) -> dict:
    """Identity of a CFM version from cfm_version(): its checkout and its Python environment."""
    repo, python = version["repo"], version["python"]
    try:
        commit = _run("git", "-C", repo, "rev-parse", "HEAD").strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise RuntimeError(f"CFM repo {repo} of version {version['label']!r} is not a git checkout.") from error
    status = _run("git", "-C", repo, "status", "--porcelain", "--untracked-files=all")
    diff = _run("git", "-C", repo, "diff", "HEAD", "--binary")
    # the diff of the .py files only (":/*.py" is every .py file from the top of the checkout): the run hash uses
    # this one, so a change to example output in the checkout does not give a new run
    code_diff = _run("git", "-C", repo, "diff", "HEAD", "--binary", "--", ":/*.py")
    pip_freeze = [line for line in _run(python, "-m", "pip", "freeze").splitlines() if not _OWN_PACKAGE.match(line)]

    return {
        **version,
        "commit": commit,
        "status": status.splitlines(),
        "code_changes": [path for path in changed_paths(status) if path.endswith(".py")],
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "code_diff_sha256": hashlib.sha256(code_diff.encode()).hexdigest(),
        "python_version": _run(python, "-c", "import sys; print(sys.version)").strip(),
        "pip_freeze": pip_freeze,
    }


def check_cfm_identity(identity: dict) -> None:
    """Refuse a CFM checkout with uncommitted code changes."""
    repo, code = identity["repo"], identity["code_changes"]
    if code:
        raise RuntimeError(
            f"CFM checkout {repo} has uncommitted code changes: {', '.join(code)}. "
            "Commit them to a branch of the CFM checkout first."
        )

    other = [path for path in changed_paths("\n".join(identity["status"])) if path not in code]
    if other:
        logging.warning(f"CFM checkout {repo} has uncommitted changes to files that are not code: {', '.join(other)}.")


# the version of the run code (src/ without RUN_CODE_EXCLUDED), in the run hash in place of a hash of the code
# (FEAT-005). Version 1 is the run code at dev b89d566 and d53fd13. Increase it in the same commit as a change that can
# change the results of a CFM run: the forcing that sim.run gives to the CFM (units, signs, column maps, the spin-up),
# the cfm_config that sim.run makes, or a default value that a run uses. Do not increase it for comments, docstrings,
# log messages, names of local variables, a refactor that gives the same values, or code that a CFM run does not
# import (src/plot/, src/sim/runs.py, scripts/). If you are not sure, increase it: a new version costs only new runs,
# but an old version for a different result gives incorrect results. The drift warning of sim.run tells you when the
# run code changed since the last complete run of this version
RUN_CODE_VERSION = 1

# the files and folders of src/ that a CFM run does not import (tests/test_identity.py checks this). They are not in
# run_code_sha256, so a change to them gives no drift warning, and they can have uncommitted changes
RUN_CODE_EXCLUDED = ("src/plot/", "src/sim/runs.py")


def _is_run_code(path: str, excluded: tuple[str, ...] = RUN_CODE_EXCLUDED) -> bool:
    return path.startswith("src/") and not path.startswith(excluded)


def run_code_sha256(commit: str = "HEAD", excluded: tuple[str, ...] = RUN_CODE_EXCLUDED) -> str:
    """SHA-256 of the git ls-tree list (the path and blob hash of each file) of the run code at commit: src/ without
    excluded. scripts/rekey_runs.py also gives the rule of hash_version 2, src/ without src/plot/ (FEAT-005)."""
    tree = _run("git", "-C", _REPO_ROOT, "ls-tree", "-r", "--full-tree", commit, "src")
    run_code = [line for line in tree.splitlines() if _is_run_code(line.split("\t", 1)[1], excluded)]
    return hashlib.sha256("\n".join(run_code).encode()).hexdigest()


def wrapper_identity() -> dict | None:
    """Commit, local changes and the run code of this repo, or None if it is not a git checkout.

    The run code is src/ without RUN_CODE_EXCLUDED, at HEAD. The run hash uses run_code_version (RUN_CODE_VERSION), not
    the code itself, so only a new version gives new runs (FEAT-005). run_code_sha256 is the SHA-256 of the git
    ls-tree list of the run code (the path and blob hash of each file). It is recorded only, like the commit and
    src_tree (git rev-parse HEAD:src), and sim.run warns if it differs from that of the last run of the version.
    src_code_changes are the uncommitted .py changes in the run code, which check_wrapper_identity() refuses, so that
    the recorded commit always gives the code of the run.
    """
    try:
        commit = _run("git", "-C", _REPO_ROOT, "rev-parse", "HEAD").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    status = _run("git", "-C", _REPO_ROOT, "status", "--porcelain")
    src_status = _run("git", "-C", _REPO_ROOT, "status", "--porcelain", "--untracked-files=all", "--", ":/src")
    return {
        "path": _REPO_ROOT,
        "commit": commit,
        "status": status.splitlines(),
        "src_tree": _run("git", "-C", _REPO_ROOT, "rev-parse", "HEAD:src").strip(),
        "run_code_version": RUN_CODE_VERSION,
        "run_code_sha256": run_code_sha256(commit),
        "src_code_changes": [
            path for path in changed_paths(src_status) if path.endswith(".py") and _is_run_code(path)
        ],
    }


def check_wrapper_identity(identity: dict | None) -> None:
    """Refuse a run if this repo is not a git checkout, or if the run code has uncommitted .py changes (FEAT-001 Q4)."""
    if identity is None:
        raise RuntimeError(f"{_REPO_ROOT} is not a git checkout, so the run hash cannot include the code in src/.")
    if identity["src_code_changes"]:
        raise RuntimeError(
            f"{identity['path']} has uncommitted code changes in src/: {', '.join(identity['src_code_changes'])}. "
            "Commit them first, so that the manifest records the code of the run."
        )
