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
    """Refuse a CFM checkout with uncommitted code changes, unless its version sets allow_uncommitted."""
    repo, code = identity["repo"], identity["code_changes"]
    if code:
        message = f"CFM checkout {repo} has uncommitted code changes: {', '.join(code)}."
        if not identity["allow_uncommitted"]:
            raise RuntimeError(f"{message} Commit them to a branch of the CFM checkout first.")
        logging.warning(f"{message} Allowed, as version {identity['label']!r} sets allow_uncommitted.")

    other = [path for path in changed_paths("\n".join(identity["status"])) if path not in code]
    if other:
        logging.warning(f"CFM checkout {repo} has uncommitted changes to files that are not code: {', '.join(other)}.")


def wrapper_identity() -> dict | None:
    """Commit, local changes and src/ tree of this repo, or None if it is not a git checkout.

    src_tree is the git tree hash of src/ at HEAD (git rev-parse HEAD:src). The run hash uses it, not the commit, so a
    commit that changes only docs, scripts or tests gives the same run hash. src_code_changes are the uncommitted .py
    changes in src/, which check_wrapper_identity() refuses, as src_tree does not include them.
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
        "src_code_changes": [path for path in changed_paths(src_status) if path.endswith(".py")],
    }


def check_wrapper_identity(identity: dict | None) -> None:
    """Refuse a run if this repo is not a git checkout, or if src/ has uncommitted .py changes (FEAT-001 Q4)."""
    if identity is None:
        raise RuntimeError(f"{_REPO_ROOT} is not a git checkout, so the run hash cannot include the code in src/.")
    if identity["src_code_changes"]:
        raise RuntimeError(
            f"{identity['path']} has uncommitted code changes in src/: {', '.join(identity['src_code_changes'])}. "
            "Commit them first, so that the run hash includes them."
        )
