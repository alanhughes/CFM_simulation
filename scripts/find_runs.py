"""List the CFM runs that agree with the filters, from their manifests (FEAT-001).

Usage:
    python scripts/find_runs.py                                   # all the complete runs
    python scripts/find_runs.py --site CI-0 --physrho GSFC2020 --cfm ab871c8
    python scripts/find_runs.py --cfm 3e3bf7e-speersm --spin-end-year 1995   # the runs of one spin-up (FEAT-004)
    python scripts/find_runs.py --cfm-config SEB=false            # the runs with cfm_config SEB false (FEAT-009)
    python scripts/find_runs.py --status all                      # also the running and failed runs
    python scripts/find_runs.py --index                           # also write index.csv in {CFM_share_data_path}, or in {CFM_data_path}
    python scripts/find_runs.py --index runs.parquet              # the same table as parquet (needs pyarrow)

The index is a CSV export of the table for the user. The code does not read it. The column location is "local",
"share" or "both" (FEAT-008). --index with no file name writes index.csv in {CFM_share_data_path} if it is set, else in
{CFM_data_path}. The script writes the table to <file>.tmp in the same folder, then renames it to <file>, so a write
that stops leaves the old index. If CFM_share_data_path is set and the share is not available, --index writes nothing
and the exit code is 1 (the index would not have the share runs). This is checked before the query. The exit code is
also 1 for a file name that is not .csv or .parquet, for a .parquet file name without pyarrow, for a write that
fails, and for an OSError of the query (for example, the share stops during the query).
"""

import argparse
import json
import os
import sys
from typing import List, Optional

import pandas as pd

from cfm_config import config, share_available, share_data_path
from sim.runs import find_runs, runs_table


def default_index_path() -> str:
    """index.csv in CFM_share_data_path if it is set, else in CFM_data_path."""
    return os.path.join(share_data_path() or config["CFM_data_path"], "index.csv")


def write_index(table: pd.DataFrame, path: str) -> None:
    """Write the table to <path>.tmp, fsync it, and rename it to path. The old file stays if the write stops."""
    tmp = f"{path}.tmp"
    try:
        if path.endswith(".parquet"):
            table.to_parquet(tmp, index=False)
        else:
            table.to_csv(tmp, index=False)
        with open(tmp, "rb") as file:
            os.fsync(file.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def cfm_config_item(text: str) -> tuple:
    """KEY=VALUE of --cfm-config: VALUE is read as JSON (SEB=false gives False, SEB_TL_thick=0.05 gives 0.05), or is
    the text if it is not JSON (meltwater_solver=ncz)."""
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"give KEY=VALUE, not {text!r}")
    try:
        return key, json.loads(value)
    except ValueError:
        return key, value


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="List the CFM runs that agree with the filters")
    parser.add_argument("--site", help="Site name in sites.yaml (the runs of its model point)")
    parser.add_argument("--model-point", help="ModelPoint.tag, for example RACMO_261_126")
    parser.add_argument("--rcm")
    parser.add_argument("--physrho")
    parser.add_argument("--liquid")
    parser.add_argument("--meltwater-solver", help="cfm_config meltwater_solver (a run without the key: enthalpy)")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--spin-end-year", type=int, help="run.spin_end_year in the manifest (no filter if not given)")
    parser.add_argument("--cfm", help="Label of the CFM version")
    parser.add_argument("--hash", help="Run hash, or its first characters")
    parser.add_argument(
        "--cfm-config",
        type=cfm_config_item,
        action="append",
        metavar="KEY=VALUE",
        help="only the runs with this cfm_config value; VALUE is JSON, for example SEB=false. Give it more than one time "
        "for more keys. A run whose value is not known does not agree (FEAT-009)",
    )
    parser.add_argument("--status", default="complete", help="complete (default), running, failed or all")
    parser.add_argument(
        "--index",
        nargs="?",
        const="",  # no file name: default_index_path()
        help="Write the table to this .csv or .parquet file (default: index.csv in {CFM_share_data_path} if set, "
        "else in {CFM_data_path})",
    )
    args = parser.parse_args(argv)

    index = default_index_path() if args.index == "" else args.index
    if index:
        if not index.endswith((".csv", ".parquet")):
            print(f"Cannot write {index}: give a .csv or .parquet file name.", file=sys.stderr)
            return 1
        if index.endswith(".parquet"):
            try:
                import pyarrow  # noqa: F401
            except ImportError:
                print(f"Cannot write {index}: pyarrow is not installed. Use a .csv file name.", file=sys.stderr)
                return 1
        if share_data_path() and not share_available():
            print(
                f"ERROR: the share {share_data_path()} is not available, so the index would not have the share runs. "
                "Nothing is written.",
                file=sys.stderr,
            )
            return 1

    try:
        runs = find_runs(
            site=args.site,
            model_point=args.model_point,
            rcm_name=args.rcm,
            physRho=args.physrho,
            liquid=args.liquid,
            meltwater_solver=args.meltwater_solver,
            start_year=args.start_year,
            end_year=args.end_year,
            spin_end_year=args.spin_end_year,
            cfm=args.cfm,
            run_hash=args.hash,
            status=None if args.status == "all" else args.status,
            cfm_config=dict(args.cfm_config) if args.cfm_config else None,
        )
    except OSError as error:  # the share stopped after the check of run_roots() (FEAT-008)
        print(f"ERROR: cannot read the runs ({error})", file=sys.stderr)
        return 1
    table = runs_table(runs)
    with pd.option_context("display.max_rows", None, "display.width", 200, "display.max_colwidth", 40):
        print(table.drop(columns=["folder", "started"]).assign(run_hash=table["run_hash"].str[:8]).to_string(index=False))
    print(f"{len(table)} runs.")

    if index:
        try:
            write_index(table, index)
        except Exception as error:
            print(f"Cannot write {index}: {error}", file=sys.stderr)
            return 1
        print(f"Wrote {index}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
