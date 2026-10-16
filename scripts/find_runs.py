"""List the CFM runs that agree with the filters, from their manifests (FEAT-001).

Usage:
    python scripts/find_runs.py                                   # all the complete runs
    python scripts/find_runs.py --site CI-0 --physrho GSFC2020 --cfm ab871c8
    python scripts/find_runs.py --status all                      # also the running and failed runs
    python scripts/find_runs.py --index                           # also write {CFM_data_path}/index.parquet
    python scripts/find_runs.py --index runs.csv                  # the same table as a CSV file

The index is an export of the table for the user. The code does not read it.
"""

import argparse
import os

import pandas as pd

from cfm_config import config
from sim.runs import find_runs, runs_table

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List the CFM runs that agree with the filters")
    parser.add_argument("--site", help="Site name in sites.yaml (the runs of its model point)")
    parser.add_argument("--model-point", help="ModelPoint.tag, for example RACMO_261_126")
    parser.add_argument("--rcm")
    parser.add_argument("--physrho")
    parser.add_argument("--liquid")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--cfm", help="Label of the CFM version")
    parser.add_argument("--hash", help="Run hash, or its first characters")
    parser.add_argument("--status", default="complete", help="complete (default), running, failed or all")
    parser.add_argument(
        "--index",
        nargs="?",
        const=os.path.join(config["CFM_data_path"], "index.parquet"),
        help="Write the table to this .parquet or .csv file (default: {CFM_data_path}/index.parquet)",
    )
    args = parser.parse_args()

    runs = find_runs(
        site=args.site,
        model_point=args.model_point,
        rcm_name=args.rcm,
        physRho=args.physrho,
        liquid=args.liquid,
        start_year=args.start_year,
        end_year=args.end_year,
        cfm=args.cfm,
        run_hash=args.hash,
        status=None if args.status == "all" else args.status,
    )
    table = runs_table(runs)
    with pd.option_context("display.max_rows", None, "display.width", 200, "display.max_colwidth", 40):
        print(table.drop(columns=["folder", "started"]).assign(run_hash=table["run_hash"].str[:8]).to_string(index=False))
    print(f"{len(table)} runs.")

    if args.index:
        if args.index.endswith(".csv"):
            table.to_csv(args.index, index=False)
        else:
            table.to_parquet(args.index, index=False)  # needs pyarrow or fastparquet
        print(f"Wrote {args.index}.")
