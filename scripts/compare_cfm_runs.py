"""Script to compare two CFM output folders, e.g. a regenerated run against a reference run on luna. Prints two
reports:

1. Exact: for each of CFMforcing.hdf5, CFMspin.hdf5 and CFMresults.hdf5, which datasets are identical and, for those
   that differ, the max absolute and relative difference and how many elements differ.
2. Physical: firn air content (DIP), density and temperature at fixed depths, and the depths at which the final
   density profile reaches 550 and 830 kg/m3. This assumes grid_outputs is true, so both runs share a depth grid.

Only reads the two folders, never writes. Does not need config.yaml.

Matrix mode (FEAT-001): --baseline <label> --cfm <label> [<label> ...] finds the runs with find_runs (from the
manifests, so it needs config.yaml) and compares each run of each --cfm version with the baseline run of the same model
point, schemes and years. It prints one row for each pair: the identical datasets of each file, and the physical
differences of the summary(). The filters --site, --model-point, --rcm, --physrho, --liquid and --spin-end-year limit
the runs. If more than one run of a version agrees, it uses the run that finished last (as find_run).

A different spin_end_year gives different initial conditions, so the difference of such a pair contains the effect of
the spin-up and of the CFM version together (FEAT-004). Each row gives spin_end_year and baseline_spin_end_year, and
the note "spin-up differs" if they are different. The key of a pair (PAIR_KEY) does not contain spin_end_year, so such
a pair is still compared. Give --spin-end-year to compare the runs of one spin-up only.

Usage:
    - python scripts/compare_cfm_runs.py <new_output_folder> <reference_output_folder>
    - python scripts/compare_cfm_runs.py --baseline 3e3bf7e-speersm --cfm ab871c8 [--site CI-0] [--rcm RACMO] [--csv out.csv]
    - python scripts/compare_cfm_runs.py --baseline 3e3bf7e-speersm --cfm ab871c8 --spin-end-year 1989
"""

import argparse
import os
from typing import Dict

import h5py
import numpy as np

CFM_FILES = ["CFMforcing.hdf5", "CFMspin.hdf5", "CFMresults.hdf5"]
COMPARE_DEPTHS = [1, 5, 10, 20, 40]  # m
COMPARE_DENSITIES = [550, 830]  # kg/m3
# the Run properties that make a pair of the matrix mode. spin_end_year is not in the key: the row shows it (FEAT-004)
PAIR_KEY = ("model_point", "physRho", "liquid", "start_year", "end_year")


def datasets(h5_file: h5py.File) -> Dict[str, h5py.Dataset]:
    """All datasets in an HDF5 file, including those nested in groups, keyed by path."""
    found = {}
    h5_file.visititems(lambda name, obj: found.__setitem__(name, obj) if isinstance(obj, h5py.Dataset) else None)
    return found


def compare_exact(new_dir: str, ref_dir: str) -> None:
    """Print which datasets are identical in each CFM output file, and summarise differences for those that are not."""
    for file_name in CFM_FILES:
        missing = [d for d in (new_dir, ref_dir) if not os.path.exists(f"{d}/{file_name}")]
        if missing:
            print(f"== {file_name}: MISSING in {missing} (did the run crash?)")
            continue
        with h5py.File(f"{new_dir}/{file_name}", "r") as new_file, h5py.File(f"{ref_dir}/{file_name}", "r") as ref_file:
            new_sets, ref_sets = datasets(new_file), datasets(ref_file)
            print(f"== {file_name}: {len(new_sets)} datasets")
            if new_sets.keys() != ref_sets.keys():
                print(f"  only in new: {sorted(new_sets.keys() - ref_sets.keys())}")
                print(f"  only in reference: {sorted(ref_sets.keys() - new_sets.keys())}")

            common = sorted(new_sets.keys() & ref_sets.keys())
            n_identical = 0
            for name in common:
                new, ref = new_sets[name][()], ref_sets[name][()]
                if new.shape != ref.shape:
                    print(f"  {name}: SHAPE {new.shape} vs {ref.shape}")
                    continue
                equal_nan = new.dtype.kind not in "SOU"  # equal_nan is not defined for string arrays
                if np.array_equal(new, ref, equal_nan=equal_nan):
                    n_identical += 1
                    continue
                diff = np.abs(new.astype(float) - ref.astype(float))
                rel = diff / np.maximum(np.abs(ref.astype(float)), 1e-12)
                print(
                    f"  {name}: DIFFERS max|d|={np.nanmax(diff):.3g} max_rel={np.nanmax(rel):.3g} "
                    f"n_diff={np.count_nonzero(diff > 0)}/{diff.size}"
                )
            print(f"  identical: {n_identical}/{len(common)}")


def summary(new_dir: str, ref_dir: str) -> dict:
    """The numbers of one comparison, for the matrix mode: identical datasets per file and the physical differences."""
    row = {}
    for file_name in CFM_FILES:
        key = file_name.replace(".hdf5", "")
        if not all(os.path.exists(f"{d}/{file_name}") for d in (new_dir, ref_dir)):
            row[key] = "missing"
            continue
        with h5py.File(f"{new_dir}/{file_name}", "r") as new_file, h5py.File(f"{ref_dir}/{file_name}", "r") as ref_file:
            new_sets, ref_sets = datasets(new_file), datasets(ref_file)
            common = sorted(new_sets.keys() & ref_sets.keys())
            identical = 0
            for name in common:
                new, ref = new_sets[name][()], ref_sets[name][()]
                equal_nan = new.dtype.kind not in "SOU"
                identical += new.shape == ref.shape and np.array_equal(new, ref, equal_nan=equal_nan)
            row[key] = f"{identical}/{len(set(new_sets) | set(ref_sets))}"
    row["bit_for_bit"] = all(
        isinstance(value, str) and value != "missing" and value.split("/")[0] == value.split("/")[1]
        for key, value in row.items()
    )
    if all(os.path.exists(f"{d}/CFMresults.hdf5") for d in (new_dir, ref_dir)):
        with h5py.File(f"{new_dir}/CFMresults.hdf5", "r") as new, h5py.File(f"{ref_dir}/CFMresults.hdf5", "r") as ref:
            if np.array_equal(new["depth"][()], ref["depth"][()]):
                depth = new["depth"][1:]
                rho_new, rho_ref = new["density"][-1, 1:], ref["density"][-1, 1:]
                row["fac_final_diff_m"] = float(new["DIP"][-1, 1] - ref["DIP"][-1, 1])
                row["density_final_mean_abs_diff"] = float(np.nanmean(np.abs(rho_new - rho_ref)))
                for rho in COMPARE_DENSITIES:
                    row[f"depth_{rho}_diff_m"] = float(depth[np.argmax(rho_new >= rho)] - depth[np.argmax(rho_ref >= rho)])
                col = 1 + np.argmin(np.abs(depth - COMPARE_DEPTHS[-1]))
                row[f"temp_{COMPARE_DEPTHS[-1]}m_mean_diff_K"] = float(
                    np.nanmean(new["temperature"][1:, col] - ref["temperature"][1:, col])
                )
            else:
                row["note"] = "depth grids differ"
    return row


def matrix(baseline: str, versions: list, filters: dict) -> list:
    """Compare the runs of each version with the baseline runs of the same model point, schemes and years."""
    from sim.runs import find_runs

    def key(run):
        return tuple(getattr(run, name) for name in PAIR_KEY)

    def latest(runs):
        by_key = {}
        for run in sorted(runs, key=lambda run: run.finished):
            by_key[key(run)] = run  # the last to finish
        return by_key

    base = latest(find_runs(cfm=baseline, **filters))
    rows = []
    for version in versions:
        other = latest(find_runs(cfm=version, **filters))
        for k in sorted(base.keys() | other.keys(), key=lambda k: tuple(str(value) for value in k)):
            row = dict(zip(PAIR_KEY, k), cfm=version)
            if k in other:
                row["spin_end_year"] = other[k].spin_end_year
            if k in base:
                row["baseline_spin_end_year"] = base[k].spin_end_year
            if k not in other or k not in base:
                row["note"] = f"no run of {version if k not in other else baseline}"
            else:
                row["run"], row["baseline_run"] = other[k].run_hash[:8], base[k].run_hash[:8]
                row.update(summary(other[k].folder, base[k].folder))
                if other[k].spin_end_year != base[k].spin_end_year:
                    row["note"] = "; ".join(filter(None, ["spin-up differs", row.get("note")]))
            rows.append(row)
    return rows


def compare_physical(new_dir: str, ref_dir: str) -> None:
    """Print differences in firn air content, density and temperature between two CFMresults.hdf5 files.

    Row 0 of the gridded outputs holds the depth grid and column 0 holds model time, so both are skipped.
    """
    if not all(os.path.exists(f"{d}/CFMresults.hdf5") for d in (new_dir, ref_dir)):
        print("CFMresults.hdf5 missing, skipping the physical comparison.")
        return
    with (
        h5py.File(f"{new_dir}/CFMresults.hdf5", "r") as new,
        h5py.File(f"{ref_dir}/CFMresults.hdf5", "r") as ref,
    ):
        if not np.array_equal(new["depth"][()], ref["depth"][()]):
            print("Depth grids differ, so a physical comparison at fixed depths is not meaningful.")
            return
        depth = new["depth"][1:]

        fac_new, fac_ref = new["DIP"][1:, 1], ref["DIP"][1:, 1]
        print("== Firn air content (DIP column 1)")
        print(f"  mean:  new {fac_new.mean():.2f} m, reference {fac_ref.mean():.2f} m")
        print(f"  final: new {fac_new[-1]:.2f} m, reference {fac_ref[-1]:.2f} m")
        print(f"  max |new - reference|: {np.abs(fac_new - fac_ref).max():.2f} m")

        print("== At fixed depths, over all timesteps (new - reference)")
        print("  depth | density mean / max|d| (kg/m3) | temperature mean / max|d| (K)")
        for target in COMPARE_DEPTHS:
            col = 1 + np.argmin(np.abs(depth - target))
            d_rho = new["density"][1:, col] - ref["density"][1:, col]
            d_temp = new["temperature"][1:, col] - ref["temperature"][1:, col]
            print(
                f"  {target:4d} m | {np.nanmean(d_rho):+7.2f} / {np.nanmax(np.abs(d_rho)):6.1f}"
                f" | {np.nanmean(d_temp):+6.3f} / {np.nanmax(np.abs(d_temp)):5.2f}"
            )

        rho_new, rho_ref = new["density"][-1, 1:], ref["density"][-1, 1:]
        print("== Final density profile")
        print(f"  mean |new - reference|: {np.nanmean(np.abs(rho_new - rho_ref)):.2f} kg/m3")
        for rho in COMPARE_DENSITIES:
            print(
                f"  depth of {rho} kg/m3: new {depth[np.argmax(rho_new >= rho)]:.2f} m, "
                f"reference {depth[np.argmax(rho_ref >= rho)]:.2f} m"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two CFM output folders, or the runs of CFM versions")
    parser.add_argument("new_dir", nargs="?", help="Output folder of the new run")
    parser.add_argument("ref_dir", nargs="?", help="Output folder of the reference run")
    parser.add_argument("--baseline", help="Matrix mode: the label of the baseline CFM version")
    parser.add_argument("--cfm", nargs="+", help="Matrix mode: the labels of the CFM versions to compare")
    parser.add_argument("--site")
    parser.add_argument("--model-point")
    parser.add_argument("--rcm")
    parser.add_argument("--physrho")
    parser.add_argument("--liquid")
    parser.add_argument("--spin-end-year", type=int, help="Matrix mode: only the runs of this spin_end_year")
    parser.add_argument("--csv", help="Matrix mode: also write the table to this CSV file")
    args = parser.parse_args()

    if args.baseline:
        if not args.cfm:
            parser.error("--baseline needs --cfm")
        import pandas as pd

        filters = {
            "site": args.site,
            "model_point": args.model_point,
            "rcm_name": args.rcm,
            "physRho": args.physrho,
            "liquid": args.liquid,
            "spin_end_year": args.spin_end_year,
        }
        table = pd.DataFrame(matrix(args.baseline, args.cfm, filters))
        with pd.option_context("display.max_rows", None, "display.width", 250, "display.float_format", "{:.3g}".format):
            print(table.to_string(index=False))
        if args.csv:
            table.to_csv(args.csv, index=False)
    else:
        if not (args.new_dir and args.ref_dir):
            parser.error("give two folders, or --baseline and --cfm")
        compare_exact(args.new_dir, args.ref_dir)
        print()
        compare_physical(args.new_dir, args.ref_dir)
