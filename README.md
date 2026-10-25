# CFM_simulation
Column and grid simulation from the Community Firn Model (CFM). Relies on functionality from the CFM repository https://github.com/UWGlaciology/CommunityFirnModel.

## Setup

There is no build step — this is pure Python. Three things need to be in place.

**1. The CFM checkout.** This repo does not vendor or pip-install the CFM. Clone the CFM alongside this repo and list it under `cfm_versions` in `config.yaml`: a label, the checkout's `CFM_main` folder (not the repo root) and the Python to run it with. You can list several versions and pick one with `run_cfm.py --cfm <label>`. Each run starts a subprocess (`sim.worker`) with that version's Python, which appends the `CFM_main` folder to `sys.path` and imports `RCMpkl_to_spin` and `firn_density_nospin` as flat modules. That is why one process can only use one CFM.

The label is only a name. Each run's `manifest.json` records the CFM's measured commit, its uncommitted changes, the Python version and `pip freeze`. A checkout with uncommitted `.py` changes is refused, so commit local patches to a branch of the CFM checkout. For example, the speersm reference version is 3e3bf7e plus speersm's SEB/melt patch, committed on a local branch `speersm-3e3bf7e` (see `config.example.yaml`).

**2. The environment.** Python >=3.10 with the CFM's own dependency set:

```bash
conda create -n cfm python=3.12
conda activate cfm
pip install -e .
```

The editable install is required — the scripts import `force`, `sim` and `plot` as top-level packages from `src/`.

**3. `config.yaml`.** All paths live here. It is gitignored, so create it from the tracked template and fill in the paths for your machine:

```bash
cp config.example.yaml config.yaml
```

The keys that are machine-specific are the directory block at the top:

| Key | Meaning |
| --- | --- |
| `cpom_mount` | Where the luna CPOM share is mounted (see below) — absolute path, `~` is not expanded |
| `cfm_versions` / `default_cfm_version` | The CFM checkouts (each `CFM_main` folder) with the Python for each, and the one to use by default |
| `CFM_data_path` | Where the wrapper writes forcing CSVs, model output and figures — a local folder or your own folder on the share (see below) |
| `MAR_data_path` / `RACMO_data_path` / `RACMO23_data_path` | Where the raw RCM NetCDF forcing files are read from |
| `in_situ_data_path` | Where the in situ depth-density CSVs are read from |

Everything below that block (densification options, RCM column mappings, plot limits) is machine-independent. The sites are not in `config.yaml`: see [Sites](#sites).

## Data on the luna share

The canonical way to run this repo is on a Lancaster University VM. There the luna CPOM share is mounted automatically at `/home/<username>/luna/CPOM`, so set:

```yaml
cpom_mount: /home/<username>/luna/CPOM
```

The RCM forcing and in situ data are read through it, and the default `*_data_path` entries in `config.example.yaml` work as-is once `cpom_mount` is set.

### What you can write to

The share holds other people's data, including the previous user's runs under `speersm/`. Server-side permissions are the guard rail: you only have read/write access in two places on it.

- `<username>/` — your own folder at the top level of CPOM (the same `<username>` as in the mount path).
- `archive/` — shared reference data (the RACMO forcing lives here). It is writable, but treat it as read-only: the code never needs to write there.

Everywhere else (other users' folders such as `speersm/` or `briggsk/`) is read-only to you, so it can't be overwritten by accident.

`CFM_data_path` is where every file the code writes goes — forcing CSVs, CFM output, figures, melt files. Point it at a local folder on the VM (e.g. `/home/<username>/CFM_data`) or at your own folder on the share (`{cpom_mount}/<username>/CFM_data`). Never point it inside `archive/`.

Nothing needs to be copied from `speersm/`: the model regenerates everything there except the in situ data, which has been copied to `hughes54/CFM_data/in_situ`.

### Alternative: mounting from a Mac over SMB

If you're working off a Mac rather than a VM, mount the share yourself. The same server permissions apply, but mounting **read-only** adds a second guard rail (in particular for `archive/`), and output then has to be written to a local `CFM_data_path` and copied up as a separate, deliberate step.

Mount in your home folder: `/Volumes` is owned by root, so `mount_smbfs` can't create a mount point there without admin rights.

```bash
diskutil unmount /Volumes/cpom      # only if Finder has already mounted the share (read-write)
mkdir -p ~/mnt/cpom
mount_smbfs -o rdonly //<username>@luna.lancaster.ac.uk/fst/lec/cpom ~/mnt/cpom
```

Set `cpom_mount: /Users/<you>/mnt/cpom`. It may prompt for your luna password. Check the mount really is read-only before running anything:

```bash
mount | grep cpom                                  # should list "read-only"
touch ~/mnt/cpom/<username>/CFM_data/test_write    # should fail: "Read-only file system"
```

The mount does not reconnect itself after sleep or a network drop. If reads from the share start failing with I/O errors, run `diskutil unmount ~/mnt/cpom` and the `mount_smbfs` line again.

To copy results up, remount read-write only for the copy, then go straight back to read-only:

```bash
diskutil unmount ~/mnt/cpom
mount_smbfs //<username>@luna.lancaster.ac.uk/fst/lec/cpom ~/mnt/cpom           # read-write

rsync -avn /path/to/local/CFM_data/ ~/mnt/cpom/<username>/CFM_data/    # dry run: check the file list
rsync -av  /path/to/local/CFM_data/ ~/mnt/cpom/<username>/CFM_data/

diskutil unmount ~/mnt/cpom
mount_smbfs -o rdonly //<username>@luna.lancaster.ac.uk/fst/lec/cpom ~/mnt/cpom  # back to read-only
```

Keep the trailing `/` on both `rsync` paths, so the folder's contents are copied rather than a nested `CFM_data/CFM_data`. Never add `--delete`: without it `rsync` only adds and updates files under `<username>/CFM_data` and touches nothing else on the share. Don't run any scripts while the share is read-write.

## Running

**All scripts must be run from the repo root** — `src/cfm_config.py` opens `config.yaml` by relative path. It also expands `{cpom_mount}` in the top-level paths, so if luna is mounted somewhere else on your machine, only `cpom_mount` needs changing.

```bash
python scripts/run_cfm.py                                   # uses config.yaml defaults (default_site, ...)
python scripts/run_cfm.py --site CI-0 \
    --physrho GSFC2020 --rcm RACMO --liquid bucket          # any argument left out comes from config.yaml
```

**Years.** `start_year` and `end_year` in `config.yaml` are the default years of every RCM, but the end year is clipped to the last complete year of the RCM's data, found from the data file names (DISC-001). The ANT-12 `RACMO` data stops at 2023-12-31, so with `end_year: 2024` a `RACMO` run is 1979-2023, and `MAR` and `RACMO23` (data to 2024-12-31) runs are 1979-2024. Each clip logs a warning such as `RACMO (ANT-12_...) data stops at 2023-12-31: end_year 2023, not 2024 from config.yaml`. Only the end is clipped: data that starts after `start_year`, or a clipped end year before `spin_end_year`, stops with an error, and a missing year inside the range still stops the forcing (BUG-003). `--start-year` and `--end-year` (on `run_cfm.py`, `extract_forcing.py`, `plot_cfm.py`, `calc_dh.py`, `plot_density_error.py`, `plot_rcm_ts.py`, `run_melt.py` and `plot_melt.py`) are used as they are and never clipped; an end year after the data is an error. `run_cfm.py` resolves the years once (`force.years.run_years`) and gives them to the worker process, and the plot and analysis scripts resolve them the same way, so they find the run with no year argument. The manifest records the end year that was asked for in `run.requested_end_year` and `run.end_year_source` (`config` or `cli`); the run hash and the folder name have the end year of the run.

`run_cfm.py` processes the RCM forcing of the site's model point into the forcing cache (see below), then runs the CFM, writing to

```
{CFM_data_path}/runs/<model point>/<physrho>_<liquid>_<start>-<end>/<cfm label>/<hash8>/
```

`<model point>` is the RCM grid cell, e.g. `RACMO_261_126`, so two sites in one cell share a run. `<hash8>` is the start of the **run hash**: the SHA-256 of everything that changes the results - all of `cfm_config`, `start_year`/`end_year`/`spin_end_year`/`sublim_sf`/`rho_bottom`, the model point and RCM release, the SHA-256 of the forcing CSV, the CFM's commit, `.py` patch, Python version and `pip freeze`, and the version of this repo's run code, `RUN_CODE_VERSION` in `src/sim/identity.py` (FEAT-005). Changing any of them gives a new folder. The run code itself is not hashed: a commit to this repo gives new runs only if it increases `RUN_CODE_VERSION`. The manifest still records the exact code in `wrapper`: the commit, the status, the `src/` tree, `run_code_version` and `run_code_sha256` (the hash of the committed run code, `src/` without `src/plot/` and `src/sim/runs.py`, which a run does not import). Uncommitted `.py` changes in the run code are refused, like those in the CFM, so the recorded commit always gives the code of the run.

**When to increase `RUN_CODE_VERSION`.** Increase it, in the same commit, for a change that can change the results of a CFM run, for example:

- the forcing that `sim.run` gives to the CFM (units, signs, column maps, the spin-up);
- the `cfm_config` that `sim.run` builds;
- a default value that a run uses.

Do not increase it for comments, docstrings, log messages or names of local variables, a refactor that gives the same values, or code that a CFM run does not import (`src/plot/`, `src/sim/runs.py`, `scripts/`). If you are not sure, increase it: a new version only costs new runs, but an old version for a different result gives wrong results without a sign. Version 1 is the run code at `dev` `b89d566` and `d53fd13`. You can tag each new version `run-code-v<N>` for information; the code does not read tags.

**The drift warning.** Before each run, `CFMRun.prepare()` compares the current `run_code_sha256` with that of the complete runs in `runs/` with the same `run_code_version`. If one differs, it logs, once per process (so once per `run_cfm.py` run), `The run code changed since the runs of run code version N (commit <commit>). Increase RUN_CODE_VERSION in src/sim/identity.py if the change can change the results.` It is a warning, not an error, and it does not stop or change the run: the run hash is the same, so a complete run of the old code is still skipped. It is a prompt to decide whether the change needs a new version; it repeats after every run code change until the version is increased. Migrated runs, and runs from before FEAT-005 (`hash_version` 2, no `wrapper.run_code_version`), are not compared. A change to `src/plot/` or `src/sim/runs.py` gives no warning.

Each run folder holds `CFMconfig.json`, the CFM's `.hdf5` output, the wrapper's `run.log`, and a `manifest.json` that records the run's settings, the hash and what went into it (`hash_inputs`), the sites that asked for it (`run.sites`) and their coordinates (`run.site_coords`, `{site: [lat, lon]}`, not in the hash), host and timings, and its `status`: `running`, then `complete` or `failed` (with the error). Two caching behaviours to be aware of:

- A run is **skipped if its folder has a `complete` manifest with the same hash**. Pass `--force` to rerun it. A `failed` run, or a `running` one whose process has died, is rerun on its own. A folder with results but no manifest needs `--force`.
- Forcing data that has already been processed is reused rather than regenerated. It is checked to cover the years of the run first (BUG-003): for example `--rcm RACMO --end-year 2024` stops with an error, as the ANT-12 data stops at 2023.

Scripts find runs from their manifests, not from a folder name: `sim.runs.find_runs(site="CI-0", physRho="GSFC2020", cfm="ab871c8")` (the site's model point, so two sites in one cell share a run), or from the shell:

```bash
python scripts/find_runs.py --site CI-0 --physrho GSFC2020     # table of complete runs
python scripts/find_runs.py --status all --index               # all runs; also write {CFM_data_path}/index.csv
python scripts/find_runs.py --cfm 3e3bf7e-speersm --spin-end-year 1995   # the runs of one spin-up
python scripts/link_sites.py                                   # {CFM_data_path}/sites/<site>/<rcm> -> ../../runs/<model point>
```

`plot_cfm.py`, `calc_dh.py` and `plot_density_error.py` take `--cfm <label>` (default `default_cfm_version`). If several complete runs match (e.g. before and after a new `RUN_CODE_VERSION`), they use the one that finished last and log a warning listing the others. The index file is only an export for you; no code reads it. `find_runs` also compares the coordinates of each site in the manifests it returns with `sites.yaml` (to 3 decimals), and logs a warning, once per run folder and site, if they differ or the site is not in `sites.yaml`. The run is still returned and its results are correct for its model point; the warning means the site name pointed to a different location when the run was made (a site's name and coordinates must not change, see [Sites](#sites)). `link_sites.py` replaces wrong links and never deletes anything that is not a link; `--dry-run` prints the changes only.

**Spin-up.** The spin-up repeats the forcing from `start_year` to `spin_end_year` until the firn column is new, so a different `spin_end_year` gives different initial conditions and different results (FEAT-004). Each manifest records it in `run.spin_end_year`, and `find_runs(spin_end_year=...)`, `find_runs.py --spin-end-year` and the `spin_end_year` column of the table select on it. `plot_cfm.py`, `calc_dh.py` and `plot_density_error.py` only use the runs with the `spin_end_year` of `config.yaml`, unless you give `--spin-end-year <year>`, or `--spin-end-year any` for runs of any spin-up; if no run matches, the error gives the `spin_end_year` it used. The migrated `3e3bf7e-speersm` `MAR` runs of CI-0, CI-22, CI-120, WI-0 and WI-70 used 1995, so plot them with `--cfm 3e3bf7e-speersm --spin-end-year 1995`; the other migrated runs used 1989. `compare_cfm_runs.py --baseline ... --cfm ...` pairs only runs with the same `spin_end_year` and shows `spin_end_year` and `baseline_spin_end_year` for each pair; a run with no run of the same `spin_end_year` in the other version gets the note `no run of <label> with spin_end_year <year>`. `--allow-spin-mismatch` also pairs runs with different spin-ups, with the note `spin-up differs`; `--spin-end-year` compares the runs of one spin-up only.

Old output folders (`cfm_output/CFMoutput_<lat>_<lon>_..._<rcm>/`) can be copied into this layout with `scripts/migrate_runs.py`. Give each old `cfm_output` folder the label of the CFM version that made it, since the old folders do not record it. The default is a dry run that only prints the plan; `--copy` copies (never moves) the `.hdf5` files and writes a manifest marked `migrated`:

```bash
python scripts/migrate_runs.py --source 3e3bf7e-speersm ~/luna/CPOM/speersm/CFM_data/cfm_output \
    --source ab871c8 ~/CFM_data/ab871c8/cfm_output            # dry run
python scripts/migrate_runs.py ... --copy                     # copy into {CFM_data_path}/runs
```

An old run's code and environment are not known, so its hash is built from what is known (config JSON if any, schemes, years, `spin_end_year`, model point, forcing CSV hash, CFM label, `.hdf5` sizes) with `hash_version: migrated-2`; a new run never reuses it. The old folders do not record `spin_end_year`, so the script measures it from the repeated blocks of the spin-up in `CFMforcing.hdf5`, and writes it to `run.spin_end_year` (the block length and the number of blocks go in `migrated.spin`). The dry run shows it for each folder and counts the COPY folders of each label by `spin_end_year`; a folder whose spin-up cannot be measured gets `spin_end_year: null` and the note "the spin-up is not measured". Runs copied before this (`migrated-1`) have no `run.spin_end_year`, so `find_runs(spin_end_year=...)` and the plot scripts' default do not find them. Its model point is the cell its forcing came from, which for CI-0 and WI-70 on `RACMO` and CI-22 on `MAR` is the cell from before the BUG-002 fix, so find those runs with `--model-point`, not `--site`.

The `scripts/*.sh` batch wrappers loop over sites and densification schemes, and `cd` to the repo root themselves, so they can be run from anywhere. `run_cfm_grid.sh` is a SLURM array job: submit it from the repo root (or set `CFM_SIM_ROOT`), and set `CFM_CONDA_ENV` if your env isn't called `cfm` (on luna it was `CFM3.10.11`).

## Sites

The sites (borehole locations with in situ measurements) are in `sites.yaml`, which git tracks, so everyone has the same table. Each site has `lat`, `lon`, the model time `windows` that `plot_density_error.py` compares with the measurements, an optional `profile` file name, and `active: false` if the plot and analysis scripts should leave it out by default. `batches` are named lists of sites for the batch wrappers (`python scripts/list_sites.py --batch batch_1`).

**A site's name and coordinates never change.** `sites.lock` records them and `tests/test_sites.py` fails if they differ. `find_runs` gives a warning if a run's manifest records other coordinates for a site, e.g. a run made from another checkout. To correct a site, add a new one (e.g. `JP21-v2`), and give the old one `active: false` and `superseded_by: JP21-v2`. Add every new site to `sites.lock` in the same commit. To run the CFM somewhere new, add it to both files first.

### Model points and the forcing cache

The CFM is forced with the data of a single RCM grid cell: the cell whose centre is nearest the site (great-circle distance). That cell is the site's *model point*, `<rcm>_<i>_<j>` (`force.model_point.model_point(site, rcm_name)`). Sites in the same cell get identical forcing and identical CFM output; the plots say so, e.g. `J409: RACMO_253_145, 3.8 km from the cell centre, same cell as J109`.

Forcing is extracted once per model point, into a cache shared by all CFM versions:

```
{forcing_data_path}/<rcm_name>/<release>/<i>_<j>_<start_year>_<end_year>.csv
{forcing_data_path}/<rcm_name>/<release>/<i>_<j>_<start_year>_<end_year>.json   # cell, source files, column map, units, dates
```

`forcing_data_path` defaults to `{CFM_data_path}/forcing`. `<release>` comes from the RCM file names, so a new data release gets new forcing. Old `cfm_input/` CSVs can be checked against the cache layout with `python scripts/migrate_forcing_csvs.py --dry-run <cfm_input folder>` (it only prints a plan).

#### Fast forcing extraction

The `RACMO`, `RACMO23` and `MAR` forcing is extracted with `h5py` in one parallel pass over the files (`src/force/extract.py`). Each HDF5 chunk is a full daily map, so any method must read all the bytes (66 GB for `RACMO`, 47 GB for `RACMO23`, 24 GB for `MAR` 1979-2024). One pass takes approximately 1.5 minutes (`MAR`) to 5 minutes, and `forcing_processes` (default 8) sets the number of worker processes. The first pass also keeps all grid cells in a box around the active sites in `sites.yaml` in a local `cache.h5` (next to the forcing CSVs; `forcing_cache`, `forcing_cache_dir` and `forcing_cache_margin` in `config.example.yaml`). A later site inside the box then takes under one second. `python scripts/extract_forcing.py` makes the forcing for all active sites in one pass (`--rcm MAR` for MAR). For `MAR`, the extraction keeps `SECTOR` = `sector` and the surface layer (`OUTLAY` = 0), as the old code did.

The worker processes start with `forkserver` on Linux (`spawn` on macOS and Windows), not `fork`, which can deadlock in a process with threads (BUG-008). Each worker therefore imports the main script again. A script that starts the extraction, directly or through `ProcessRACMO`/`ProcessRACMO23`/`ProcessMAR.process()`, must do its work inside `if __name__ == "__main__":`, as `scripts/run_cfm.py` and `scripts/extract_forcing.py` do. Without the guard, the pool stops with `BrokenProcessPool`.

## Sanity check

The CFM's own example verifies the model side independently of any forcing data:

```bash
CFM_repo_path=/path/to/CommunityFirnModel/CFM_main   # the repo of the CFM version to check
out=$(mktemp -d)
python - "$CFM_repo_path" "$out" <<'EOF'
import json, os, sys
repo, out = sys.argv[1:]
with open(os.path.join(repo, "example_df.json")) as f:
    c = json.load(f)
c["InputFileFolder"] = os.path.join(repo, c["InputFileFolder"])
c["resultsFolder"] = os.path.join(out, "results")
with open(os.path.join(out, "example_df.json"), "w") as f:
    json.dump(c, f, indent=2)
EOF
(cd "$out" && python "$CFM_repo_path/main.py" "$out/example_df.json" -n)
```

It takes a few seconds, and writes everything to the temporary folder `$out`. Do not run `python main.py example_df.json` inside the CFM repo: that overwrites the tracked files in `CFMoutput_example/` and leaves `diagnostics_enthalpy.csv` behind. The config path must be absolute because `main.py` resolves a relative one against its own folder.
