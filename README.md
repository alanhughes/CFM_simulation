# CFM_simulation
Column and grid simulation from the Community Firn Model (CFM). Relies on functionality from the CFM repository https://github.com/UWGlaciology/CommunityFirnModel.

## Setup

There is no build step — this is pure Python. Three things need to be in place.

**1. The CFM checkout.** This repo does not vendor or pip-install the CFM. Clone the CFM alongside this repo and list it under `cfm_versions` in `config.yaml`: a label, the checkout's `CFM_main` folder (not the repo root) and the Python to run it with. You can list several versions and pick one with `run_cfm.py --cfm <label>`. Each run starts a subprocess (`sim.worker`) with that version's Python, which appends the `CFM_main` folder to `sys.path` and imports `RCMpkl_to_spin` and `firn_density_nospin` as flat modules. That is why one process can only use one CFM.

The label is only a name. Each run's `manifest.json` records the CFM's measured commit, its uncommitted changes, the Python version and `pip freeze`. A checkout with uncommitted `.py` changes is refused, so commit local patches to a branch of the CFM checkout. The only exception is a version that sets `allow_uncommitted: true`, kept for the speersm reference test (see `config.example.yaml`).

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

`run_cfm.py` processes the RCM forcing of the site's model point into the forcing cache (see below), then runs the CFM, writing to

```
{CFM_data_path}/runs/<model point>/<physrho>_<liquid>_<start>-<end>/<cfm label>/<hash8>/
```

`<model point>` is the RCM grid cell, e.g. `RACMO_261_126`, so two sites in one cell share a run. `<hash8>` is the start of the **run hash**: the SHA-256 of everything that changes the results - all of `cfm_config`, `start_year`/`end_year`/`spin_end_year`/`sublim_sf`/`rho_bottom`, the model point and RCM release, the SHA-256 of the forcing CSV, the CFM's commit, `.py` patch, Python version and `pip freeze`, and the committed run code of this repo (`src/` without `src/plot/`, which a run does not import). Changing any of them gives a new folder; commits that touch only docs, scripts, tests or `src/plot/` do not. Uncommitted `.py` changes in the run code are refused, like those in the CFM.

Each run folder holds `CFMconfig.json`, the CFM's `.hdf5` output, the wrapper's `run.log`, and a `manifest.json` that records the run's settings, the hash and what went into it (`hash_inputs`), the sites that asked for it, host and timings, and its `status`: `running`, then `complete` or `failed` (with the error). Two caching behaviours to be aware of:

- A run is **skipped if its folder has a `complete` manifest with the same hash**. Pass `--force` to rerun it. A `failed` run, or a `running` one whose process has died, is rerun on its own. A folder with results but no manifest needs `--force`.
- Forcing data that has already been processed is reused rather than regenerated. It is checked to cover `start_year` to `end_year` first: the ANT-12 `RACMO` data stops at 2023, so `end_year: 2024` stops with an error for `--rcm RACMO`.

Scripts find runs from their manifests, not from a folder name: `sim.runs.find_runs(site="CI-0", physRho="GSFC2020", cfm="ab871c8")` (the site's model point, so two sites in one cell share a run), or from the shell:

```bash
python scripts/find_runs.py --site CI-0 --physrho GSFC2020     # table of complete runs
python scripts/find_runs.py --status all --index               # all runs; also write {CFM_data_path}/index.csv
python scripts/link_sites.py                                   # {CFM_data_path}/sites/<site>/<rcm> -> ../../runs/<model point>
```

`plot_cfm.py`, `calc_dh.py` and `plot_density_error.py` take `--cfm <label>` (default `default_cfm_version`). If several complete runs match (e.g. before and after a change to the run code), they use the one that finished last and log a warning listing the others. The index file is only an export for you; no code reads it. `link_sites.py` replaces wrong links and never deletes anything that is not a link; `--dry-run` prints the changes only.

Old output folders (`cfm_output/CFMoutput_<lat>_<lon>_..._<rcm>/`) can be copied into this layout with `scripts/migrate_runs.py`. Give each old `cfm_output` folder the label of the CFM version that made it, since the old folders do not record it. The default is a dry run that only prints the plan; `--copy` copies (never moves) the `.hdf5` files and writes a manifest marked `migrated`:

```bash
python scripts/migrate_runs.py --source 3e3bf7e-speersm ~/luna/CPOM/speersm/CFM_data/cfm_output \
    --source ab871c8 ~/CFM_data/ab871c8/cfm_output            # dry run
python scripts/migrate_runs.py ... --copy                     # copy into {CFM_data_path}/runs
```

An old run's code and environment are not known, so its hash is built from what is known (config JSON if any, schemes, years, model point, forcing CSV hash, CFM label, `.hdf5` sizes) with `hash_version: migrated-1`; a new run never reuses it. Its model point is the cell its forcing came from, which for CI-0 and WI-70 on `RACMO` and CI-22 on `MAR` is the cell from before the BUG-002 fix, so find those runs with `--model-point`, not `--site`.

The `scripts/*.sh` batch wrappers loop over sites and densification schemes, and `cd` to the repo root themselves, so they can be run from anywhere. `run_cfm_grid.sh` is a SLURM array job: submit it from the repo root (or set `CFM_SIM_ROOT`), and set `CFM_CONDA_ENV` if your env isn't called `cfm` (on luna it was `CFM3.10.11`).

## Sites

The sites (borehole locations with in situ measurements) are in `sites.yaml`, which git tracks, so everyone has the same table. Each site has `lat`, `lon`, the model time `windows` that `plot_density_error.py` compares with the measurements, an optional `profile` file name, and `active: false` if the plot and analysis scripts should leave it out by default. `batches` are named lists of sites for the batch wrappers (`python scripts/list_sites.py --batch batch_1`).

**A site's name and coordinates never change.** `sites.lock` records them and `tests/test_sites.py` fails if they differ. To correct a site, add a new one (e.g. `JP21-v2`), and give the old one `active: false` and `superseded_by: JP21-v2`. Add every new site to `sites.lock` in the same commit. To run the CFM somewhere new, add it to both files first.

### Model points and the forcing cache

The CFM is forced with the data of a single RCM grid cell: the cell whose centre is nearest the site (great-circle distance). That cell is the site's *model point*, `<rcm>_<i>_<j>` (`force.model_point.model_point(site, rcm_name)`). Sites in the same cell get identical forcing and identical CFM output; the plots say so, e.g. `J409: RACMO_253_145, 3.8 km from the cell centre, same cell as J109`.

Forcing is extracted once per model point, into a cache shared by all CFM versions:

```
{forcing_data_path}/<rcm_name>/<release>/<i>_<j>_<start_year>_<end_year>.csv
{forcing_data_path}/<rcm_name>/<release>/<i>_<j>_<start_year>_<end_year>.json   # cell, source files, column map, units, dates
```

`forcing_data_path` defaults to `{CFM_data_path}/forcing`. `<release>` comes from the RCM file names, so a new data release gets new forcing. Old `cfm_input/` CSVs can be checked against the cache layout with `python scripts/migrate_forcing_csvs.py --dry-run <cfm_input folder>` (it only prints a plan).

#### Fast RACMO extraction

The `RACMO` and `RACMO23` forcing is extracted with `h5py` in one parallel pass over the files (`src/force/extract.py`). Each HDF5 chunk is a full daily map, so any method must read all the bytes (66 GB for `RACMO`, 47 GB for `RACMO23`). One pass takes approximately 5 minutes, and `forcing_processes` (default 8) sets the number of worker processes. The first pass also keeps all grid cells in a box around the active sites in `sites.yaml` in a local `cache.h5` (next to the forcing CSVs; `forcing_cache`, `forcing_cache_dir` and `forcing_cache_margin` in `config.example.yaml`). A later site inside the box then takes under one second. `python scripts/extract_forcing.py` makes the forcing for all active sites in one pass. MAR uses the older xarray code.

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
