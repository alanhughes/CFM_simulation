# CFM_simulation
Column and grid simulation from the Community Firn Model (CFM). Relies on functionality from the CFM repository https://github.com/UWGlaciology/CommunityFirnModel.

## Setup

There is no build step — this is pure Python. Three things need to be in place.

**1. The CFM checkout.** This repo does not vendor or pip-install the CFM. `src/sim/run.py` appends `CFM_repo_path` from `config.yaml` to `sys.path` and imports `RCMpkl_to_spin` and `firn_density_nospin` as flat modules. Clone the CFM alongside this repo and point `CFM_repo_path` at its `CFM_main` folder (not the repo root).

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
| `CFM_repo_path` | The CFM's `CFM_main` folder |
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

`run_cfm.py` processes the RCM forcing of the site's model point into the forcing cache (see below), then runs the CFM, writing to `{CFM_data_path}/cfm_output/CFMoutput_<lat>_<lon>_<years>_<physrho>_<liquid>_<rcm>/`.

Each output folder holds a `manifest.json` that records the run's settings, host and timings, and its `status`: `running`, then `complete` or `failed` (with the error). The wrapper's log goes to `run.log` next to it. Two caching behaviours to be aware of:

- A run is **skipped if its folder has a `complete` manifest**, even if `config.yaml` has changed since, because the folder name does not yet capture every setting. Pass `--force` to rerun it. A `failed` run, or a `running` one whose process has died, is rerun on its own. An old folder with `CFMresults.hdf5` but no manifest is treated as complete.
- Forcing data that has already been processed is reused rather than regenerated. It is checked to cover `start_year` to `end_year` first: the ANT-12 `RACMO` data stops at 2023, so `end_year: 2024` stops with an error for `--rcm RACMO`.

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

## Sanity check

The CFM's own example verifies the model side independently of any forcing data:

```bash
cd $CFM_repo_path && python main.py example_df.json -n
```

It takes a few seconds. Note it overwrites `CFMoutput_example/`, which will show as modified in the CFM repo's git status.
