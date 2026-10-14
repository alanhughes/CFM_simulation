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
| `CFM_data_path` | Where the wrapper writes forcing CSVs, model output and figures — a local folder, not the share |
| `MAR_data_path` / `RACMO_data_path` / `RACMO23_data_path` | Where the raw RCM NetCDF forcing files are read from |
| `in_situ_data_path` | Where the in situ depth-density CSVs are read from |

Everything below that block (borehole sites, densification options, RCM column mappings, plot limits) is machine-independent.

## Data on the luna share

The CPOM share on luna is **mounted read-only, and all output is written locally**. The share holds other people's data, including the previous user's runs under `speersm/`. We have no admin rights to change permissions on it, so a read-only mount is the only way to guarantee nothing there gets overwritten. Results are copied up to your own folder on the share (`hughes54/CFM_data`) as a separate, deliberate step.

In `config.yaml` that means:

- `cpom_mount` is the read-only mount point (e.g. `/Users/alan/mnt/cpom`). The RCM forcing and in situ data are read through it.
- `CFM_data_path` is a local folder (e.g. `/Users/alan/code/exageo/CFM_data`). Every file the code writes — forcing CSVs, CFM output, figures, melt files — goes under it.

Nothing needs to be copied from `speersm/`: the model regenerates everything there except the in situ data, which has been copied to `hughes54/CFM_data/in_situ`.

### Mount read-only

Mount in your home folder: `/Volumes` is owned by root, so `mount_smbfs` can't create a mount point there without admin rights.

```bash
diskutil unmount /Volumes/cpom      # only if Finder has already mounted the share (read-write)
mkdir -p ~/mnt/cpom
mount_smbfs -o rdonly //hughes54@luna.lancaster.ac.uk/fst/lec/cpom ~/mnt/cpom
```

It may prompt for your luna password. Check the mount really is read-only before running anything:

```bash
mount | grep cpom                                # should list "read-only"
touch ~/mnt/cpom/hughes54/CFM_data/test_write    # should fail: "Read-only file system"
```

The mount does not reconnect itself after sleep or a network drop. If reads from the share start failing with I/O errors, run `diskutil unmount ~/mnt/cpom` and the `mount_smbfs` line again.

### Copy results to the share

Remount read-write only for the copy, then go straight back to read-only:

```bash
diskutil unmount ~/mnt/cpom
mount_smbfs //hughes54@luna.lancaster.ac.uk/fst/lec/cpom ~/mnt/cpom           # read-write

rsync -avn /Users/alan/code/exageo/CFM_data/ ~/mnt/cpom/hughes54/CFM_data/    # dry run: check the file list
rsync -av  /Users/alan/code/exageo/CFM_data/ ~/mnt/cpom/hughes54/CFM_data/

diskutil unmount ~/mnt/cpom
mount_smbfs -o rdonly //hughes54@luna.lancaster.ac.uk/fst/lec/cpom ~/mnt/cpom  # back to read-only
```

Keep the trailing `/` on both `rsync` paths, so the folder's contents are copied rather than a nested `CFM_data/CFM_data`. Never add `--delete`: without it `rsync` only adds and updates files under `hughes54/CFM_data` and touches nothing else on the share. Don't run any scripts while the share is read-write.

## Running

**All scripts must be run from the repo root** — `src/cfm_config.py` opens `config.yaml` by relative path. It also expands `{cpom_mount}` in the top-level paths, so if luna is mounted somewhere else on your machine, only `cpom_mount` needs changing.

```bash
python scripts/run_cfm.py                                   # uses config.yaml defaults
python scripts/run_cfm.py --lat -66.403 --lon -63.376 \
    --physrho GSFC2020 --rcm RACMO --liquid bucket          # all five args required to override
```

`run_cfm.py` processes the RCM forcing into a CSV under `{CFM_data_path}/cfm_input/`, then runs the CFM, writing to `{CFM_data_path}/cfm_output/CFMoutput_<lat>_<lon>_<years>_<physrho>_<liquid>_<rcm>/`.

Two caching behaviours to be aware of: a run whose **output folder already exists is skipped silently** (delete the folder to re-run), and forcing data that has already been processed is reused rather than regenerated.

The `scripts/*.sh` batch wrappers loop over sites and densification schemes, and `cd` to the repo root themselves, so they can be run from anywhere. `run_cfm_grid.sh` is a SLURM array job: submit it from the repo root (or set `CFM_SIM_ROOT`), and set `CFM_CONDA_ENV` if your env isn't called `cfm` (on luna it was `CFM3.10.11`).

## Sanity check

The CFM's own example verifies the model side independently of any forcing data:

```bash
cd $CFM_repo_path && python main.py example_df.json -n
```

It takes a few seconds. Note it overwrites `CFMoutput_example/`, which will show as modified in the CFM repo's git status.
