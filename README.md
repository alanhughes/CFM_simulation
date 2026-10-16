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

Everything below that block (borehole sites, densification options, RCM column mappings, plot limits) is machine-independent.

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
python scripts/run_cfm.py                                   # uses config.yaml defaults
python scripts/run_cfm.py --lat -66.403 --lon -63.376 \
    --physrho GSFC2020 --rcm RACMO --liquid bucket          # all five args required to override
```

`run_cfm.py` processes the RCM forcing into a CSV under `{CFM_data_path}/cfm_input/`, then runs the CFM, writing to `{CFM_data_path}/cfm_output/CFMoutput_<lat>_<lon>_<years>_<physrho>_<liquid>_<rcm>/`.

Each output folder holds a `manifest.json` that records the run's settings, host and timings, and its `status`: `running`, then `complete` or `failed` (with the error). The wrapper's log goes to `run.log` next to it. Two caching behaviours to be aware of:

- A run is **skipped if its folder has a `complete` manifest**, even if `config.yaml` has changed since, because the folder name does not yet capture every setting. Pass `--force` to rerun it. A `failed` run, or a `running` one whose process has died, is rerun on its own. An old folder with `CFMresults.hdf5` but no manifest is treated as complete.
- Forcing data that has already been processed is reused rather than regenerated.

The `scripts/*.sh` batch wrappers loop over sites and densification schemes, and `cd` to the repo root themselves, so they can be run from anywhere. `run_cfm_grid.sh` is a SLURM array job: submit it from the repo root (or set `CFM_SIM_ROOT`), and set `CFM_CONDA_ENV` if your env isn't called `cfm` (on luna it was `CFM3.10.11`).

## Sanity check

The CFM's own example verifies the model side independently of any forcing data:

```bash
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
