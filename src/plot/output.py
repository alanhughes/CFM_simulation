"""The output of one CFM run, read from its folder (FEAT-007).

RunOutput reads CFMresults.hdf5 and CFMforcing.hdf5 lazily: each array is read the first time it is used, so a figure
that needs only DIP does not read the density. RunOutput.from_folder() does not need config.yaml.
exactly_one() and site_profile() use the run manifests and sites.yaml, so they need config.yaml.

Row 0 of the gridded outputs is the initial state at the start of the spin-up, and column 0 is the model time. Each
property removes both.
"""

import os
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, Optional, Sequence, Tuple

import h5py
import numpy as np

from plot import metrics

RESULTS_FILE = "CFMresults.hdf5"
FORCING_FILE = "CFMforcing.hdf5"
MAIN_RUN_START = 1980.0  # the first year of the CFM output (TWriteStart); the forcing summary uses the same years


@dataclass(frozen=True)
class ForcingSummary:
    """Means of the forcing of a run. The means "since 1980" use the main run only; iceout uses all the forcing, with
    the repeated spin-up blocks, as the CFM does."""

    spin_start: float  # the first time of the spin-up
    t2m_since_1980: float  # K
    bdot_since_1980: float  # m ice eq. per year
    rain_since_1980: float  # m w.e. per year, as in the forcing
    iceout: float  # m ice eq. per year: the mean BDOT of all the forcing

    @property
    def iceout_excess(self) -> float:
        """The mean accumulation since 1980 minus iceout, in m ice eq. per year."""
        return self.bdot_since_1980 - self.iceout


@dataclass
class RunOutput:
    """The arrays of one run. label is free text for legends and errors, for example the scheme."""

    folder: str
    label: str = ""
    _cache: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @classmethod
    def from_folder(cls, folder: str, label: str = "") -> "RunOutput":
        if not os.path.exists(os.path.join(folder, RESULTS_FILE)):
            raise FileNotFoundError(f"No {RESULTS_FILE} in {folder}")
        return cls(folder=folder, label=label)

    @classmethod
    def from_run(cls, run, label: Optional[str] = None) -> "RunOutput":
        """From a sim.runs.Run. The label is the scheme unless given."""
        return cls.from_folder(run.folder, label=run.physRho if label is None else label)

    def _read(self, name: str, rows=slice(1, None), cols=None) -> np.ndarray:
        key = f"{name}{rows}{cols}"
        if key not in self._cache:
            with h5py.File(os.path.join(self.folder, RESULTS_FILE), "r") as results:
                data = results[name]
                self._cache[key] = data[rows] if cols is None else data[rows, cols]
        return self._cache[key]

    # the arrays of the output
    @property
    def time(self) -> np.ndarray:
        """Decimal year of each output step."""
        return self._read("DIP", cols=0)

    @property
    def depth(self) -> np.ndarray:
        """The depth grid, m (grid_outputs)."""
        return self._read("depth")

    @property
    def density(self) -> np.ndarray:
        """(time, depth), kg/m3."""
        return self._read("density", cols=slice(1, None))

    @property
    def temperature(self) -> np.ndarray:
        """(time, depth), K."""
        return self._read("temperature", cols=slice(1, None))

    @property
    def fac(self) -> np.ndarray:
        """Firn air content (DIP column 1), m."""
        return self._read("DIP", cols=1)

    @property
    def surface_height(self) -> np.ndarray:
        """Cumulative surface height change since the first output step (the sum of DIP column 2), m."""
        return np.cumsum(self._read("DIP", cols=2))

    def per_step(self, name: str) -> np.ndarray:
        """meltvol, refreeze or runoff: the amount of each time step, m w.e."""
        return self._read(name, cols=1)

    # quantities
    def depth_of_density(self, target: float, step: int = -1) -> float:
        """For example z830 of the final profile."""
        return metrics.depth_of_density(self.depth, self.density[step], target)

    def at_depth(self, field_name: str, depth: float) -> np.ndarray:
        """The time series of density or temperature at the grid depth nearest to depth."""
        return getattr(self, field_name)[:, metrics.depth_index(self.depth, depth)]

    def dhdt(self) -> float:
        """The linear trend of the cumulative surface height change, m/yr."""
        return metrics.linear_trend(self.time, self.surface_height)

    def compare_with_profile(
        self, profile: metrics.Profile, windows: Sequence[Tuple[float, float]]
    ) -> metrics.ProfileComparison:
        return metrics.compare_with_profile(self.time, self.depth, self.density, profile, windows)

    def water_budget(self) -> Dict[str, float]:
        return metrics.water_budget(
            self.time,
            self.per_step("meltvol"),
            self.per_step("refreeze"),
            self.per_step("runoff"),
            self.forcing.bdot_since_1980,
        )

    @cached_property
    def forcing(self) -> ForcingSummary:
        with h5py.File(os.path.join(self.folder, FORCING_FILE), "r") as forcing:
            time = forcing["main/time"][()]
            main = time >= MAIN_RUN_START
            bdot = forcing["main/BDOT"][()]
            return ForcingSummary(
                spin_start=float(time[0]),
                t2m_since_1980=float(forcing["main/T2m"][()][main].mean()),
                bdot_since_1980=float(bdot[main].mean()),
                rain_since_1980=float(forcing["main/RAIN"][()][main].mean()),
                iceout=float(bdot.mean()),
            )


def exactly_one(**filters):
    """The one complete run that agrees with the filters of sim.runs.find_runs. Raises LookupError if no run or more
    than one run agrees, so that a figure never takes a run by chance (find_run takes the newest, with a warning).
    With more than one run, the message gives the cfm_config keys that differ between them (FEAT-009)."""
    from sim.runs import describe_cfm_config_differences, find_runs

    runs = find_runs(**filters)
    if len(runs) != 1:
        found = "\n".join(f"  {run.folder}" for run in runs) or "  (none)"
        hint = f"\n{describe_cfm_config_differences(runs)}" if len(runs) > 1 else ""
        raise LookupError(f"Expected exactly one run for {filters}, found {len(runs)}:\n{found}{hint}")
    return runs[0]


def site_profile(site: str) -> metrics.Profile:
    """The in situ profile of a site in sites.yaml, from in_situ_data_path."""
    from cfm_config import config, get_site

    return metrics.load_profile(os.path.join(config["in_situ_data_path"], get_site(site).profile_file), site=site)


def site_windows(site: str):
    """The measurement windows of a site in sites.yaml, in decimal years."""
    from cfm_config import get_site

    return get_site(site).decimal_year_windows()
