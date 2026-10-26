"""Quantities of one CFM run, and the comparison of a run with an in situ density profile (FEAT-007).

The functions take numpy arrays and do not read config.yaml, so scripts/compare_cfm_runs.py can use them in its folder
mode. plot.output.RunOutput gives the arrays of a run. The arrays are those of the gridded output (grid_outputs true):
depth is 1-D, and density and temperature are (time, depth), without row 0 (the initial state) and column 0 (time).

The comparison with a profile follows scripts/plot_density_error.py: the model density of each time step in the
measurement windows is interpolated to the depths of the profile, and the bias is the mean of model - observation over
those depths.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

ICE_EQUIVALENT_DENSITY = 917.0  # kg/m3, for BDOT (m ice eq. per year) to kg/m2 per year


def depth_index(depth: np.ndarray, target: float) -> int:
    """The index of the depth nearest to target."""
    return int(np.argmin(np.abs(depth - target)))


def depth_of_density(depth: np.ndarray, profile: np.ndarray, target: float) -> float:
    """The first depth at which the profile reaches target (for example z830). If the profile never reaches it, this is
    depth[0], as in scripts/compare_cfm_runs.py."""
    return float(depth[np.argmax(profile >= target)])


def linear_trend(time: np.ndarray, values: np.ndarray) -> float:
    """The slope of a least-squares line, in the units of values per year."""
    return float(np.polyfit(time, values, 1)[0])


def annual_total(time: np.ndarray, per_step: np.ndarray) -> float:
    """The mean amount per year of a quantity given per time step (for example meltvol, in m w.e.): the sum of all the
    steps divided by the length of the record in years."""
    return float(np.nansum(per_step) / (time[-1] - time[0]))


def window_mask(time: np.ndarray, windows: Iterable[Tuple[float, float]]) -> np.ndarray:
    """True for the time steps in any of the windows. A window includes both of its ends, as plot_density_error.py."""
    mask = np.zeros(time.shape, dtype=bool)
    for start, end in windows:
        mask |= (time >= start) & (time <= end)
    return mask


@dataclass
class Profile:
    """An in situ depth-density profile. It is one borehole log: a snapshot, with no time dimension."""

    site: str
    depth: np.ndarray  # m
    density: np.ndarray  # kg/m3

    def running_mean(self, window_m: float = 1.0) -> np.ndarray:
        """The density smoothed with a centred running mean of window_m metres (the profile has a regular spacing)."""
        spacing = float(np.median(np.diff(self.depth)))
        n = max(1, int(round(window_m / spacing)))
        return pd.Series(self.density).rolling(n, center=True, min_periods=1).mean().to_numpy()

    def fraction_above(self, density: float, top: float, bottom: float) -> float:
        """The fraction of the values in [top, bottom) that are above density."""
        values = self.density[(self.depth >= top) & (self.depth < bottom)]
        return float(np.mean(values > density))


def load_profile(path: str, site: str = "") -> Profile:
    """Read an in situ CSV with the columns Depth (m) and Density (kg/m3)."""
    data = pd.read_csv(path)
    return Profile(
        site=site,
        depth=pd.to_numeric(data["Depth"]).to_numpy(float),
        density=pd.to_numeric(data["Density"]).to_numpy(float),
    )


@dataclass
class ProfileComparison:
    """The model in the measurement windows against one profile, on the depths of the profile."""

    depth: np.ndarray  # the depths of the profile
    mean: np.ndarray  # the mean model density over the time steps in the windows
    lower: np.ndarray  # the smallest model density in the windows
    upper: np.ndarray  # the largest model density in the windows
    observed: np.ndarray
    steps: int  # the number of model time steps in the windows

    @property
    def error(self) -> np.ndarray:
        return self.mean - self.observed

    @property
    def bias(self) -> float:
        return float(self.error.mean())

    @property
    def rmse(self) -> float:
        return float(np.sqrt((self.error**2).mean()))

    def rmse_above(self, depth: float) -> float:
        """The RMSE of the depths at or above depth (for example the top 20 m)."""
        top = self.depth <= depth
        return float(np.sqrt((self.error[top] ** 2).mean()))


def compare_with_profile(
    time: np.ndarray,
    depth: np.ndarray,
    density: np.ndarray,
    profile: Profile,
    windows: Sequence[Tuple[float, float]],
) -> ProfileComparison:
    """Interpolate the model density of each time step in the windows to the depths of the profile, and compare.

    Only the model depths within the range of the profile are used, as in plot_density_error.py. Raises ValueError if
    no time step is in the windows.
    """
    steps = window_mask(time, windows)
    if not steps.any():
        raise ValueError(f"No model time step is in the windows {list(windows)} (model {time[0]:.2f}-{time[-1]:.2f})")
    inside = (depth >= profile.depth.min()) & (depth <= profile.depth.max())
    interpolated = np.array([np.interp(profile.depth, depth[inside], row[inside]) for row in density[steps]])
    return ProfileComparison(
        depth=profile.depth,
        mean=interpolated.mean(axis=0),
        lower=interpolated.min(axis=0),
        upper=interpolated.max(axis=0),
        observed=profile.density,
        steps=int(steps.sum()),
    )


def water_budget(
    time: np.ndarray,
    meltvol: np.ndarray,
    refreeze: np.ndarray,
    runoff: np.ndarray,
    bdot_since_1980: float,
) -> Dict[str, float]:
    """The mean annual accumulation, melt, refreezing (including rain) and runoff, in kg/m2 per year.

    meltvol, refreeze and runoff are the CFM outputs per time step in m w.e. (1 m w.e. = 1000 kg/m2). bdot_since_1980 is
    the mean forcing BDOT in m ice eq. per year.
    """
    budget = {
        "accumulation": bdot_since_1980 * ICE_EQUIVALENT_DENSITY,
        "melt": 1000 * annual_total(time, meltvol),
        "refreeze": 1000 * annual_total(time, refreeze),
        "runoff": 1000 * annual_total(time, runoff),
    }
    budget["refreeze_fraction"] = budget["refreeze"] / budget["accumulation"]
    return budget


def largest_difference_by_time(density_a: np.ndarray, density_b: np.ndarray) -> np.ndarray:
    """The largest |a - b| over depth at each time step, on the depths that both runs have."""
    n = min(density_a.shape[1], density_b.shape[1])
    return np.nanmax(np.abs(density_a[:, :n] - density_b[:, :n]), axis=1)


def annual(time: np.ndarray, values: np.ndarray, statistic=np.nanmax) -> Tuple[np.ndarray, np.ndarray]:
    """A statistic of each calendar year (the maximum by default), at the middle of the year (year + 0.5)."""
    years = np.floor(time).astype(int)
    unique = np.unique(years)
    return unique + 0.5, np.array([statistic(values[years == year]) for year in unique])
