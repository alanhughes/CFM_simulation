"""Profile panels: density or temperature against depth (FEAT-007).

The functions draw on a matplotlib Axes and return nothing, so a figure can combine them. Depth increases downwards.
Colour is the scheme and line style is the side of a pair (plot.style).
"""

from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from plot import style
from plot.metrics import Profile, ProfileComparison


def depth_axes(ax: plt.Axes, bottom: float, xlabel: str, xlim: Optional[Tuple[float, float]] = None) -> None:
    """Depth downwards from 0 to bottom (m), with the x label and limits."""
    style.depth_axis(ax, bottom)
    ax.set_xlabel(xlabel)
    if xlim is not None:
        ax.set_xlim(*xlim)


def draw_observed(ax: plt.Axes, profile: Profile, smooth_m: Optional[float] = 1.0, raw: bool = True) -> None:
    """The observed profile: the raw values as a light grey line, and a running mean of smooth_m metres in black. With
    smooth_m None, the raw values are drawn in black."""
    if smooth_m is None:
        ax.plot(profile.density, profile.depth, label="Observed", **style.OBSERVATION)
        return
    if raw:
        ax.plot(profile.density, profile.depth, label="_raw", zorder=1, **style.OBSERVATION_RAW)
    ax.plot(
        profile.running_mean(smooth_m),
        profile.depth,
        label=f"Observed ({smooth_m:g} m mean)",
        zorder=4,
        **style.OBSERVATION,
    )


def draw_comparison(
    ax: plt.Axes,
    comparison: ProfileComparison,
    scheme: str,
    linestyle: str = style.CURRENT,
    band: bool = False,
    label: Optional[str] = None,
) -> None:
    """The model mean in the measurement windows, and with band, the range of the model within the windows."""
    colour = style.scheme_colour(scheme)
    if band:
        ax.fill_betweenx(comparison.depth, comparison.lower, comparison.upper, color=colour, alpha=0.15, lw=0, zorder=2)
    ax.plot(comparison.mean, comparison.depth, color=colour, linestyle=linestyle, label=label or scheme, zorder=3)


def draw_profile(
    ax: plt.Axes,
    values: np.ndarray,
    depth: np.ndarray,
    scheme: str,
    linestyle: str = style.CURRENT,
    label: Optional[str] = None,
    **kwargs,
) -> None:
    """One profile in the colour of its scheme, for example a run-mean temperature or a difference of a pair."""
    ax.plot(values, depth, color=style.scheme_colour(scheme), linestyle=linestyle, label=label or scheme, **kwargs)


def reference_value(ax: plt.Axes, value: float, label: Optional[str] = None, side: str = "right") -> None:
    """A vertical reference line (for example 830 kg/m3, or zero for a difference), labelled at the top of the panel
    (the surface, where the profiles rarely are), on the given side of the line."""
    ax.axvline(value, zorder=1, **style.REFERENCE)
    if label:
        ax.annotate(
            label,
            (value, 1),
            xycoords=("data", "axes fraction"),
            xytext=(3 if side == "right" else -3, -3),
            textcoords="offset points",
            ha="left" if side == "right" else "right",
            va="top",
            fontsize=7,
            color=style.TEXT_SECONDARY,
        )


def shade_depths(ax: plt.Axes, bands: Iterable[Tuple[float, float, str]]) -> None:
    """Grey bands across the panel for depth intervals (top, bottom, label), for example the units of a borehole. The
    bands alternate between two greys. Each label is vertical at the left edge, where firn profiles rarely are, and
    starts at the top of its band (a legend often fills the bottom of a profile panel)."""
    for i, (top, bottom, label) in enumerate(bands):
        ax.axhspan(top, bottom, color=style.SHADE if i % 2 == 0 else "#f6f5f2", zorder=0, lw=0)
        ax.annotate(
            label,
            (0, top),
            xycoords=("axes fraction", "data"),
            xytext=(4, -4),
            textcoords="offset points",
            ha="left",
            va="top",
            rotation=90,
            fontsize=7,
            color=style.TEXT_SECONDARY,
        )


def mark_point(ax: plt.Axes, value: float, depth: float, label: str, filled: bool = True) -> None:
    """A single measured or published value at one depth, as a black marker."""
    ax.plot(
        value,
        depth,
        marker="o",
        markersize=6,
        linestyle="none",
        markerfacecolor="black" if filled else "white",
        markeredgecolor="black",
        markeredgewidth=1.0,
        label=label,
        zorder=5,
    )


def scheme_legend(ax: plt.Axes, loc: str = "lower left") -> None:
    """A legend without the hidden entries (labels that start with "_")."""
    handles, labels = ax.get_legend_handles_labels()
    keep = [(h, l) for h, l in zip(handles, labels) if not l.startswith("_")]
    if keep:
        ax.legend(*zip(*keep), loc=loc)
