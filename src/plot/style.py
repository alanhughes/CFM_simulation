"""One plot style for all the figures (FEAT-007).

Colour identifies the densification scheme, in every figure. Line style identifies the side of a pair: solid for the
current set-up, dashed for the alternative (see PAIRS). Observations are black, and reference lines (830 kg/m3, zero,
the mean 2 m air temperature) are thin grey dashed lines. Text is never in a series colour.

The scheme colours are slots 1-5 of the categorical palette of the dataviz guidance, in its validated order. Yellow and
magenta have less than 3:1 contrast on white, so a figure with them always has a legend.
"""

import os
import string
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt

SCHEMES = ("GSFC2020", "HLdynamic", "Crocus", "Barnola1991", "Ligtenberg2011")
SCHEME_COLOURS = {
    "GSFC2020": "#2a78d6",  # blue
    "HLdynamic": "#eb6834",  # orange
    "Crocus": "#1baf7a",  # aqua
    "Barnola1991": "#eda100",  # yellow
    "Ligtenberg2011": "#e87ba4",  # magenta
}
OTHER_SCHEME_COLOUR = "#7a7a76"  # a scheme that is not in SCHEMES

# colours that are not schemes: slots 6-8 of the same palette, for the terms of the water budget
BUDGET_COLOURS = {
    "accumulation": "#9b9a95",
    "melt": "#008300",
    "refreeze": "#6250d6",
    "runoff": "#e34948",
}

CURRENT = "-"
ALTERNATIVE = "--"
# the two sides of each pair of runs: (current, alternative)
PAIRS = {
    "cfm": ("ab871c8", "3e3bf7e-speersm"),
    "cell": ("corrected cell", "old cell"),
    "rcm": ("RACMO", "MAR"),
    "spin": ("1979-1989", "1979-1995"),
}

TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
OBSERVATION = {"color": "black", "linewidth": 1.2}
OBSERVATION_RAW = {"color": "#c9c8c3", "linewidth": 0.4}
REFERENCE = {"color": "#8a8984", "linewidth": 0.8, "linestyle": (0, (4, 3))}
SHADE = "#ecebe7"  # background bands, e.g. the units of a borehole or a decade

DPI = 200
FULL_WIDTH_IN = 6.3  # 16 cm


def scheme_colour(scheme: str) -> str:
    """The colour of a densification scheme. The colour follows the scheme, never its position in a plot."""
    return SCHEME_COLOURS.get(scheme, OTHER_SCHEME_COLOUR)


def pair_style(pair: str, side: str) -> str:
    """The line style of one side of a pair, for example pair_style("rcm", "MAR") gives ALTERNATIVE."""
    current, alternative = PAIRS[pair]
    if side not in (current, alternative):
        raise ValueError(f"{side!r} is not a side of the pair {pair!r} {PAIRS[pair]}")
    return CURRENT if side == current else ALTERNATIVE


def apply() -> None:
    """Set the matplotlib defaults of the style. Call once before the figures are made."""
    mpl.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": DPI,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": TEXT_SECONDARY,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e6e5e1",
            "grid.linewidth": 0.5,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelcolor": TEXT,
            "ytick.labelcolor": TEXT,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "lines.linewidth": 1.3,
        }
    )


def panel_letters(axes: Iterable[plt.Axes], start: int = 0) -> None:
    """Write (a), (b), ... at the top left of each panel, outside the plot area."""
    for i, ax in enumerate(axes):
        ax.text(
            -0.02,
            1.02,
            f"({string.ascii_lowercase[start + i]})",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=TEXT,
        )


def depth_axis(ax: plt.Axes, bottom: float, top: float = 0.0) -> None:
    """Depth downwards, in m, from top to bottom."""
    ax.set_ylim(bottom, top)
    ax.set_ylabel("Depth (m)")


def save_figure(fig: plt.Figure, path: str) -> str:
    """Save a figure as a PNG with a white background, and close it. Returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
