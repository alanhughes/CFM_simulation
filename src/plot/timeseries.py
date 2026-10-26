"""Time series panels: surface height, firn air content and differences of a pair against time (FEAT-007).

The functions draw on a matplotlib Axes. Colour is the scheme and line style is the side of a pair (plot.style).
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from plot import style


def draw_series(
    ax: plt.Axes,
    time: np.ndarray,
    values: np.ndarray,
    scheme: str,
    linestyle: str = style.CURRENT,
    label: Optional[str] = None,
    thin: bool = False,
    **kwargs,
) -> None:
    """One time series in the colour of its scheme. thin draws a light secondary line, for example the other schemes
    behind a highlighted pair."""
    if thin:
        kwargs = {"linewidth": 0.7, "alpha": 0.6, **kwargs}
    ax.plot(time, values, color=style.scheme_colour(scheme), linestyle=linestyle, label=label or scheme, **kwargs)


def shade_years(ax: plt.Axes, start: float, end: float, label: Optional[str] = None) -> None:
    """A grey band for a period, for example a decade, labelled at its bottom (legends are usually at the top)."""
    ax.axvspan(start, end, color=style.SHADE, zorder=0, lw=0)
    if label:
        ax.annotate(
            label,
            ((start + end) / 2, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color=style.TEXT_SECONDARY,
        )


def zero_line(ax: plt.Axes) -> None:
    ax.axhline(0, zorder=1, **style.REFERENCE)


def time_axes(ax: plt.Axes, ylabel: str, start: Optional[float] = None, end: Optional[float] = None) -> None:
    ax.set_ylabel(ylabel)
    if start is not None or end is not None:
        ax.set_xlim(start, end)


def end_label(ax: plt.Axes, time: np.ndarray, values: np.ndarray, text: str, dy_points: float = 0) -> None:
    """A label at the end of a line, in text colour (the line beside it carries the identity)."""
    ax.annotate(
        text,
        (time[-1], values[-1]),
        xytext=(4, dy_points),
        textcoords="offset points",
        va="center",
        fontsize=7,
        color=style.TEXT,
    )
