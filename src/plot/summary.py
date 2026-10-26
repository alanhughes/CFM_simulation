"""Summary panels across many runs: the size of each effect, and the water budget (FEAT-007)."""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np

from plot import style


@dataclass
class EffectRow:
    """One source of difference for one quantity, for example the CFM version for FAC.

    values gives the effect of each scheme in physical units (one entry if the pair has one scheme). spread is the range
    across the schemes at the same site, in the same units, as the yardstick of the effect.
    """

    label: str
    values: Dict[str, float]
    spread: float


def _number(value: float, fmt: str) -> str:
    """A number in fmt (for example "{:+.1f}"), with 0 for a value that rounds to zero (not -0.0 or +0.0)."""
    text = fmt.format(value)
    return "0" if float(text) == 0 else text


def _row_text(values: Sequence[float], fmt: str, unit: str = "") -> str:
    """One value, or the range "lo to hi", with the unit once at the end."""
    lo, hi = _number(min(values), fmt), _number(max(values), fmt)
    text = lo if lo == hi else f"{lo} to {hi}"
    return f"{text} {unit}".strip()


def effect_panel(
    ax: plt.Axes,
    rows: Sequence[EffectRow],
    title: str,
    fmt: str,
    unit: str,
    scaled: bool,
    band: Optional[Tuple[float, float, float, str]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    show_labels: bool = True,
) -> None:
    """A dot plot of the effects, one row each, top to bottom.

    scaled True: x is the effect divided by the scheme spread of its row, with grey lines at -1 and +1 (an effect as
    large as the full spread). scaled False: x is in physical units, and a grey band behind each row spans plus and
    minus its spread. fmt and unit give the physical value written at the right of each row, for example "{:+.1f}"
    and "m". show_labels False hides the row labels (for a panel that shares its y axis with a labelled panel).
    band is an extra vertical band (low, high, spread, label) in physical units, for example the model bias; with
    scaled True it is divided by spread.
    """
    n = len(rows)
    y = np.arange(n)[::-1]
    if scaled:
        for x in (-1, 1):
            ax.axvline(x, zorder=1, **style.REFERENCE)
    ax.axvline(0, color=style.TEXT_SECONDARY, lw=0.6, zorder=1)
    if band is not None:
        low, high, spread, label = band
        if scaled:
            low, high = low / spread, high / spread
        ax.axvspan(low, high, color=style.SHADE, zorder=0, lw=0)
        ax.annotate(
            label,
            ((low + high) / 2, 0.5),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="center",
            rotation=90,
            fontsize=7,
            color=style.TEXT_SECONDARY,
        )
    for yi, row in zip(y, rows):
        x = {scheme: (v / row.spread if scaled else v) for scheme, v in row.values.items()}
        if not scaled:
            ax.fill_betweenx([yi - 0.3, yi + 0.3], -row.spread, row.spread, color=style.SHADE, lw=0, zorder=0)
        if len(x) > 1:
            ax.plot([min(x.values()), max(x.values())], [yi, yi], color="#b5b4af", lw=2.5, zorder=2,
                    solid_capstyle="round")
        for scheme, xv in x.items():
            ax.plot(xv, yi, "o", ms=6, color=style.scheme_colour(scheme), mec="white", mew=0.8, zorder=3)
        ax.annotate(
            _row_text(list(row.values.values()), fmt, unit),
            (1, yi),
            xycoords=("axes fraction", "data"),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
            color=style.TEXT,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([row.label for row in rows])
    ax.tick_params(axis="y", labelleft=show_labels, length=0)
    ax.set_ylim(-0.6, n - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left")
    if xlim is not None:
        ax.set_xlim(*xlim)


def scheme_key(fig: plt.Figure, schemes: Sequence[str], loc: str = "lower center", ncol: int = 5) -> None:
    """A figure legend of the scheme colours, as dots."""
    handles = [
        plt.Line2D([], [], marker="o", ls="none", ms=6, color=style.scheme_colour(s), mec="white", mew=0.8)
        for s in schemes
    ]
    fig.legend(handles, schemes, loc=loc, ncol=ncol, frameon=False, fontsize=7.5)


def budget_bars(
    ax: plt.Axes,
    groups: Sequence[str],
    budgets: Sequence[Sequence[Dict[str, float]]],
    terms: Sequence[str] = ("accumulation", "melt", "refreeze", "runoff"),
) -> None:
    """Grouped horizontal bars of the water budget, kg/m2 per year: one group per set of runs (top to bottom), one bar
    per term. Each bar is the mean across the runs of the group, with a whisker for their range. The refreezing as a
    percentage of the accumulation is written at the end of the refreezing bar."""
    height = 0.8 / len(terms)
    for g, (group, runs) in enumerate(zip(groups, budgets)):
        base = len(groups) - 1 - g
        for k, term in enumerate(terms):
            values = np.array([run[term] for run in runs])
            yk = base + 0.4 - height * (k + 0.5)
            ax.barh(yk, values.mean(), height=height * 0.85, color=style.BUDGET_COLOURS[term],
                    label=term if g == 0 else None, zorder=2)
            if np.allclose(values, 0):
                ax.annotate("0", (0, yk), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7,
                            color=style.TEXT_SECONDARY)
            if len(values) > 1 and np.ptp(values) > 0:
                ax.plot([values.min(), values.max()], [yk, yk], color=style.TEXT_SECONDARY, lw=0.8, zorder=3)
            if term == "refreeze":
                fraction = np.array([run["refreeze_fraction"] for run in runs])
                ax.annotate(
                    _row_text(list(100 * fraction), "{:.0f}", "%"),
                    (values.max(), yk),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=7,
                    color=style.TEXT,
                )
    ax.set_yticks(np.arange(len(groups))[::-1])
    ax.set_yticklabels(groups)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("kg/m² per year")
    names = {"accumulation": "Accumulation", "melt": "Melt", "refreeze": "Refreezing (incl. rain)", "runoff": "Runoff"}
    handles = [mpl.patches.Patch(color=style.BUDGET_COLOURS[t], label=names[t]) for t in terms]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(terms))
