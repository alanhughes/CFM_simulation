"""The RCM forcing at a site, and the CFM melt against the RCM melt (Figure 10, F-013, F-014).

The forcing CSVs have one row per day. The first column is the time: "time" in the RACMO CSVs and "TIME" in the MAR
ones; XPEN055 is stamped 00:00 and ANT-12 and MAR 12:00, on the same days. read_forcing_csv() gives all of them a daily
index at 00:00. The amounts (SMELT, BDOT, RAIN, SUBLIM) are kg/m2 per day.

The CFM output starts at MAIN_RUN_START (1980.0) with one step per forcing day, so the CFM melt of a day is the
meltvol of the step with the same position from 1980 (as in xpen055_ci0.py of A-001).

The drawing functions draw on a matplotlib Axes. Colour is the forcing (style.FORCING_COLOURS); line style is the RCM
pair (RACMO solid, MAR dashed).
"""

from typing import Dict, Iterable, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot import style
from plot.output import MAIN_RUN_START

MONTH_NAMES = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}
MONTH_NAMES.update({11: "Nov", 12: "Dec"})
MELT_SEASON = (10, 11, 12, 1, 2, 3)


def read_forcing_csv(path: str) -> pd.DataFrame:
    """A forcing CSV with a daily index at 00:00 named "time"."""
    x = pd.read_csv(path)
    x = x.rename(columns={x.columns[0]: "time"})  # "time" for RACMO, "TIME" for MAR
    x["time"] = pd.to_datetime(x["time"]).dt.normalize()
    return x.set_index("time")


def sublimation_loss(x: pd.DataFrame, rcm: str) -> pd.Series:
    """SUBLIM as a mass loss (positive = loss), kg/m2 per day. RACMO is negative for a loss, MAR positive."""
    sign = {"RACMO": -1.0, "MAR": 1.0}
    if rcm not in sign:
        raise ValueError(f"Unknown RCM {rcm!r}: expected RACMO or MAR")
    return sign[rcm] * x["SUBLIM"]


def cfm_daily_melt(out, days: pd.DatetimeIndex) -> pd.Series:
    """The CFM melt (meltvol) of each forcing day from 1980, in kg/m2. out is a RunOutput (time, per_step)."""
    days = days[days >= pd.Timestamp(int(MAIN_RUN_START), 1, 1)]
    melt = 1000 * np.asarray(out.per_step("meltvol"))[np.asarray(out.time) >= MAIN_RUN_START]
    n = min(len(days), len(melt))
    return pd.Series(melt[:n], index=days[:n], name="cfm")


def annual(daily: pd.Series) -> pd.Series:
    """Calendar-year totals of a daily amount, indexed by the year. Years with fewer than 365 days are left out."""
    years = daily.groupby(daily.index.year)
    total = years.sum()
    return total[years.count() >= 365]


def monthly_ratio(cfm: pd.Series, rcm: pd.Series, months: Iterable[int] = MELT_SEASON) -> Dict[int, float]:
    """The CFM melt over the RCM melt of each calendar month, summed over all the years both have."""
    j = pd.DataFrame({"cfm": cfm, "rcm": rcm}).dropna()
    sums = j.groupby(j.index.month).sum()
    return {m: sums.cfm[m] / sums.rcm[m] for m in months}


def draw_annual(
    ax: plt.Axes,
    totals: pd.Series,
    forcing: str,
    linestyle: str = style.CURRENT,
    label: Optional[str] = None,
    **kwargs,
) -> None:
    """Annual totals in the colour of the forcing, each at the middle of its year."""
    ax.plot(
        totals.index + 0.5,
        totals.values,
        color=style.forcing_colour(forcing),
        linestyle=linestyle,
        label=label or forcing,
        **kwargs,
    )


def mark_date(ax: plt.Axes, decimal_year: float, label: str) -> None:
    """A vertical reference line at a date, labelled at its top, to the left of the line."""
    ax.axvline(decimal_year, zorder=1, **style.REFERENCE)
    ax.annotate(
        label,
        (decimal_year, 1),
        xycoords=("data", "axes fraction"),
        xytext=(-3, -2),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=7,
        color=style.TEXT_SECONDARY,
    )


def ratio_bars(
    ax: plt.Axes,
    ratios: Mapping[str, Mapping[int, float]],
    months: Iterable[int] = MELT_SEASON,
    labels: Optional[Mapping[str, str]] = None,
) -> None:
    """Grouped bars of a monthly ratio, one bar per forcing in each month, with the value above each bar, and a
    reference line at 1."""
    months = list(months)
    forcings = list(ratios)
    width = 0.8 / len(forcings)
    x = np.arange(len(months))
    for i, forcing in enumerate(forcings):
        values = [ratios[forcing][m] for m in months]
        position = x + (i - (len(forcings) - 1) / 2) * width
        ax.bar(
            position,
            values,
            width * 0.92,  # a small gap between neighbouring bars
            color=style.forcing_colour(forcing),
            label=(labels or {}).get(forcing, forcing),
            zorder=2,
        )
        for p, v in zip(position, values):
            ax.annotate(
                f"{v:.2f}",
                (p, v),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6,
                color=style.TEXT,
            )
    ax.axhline(1, zorder=1, **style.REFERENCE)
    ax.set_xticks(x, [MONTH_NAMES[m] for m in months])
    ax.grid(axis="x", visible=False)
