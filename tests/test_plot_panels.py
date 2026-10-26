"""Smoke tests of the panel functions of plot.profiles, plot.summary and plot.timeseries on synthetic data (FEAT-007).
They check that each function draws without an error, and a few properties of the result. No run is read."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from plot import metrics, profiles, style, summary, timeseries

DEPTH = np.linspace(2.0, 50.0, 200)


@pytest.fixture
def ax():
    style.apply()
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def comparison():
    observed = 900.0 - 300.0 * np.exp(-DEPTH / 10)
    mean = observed - 50
    return metrics.ProfileComparison(DEPTH, mean, mean - 10, mean + 10, observed, steps=3)


def test_profile_panel(ax):
    profile = metrics.Profile("X", DEPTH, comparison().observed)
    profiles.shade_depths(ax, [(2.9, 20, "unit 1"), (20, 50, "unit 2")])
    profiles.draw_observed(ax, profile)
    profiles.draw_comparison(ax, comparison(), "GSFC2020", band=True)
    profiles.draw_profile(ax, DEPTH * 0 - 2, DEPTH, "Crocus", style.ALTERNATIVE)
    profiles.reference_value(ax, 830, "830")
    profiles.reference_value(ax, 700, "left", side="left")
    profiles.mark_point(ax, 850, 11, "measured")
    profiles.depth_axes(ax, 50, "Density (kg/m³)", xlim=(300, 950))
    profiles.scheme_legend(ax)
    assert ax.get_ylim() == (50, 0)  # depth downwards
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "GSFC2020" in labels and "_raw" not in labels


def test_scheme_colour_follows_the_scheme():
    assert style.scheme_colour("Crocus") == style.SCHEME_COLOURS["Crocus"]
    assert style.scheme_colour("unknown") == style.OTHER_SCHEME_COLOUR
    assert style.pair_style("rcm", "MAR") == style.ALTERNATIVE
    with pytest.raises(ValueError):
        style.pair_style("rcm", "RACMO23")


def test_effect_panel(ax):
    rows = [
        summary.EffectRow("one scheme", {"GSFC2020": 4.6}, 10.0),
        summary.EffectRow("four schemes", {"GSFC2020": -11.7, "Crocus": -14.4, "Barnola1991": -5.7}, 7.7),
    ]
    summary.effect_panel(ax, rows, "ΔFAC", "{:+.1f}", "m", scaled=True, band=(-180, -120, 60, "bias"))
    texts = [t.get_text() for t in ax.texts]
    assert "+4.6 m" in texts and "-14.4 to -5.7 m" in texts


def test_row_text_has_no_signed_zero():
    assert summary._row_text([-0.01, 0.02], "{:+.1f}", "m") == "0 m"
    assert summary._row_text([-0.01, 0.3], "{:+.1f}", "m") == "0 to +0.3 m"


def test_budget_bars(ax):
    budget = {"accumulation": 600.0, "melt": 80.0, "refreeze": 90.0, "runoff": 0.0, "refreeze_fraction": 0.15}
    summary.budget_bars(ax, ["a", "b"], [[budget, budget], [budget]])
    assert any(t.get_text() == "15 %" for t in ax.texts)


def test_timeseries_panel(ax):
    time = np.linspace(1980, 2024, 100)
    timeseries.shade_years(ax, 1990, 2000, "1990-2000")
    timeseries.draw_series(ax, time, np.sin(time), "GSFC2020")
    timeseries.draw_series(ax, time, np.cos(time), "HLdynamic", thin=True)
    timeseries.zero_line(ax)
    timeseries.end_label(ax, time, np.sin(time), "end")
    timeseries.time_axes(ax, "m", 1980, 2024)
    assert ax.get_xlim() == (1980, 2024)
