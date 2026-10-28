"""Tests of plot.forcing on synthetic forcing CSVs and a synthetic run output (Figure 10). No run is read."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from plot import forcing, style


@pytest.fixture
def ax():
    style.apply()
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def write_csv(path, time_column, hour, days):
    t = pd.date_range("1979-01-01", periods=days, freq="D") + pd.to_timedelta(hour, unit="h")
    pd.DataFrame({time_column: t.strftime("%Y-%m-%d %H:%M:%S"), "SMELT": 1.0, "SUBLIM": -0.5}).to_csv(path, index=False)


def test_read_forcing_csv_gives_the_same_days_for_racmo_and_mar(tmp_path):
    write_csv(tmp_path / "xpen.csv", "time", 0, 800)
    write_csv(tmp_path / "mar.csv", "TIME", 12, 800)
    a, b = forcing.read_forcing_csv(tmp_path / "xpen.csv"), forcing.read_forcing_csv(tmp_path / "mar.csv")
    assert a.index.name == b.index.name == "time"
    assert a.index.equals(b.index)
    assert (a.index == a.index.normalize()).all()


def test_sublimation_loss_sign():
    x = pd.DataFrame({"SUBLIM": [-2.0, 1.0]})
    assert list(forcing.sublimation_loss(x, "RACMO")) == [2.0, -1.0]
    assert list(forcing.sublimation_loss(x, "MAR")) == [-2.0, 1.0]
    with pytest.raises(ValueError):
        forcing.sublimation_loss(x, "RACMO23")


class FakeOutput:
    """time and per_step of a RunOutput: daily steps from 1980.0, with 2 mm w.e. of melt each day."""

    def __init__(self, days):
        self.time = 1980 + np.arange(days) / 365.25

    def per_step(self, name):
        assert name == "meltvol"
        return np.full(self.time.size, 0.002)


def test_cfm_daily_melt_starts_in_1980_on_the_forcing_days():
    days = pd.date_range("1979-01-01", "1982-12-31", freq="D")
    melt = forcing.cfm_daily_melt(FakeOutput(1000), days)
    assert melt.index[0] == pd.Timestamp("1980-01-01")
    assert len(melt) == 1000  # the output is shorter than the forcing
    assert melt.iloc[0] == pytest.approx(2.0)  # kg/m2


def test_annual_leaves_out_part_years():
    daily = pd.Series(1.0, index=pd.date_range("1980-01-01", "1981-06-30", freq="D"))
    totals = forcing.annual(daily)
    assert list(totals.index) == [1980] and totals[1980] == 366


def test_monthly_ratio():
    days = pd.date_range("1980-01-01", "1981-12-31", freq="D")
    rcm = pd.Series(2.0, index=days)
    cfm = pd.Series(np.where(days.month == 1, 1.0, 2.0), index=days)
    ratios = forcing.monthly_ratio(cfm, rcm)
    assert list(ratios) == [10, 11, 12, 1, 2, 3]
    assert ratios[1] == pytest.approx(0.5) and ratios[10] == pytest.approx(1.0)


def test_forcing_colour():
    assert style.forcing_colour("MAR") == style.FORCING_COLOURS["MAR"]
    assert not set(style.FORCING_COLOURS.values()) & set(style.SCHEME_COLOURS.values())
    with pytest.raises(ValueError):
        style.forcing_colour("GSFC2020")


def test_forcing_panels(ax):
    totals = pd.Series([280.0, 300.0], index=[1980, 1981])
    forcing.draw_annual(ax, totals, "ANT-12", label="ANT-12 (261, 126)")
    forcing.draw_annual(ax, totals * 1.4, "MAR", style.ALTERNATIVE)
    forcing.mark_date(ax, 2014.83, "borehole")
    assert ax.get_lines()[0].get_xdata()[0] == 1980.5
    assert ax.get_lines()[1].get_linestyle() == "--"
    fig, ax_b = plt.subplots()
    forcing.ratio_bars(
        ax_b, {"XPEN055": dict.fromkeys(forcing.MELT_SEASON, 0.3), "ANT-12": dict.fromkeys(forcing.MELT_SEASON, 0.2)}
    )
    assert len(ax_b.patches) == 12
    assert [t.get_text() for t in ax_b.get_xticklabels()] == ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
    assert sum(t.get_text() == "0.30" for t in ax_b.texts) == 6
    plt.close(fig)
