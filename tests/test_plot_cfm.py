"""Tests for scripts/plot_cfm.py: the default scheme list (BUG-201) and the spin_end_year of the runs (FEAT-004)."""

import argparse
import importlib.util
from pathlib import Path

import pytest

import plot.results
from cfm_config import config
from force.years import RunYears
from sim.runs import add_spin_end_year_argument, spin_end_year_filter

_spec = importlib.util.spec_from_file_location(
    "plot_cfm", Path(__file__).resolve().parents[1] / "scripts" / "plot_cfm.py"
)
plot_cfm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot_cfm)


def test_default_phys_rho_is_one_name_in_a_list():
    assert plot_cfm.default_phys_rho({"physRho": "GSFC2020"}) == ["GSFC2020"]


def test_spin_end_year_argument(monkeypatch):
    monkeypatch.setitem(config, "spin_end_year", 1989)
    parser = argparse.ArgumentParser()
    add_spin_end_year_argument(parser)
    assert spin_end_year_filter(parser.parse_args([]).spin_end_year) == 1989  # the default: config.yaml
    assert spin_end_year_filter(parser.parse_args(["--spin-end-year", "1995"]).spin_end_year) == 1995
    assert spin_end_year_filter(parser.parse_args(["--spin-end-year", "any"]).spin_end_year) is None
    assert spin_end_year_filter(parser.parse_args(["--spin-end-year", "ANY"]).spin_end_year) is None
    with pytest.raises(SystemExit):
        parser.parse_args(["--spin-end-year", "1995.5"])


@pytest.fixture
def plotter_calls(tmp_path, monkeypatch):
    """ResultsPlotter with no RCM data and no runs: the filters that it gives to find_run."""
    calls = []

    class FoundRun:
        results_file = "CFMresults.hdf5"

    monkeypatch.setitem(config, "CFM_data_path", str(tmp_path))
    monkeypatch.setitem(config, "spin_end_year", 1989)
    monkeypatch.setattr(plot.results, "run_years", lambda rcm, start, end: RunYears(1979, 2024, 2024, "config"))
    monkeypatch.setattr(plot.results, "find_run", lambda **filters: calls.append(filters) or FoundRun())
    monkeypatch.setattr(plot.results, "shared_model_points", lambda sites, rcm: {})
    monkeypatch.setattr(plot.results, "model_point_label", lambda site, model_points: "")
    monkeypatch.setattr(plot.results, "cfm_version", lambda label: {"label": label or "ab871c8"})
    return calls


def test_plotter_uses_the_spin_end_year_of_config(plotter_calls):
    plot.results.ResultsPlotter("WI-70", ["GSFC2020"], "MAR", "bucket")
    assert plotter_calls[-1]["spin_end_year"] == 1989
    plot.results.ResultsPlotter("WI-70", ["GSFC2020"], "MAR", "bucket", cfm="3e3bf7e-speersm", spin_end_year=1995)
    assert plotter_calls[-1]["spin_end_year"] == 1995
    plot.results.ResultsPlotter("WI-70", ["GSFC2020"], "MAR", "bucket", spin_end_year="any")
    assert plotter_calls[-1]["spin_end_year"] is None
