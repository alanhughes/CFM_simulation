"""Tests for the years of a run for each RCM (DISC-001, option D), on empty files with the names of the real data."""

import logging

import pytest

from cfm_config import config
from force.cache import ForcingCoverageError
from force.model_point import clear_cache
from force.years import CLI, CONFIG, RunYears, data_days, run_years, set_config_years

# ANT-12 (RACMO) style: several years in each file. The column map gives the source variables tas, pr and prsn
RACMO_VARS = ("tas", "pr", "prsn")
RACMO_NAME = "{var}_ANT-12_TEST_day_{first}-{last}.nc"
RACMO_GROUPS = [("19790101", "19801231"), ("19810101", "20101231"), ("20110101", "20231231")]


def _files(folder, names):
    folder.mkdir(exist_ok=True)
    for name in names:
        (folder / name).touch()
    return folder


def _racmo(folder, groups=RACMO_GROUPS, variables=RACMO_VARS):
    return _files(folder, [RACMO_NAME.format(var=v, first=a, last=b) for v in variables for a, b in groups])


def _cfg(tmp_path, racmo=None, **changes):
    cfg = {
        "RACMO_data_path": str(racmo or _racmo(tmp_path / "racmo")),
        "RACMO_to_CFM_column_map": {"tas": "T2m", "rf": "RAIN"},
        "RACMO23_data_path": str(tmp_path / "racmo23"),
        "RACMO23_to_CFM_column_map": {"tas": "T2m"},
        "MAR_data_path": str(tmp_path / "mar"),
        "MAR_to_CFM_column_map": {"TTZ": "T2m", "ME": "SMELT"},
        "sector": 1.0,
        "start_year": 1979,
        "end_year": 2024,
        "spin_end_year": 1989,
    }
    cfg.update(changes)
    return cfg


@pytest.fixture(autouse=True)
def _clear_listings():
    clear_cache()
    yield
    clear_cache()


def test_default_end_year_is_clipped_to_the_data(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        years = run_years("RACMO", cfg=_cfg(tmp_path))
    assert years == RunYears(1979, 2023, 2024, CONFIG)
    assert years.clipped
    assert "RACMO (ANT-12_TEST_day) data stops at 2023-12-31: end_year 2023, not 2024 from config.yaml" in caplog.text


def test_default_end_year_in_the_data_is_not_clipped(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        years = run_years("RACMO", cfg=_cfg(tmp_path, end_year=2023))
    assert years == RunYears(1979, 2023, 2023, CONFIG)
    assert not years.clipped
    assert caplog.text == ""


def test_one_file_per_year(tmp_path):
    """MAR and RACMO23 (XPEN055) names have the year only; each file holds the whole year."""
    _files(tmp_path / "mar", [f"MARvTEST-27.5km-daily-ERA5-{year}.nc" for year in range(1979, 2025)])
    _files(tmp_path / "racmo23", [f"{v}.KNMI-{y}.XPEN055.TEST.DD.nc" for v in ("tas",) for y in range(1979, 2025)])
    assert run_years("MAR", cfg=_cfg(tmp_path)) == RunYears(1979, 2024, 2024, CONFIG)
    assert run_years("RACMO23", cfg=_cfg(tmp_path)) == RunYears(1979, 2024, 2024, CONFIG)
    assert run_years("RACMO23", cfg=_cfg(tmp_path, end_year=2030)).end_year == 2024


def test_command_line_years_are_used_as_given(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert run_years("RACMO", 1985, 2010, cfg=_cfg(tmp_path)) == RunYears(1985, 2010, 2010, CLI)
        assert run_years("RACMO", end_year=2023, cfg=_cfg(tmp_path)) == RunYears(1979, 2023, 2023, CLI)
    assert caplog.text == ""


def test_command_line_end_year_after_the_data_is_an_error(tmp_path):
    with pytest.raises(ForcingCoverageError, match=r"stops at 2023-12-31, before December 2024 \(--end-year\)"):
        run_years("RACMO", end_year=2024, cfg=_cfg(tmp_path))


def test_data_that_starts_after_the_start_year_is_an_error(tmp_path):
    late = _racmo(tmp_path / "late", groups=[("19800101", "20231231")])
    with pytest.raises(ForcingCoverageError, match="starts at 1980-01-01, after January 1979"):
        run_years("RACMO", cfg=_cfg(tmp_path, racmo=late))
    with pytest.raises(ForcingCoverageError, match="after January 1979"):  # also a start year from the command line
        run_years("RACMO", start_year=1979, end_year=2020, cfg=_cfg(tmp_path, racmo=late))
    assert run_years("RACMO", start_year=1980, cfg=_cfg(tmp_path, racmo=late)).start_year == 1980

    mid_january = _racmo(tmp_path / "mid", groups=[("19790115", "20231231")])
    with pytest.raises(ForcingCoverageError, match="starts at 1979-01-15"):
        run_years("RACMO", cfg=_cfg(tmp_path, racmo=mid_january))


def test_partial_last_year_gives_the_year_before(tmp_path):
    partial = _racmo(tmp_path / "partial", groups=[("19790101", "20101231"), ("20110101", "20240630")])
    assert data_days("RACMO", _cfg(tmp_path, racmo=partial)).last_year == 2023
    assert run_years("RACMO", cfg=_cfg(tmp_path, racmo=partial)) == RunYears(1979, 2023, 2024, CONFIG)
    with pytest.raises(ForcingCoverageError):  # 2024 is not complete
        run_years("RACMO", end_year=2024, cfg=_cfg(tmp_path, racmo=partial))


def test_the_variable_that_stops_first_sets_the_end(tmp_path):
    folder = _racmo(tmp_path / "racmo", groups=RACMO_GROUPS + [("20240101", "20241231")], variables=("tas", "pr"))
    _racmo(folder, variables=("prsn",))  # prsn stops at 2023-12-31
    assert run_years("RACMO", cfg=_cfg(tmp_path, racmo=folder)).end_year == 2023


def test_clipped_end_year_before_spin_end_year_is_an_error(tmp_path):
    short = _racmo(tmp_path / "short", groups=[("19790101", "19881231")])
    with pytest.raises(ForcingCoverageError, match="end_year 1988 would be before spin_end_year 1989"):
        run_years("RACMO", cfg=_cfg(tmp_path, racmo=short))
    assert run_years("RACMO", cfg=_cfg(tmp_path, racmo=short, spin_end_year=1988)).end_year == 1988


def test_start_year_after_end_year_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="start year 2020 is after end year 2010"):
        run_years("RACMO", 2020, 2010, cfg=_cfg(tmp_path))


def test_unknown_rcm(tmp_path):
    with pytest.raises(ValueError, match="Unknown rcm_name 'MERRA2'"):
        run_years("MERRA2", cfg=_cfg(tmp_path))


def test_set_config_years(monkeypatch):
    monkeypatch.setitem(config, "start_year", 1979)
    monkeypatch.setitem(config, "end_year", 2024)
    set_config_years(RunYears(1979, 2023, 2024, CONFIG))
    assert (config["start_year"], config["end_year"]) == (1979, 2023)
