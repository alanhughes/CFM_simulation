"""Tests for the helpers of scripts/migrate_forcing_csvs.py (FEAT-002). No luna data is read."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "migrate_forcing_csvs", Path(__file__).resolve().parents[1] / "scripts" / "migrate_forcing_csvs.py"
)
migrate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migrate)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("RACMO_-67.0_-61.486_1979_2024.csv", ("RACMO", "-67.0", "-61.486", "1979", "2024")),
        ("RACMO23_-66.403_-63.376_1979_2024.csv", ("RACMO23", "-66.403", "-63.376", "1979", "2024")),
        ("MAR_-68.0086_-64.3668_1979_2024.csv", ("MAR", "-68.0086", "-64.3668", "1979", "2024")),
    ],
)
def test_old_name(name, expected):
    match = migrate.OLD_NAME.match(name)
    assert (match["rcm"], match["lat"], match["lon"], match["start"], match["end"]) == expected


def test_complete_end_year():
    to_2023 = pd.date_range("1979-01-01 12:00", "2023-12-31 12:00", freq="D")
    assert migrate.complete_end_year(to_2023, 1979, 2024, "RACMO") == 2023
    assert migrate.complete_end_year(to_2023, 1979, 2023, "RACMO") == 2023
    late_start = pd.date_range("1980-06-01", "2023-12-31", freq="D")
    assert migrate.complete_end_year(late_start, 1979, 2024, "RACMO") is None


def test_dry_run_is_required():
    with pytest.raises(SystemExit):
        migrate.main(["/nowhere"])
