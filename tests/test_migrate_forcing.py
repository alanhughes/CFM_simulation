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


def test_dry_run_and_copy_exclude_each_other():
    with pytest.raises(SystemExit):
        migrate.main(["--dry-run", "--copy", "/nowhere"])


# --- copy mode (FEAT-002 step 8): temporary folders only ---

CSV_TEXT = "time,T2m\n1979-01-01,250.0\n1979-12-31,251.0\n"


def make_plan(tmp_path, name="RACMO_-66.403_-63.376_1979_2023.csv", text=CSV_TEXT, action="COPY", cell=(5, 7)):
    from force.model_point import ModelPoint

    source = tmp_path / "old" / name
    source.parent.mkdir(exist_ok=True)
    source.write_text(text)
    mp = ModelPoint("RACMO", "rel", cell[0], cell[1], -66.4, -63.4, 3.0)
    target = tmp_path / "cache" / "RACMO" / "rel" / f"{cell[0]}_{cell[1]}_1979_2023.csv"
    return migrate.Plan(
        source=str(source),
        rcm="RACMO",
        site="S",
        model_point=mp,
        old_cell=cell,
        target=str(target),
        action=action,
        sha256=migrate.sha256_file(str(source)),
    )


def test_copy_writes_csv_and_json_and_keeps_source(tmp_path):
    import json

    plan = make_plan(tmp_path)
    assert migrate.copy_all([plan]) == 0
    target = Path(plan.target)
    assert target.read_bytes() == Path(plan.source).read_bytes()
    assert Path(plan.source).exists()
    record = json.loads(target.with_suffix(".json").read_text())
    assert record["csv_sha256"] == plan.sha256 == migrate.sha256_file(plan.target)
    assert (record["start_year"], record["end_year"]) == (1979, 2023)
    assert record["model_point"]["tag"] == "RACMO_5_7"
    assert not [p for p in target.parent.iterdir() if ".tmp" in p.name]


def test_copy_skips_everything_but_copy(tmp_path):
    plans = [make_plan(tmp_path, action=a) for a in ("SKIP", "DUPLICATE", "EXISTS")]
    assert migrate.copy_all(plans) == 0
    assert not (tmp_path / "cache").exists()


def test_copy_refuses_a_different_file_in_the_cache(tmp_path):
    plan = make_plan(tmp_path)
    Path(plan.target).parent.mkdir(parents=True)
    Path(plan.target).write_text("other bytes")
    with pytest.raises(RuntimeError):
        migrate.copy_plan(plan)
    assert Path(plan.target).read_text() == "other bytes"


def test_conflict_stops_before_any_copy(tmp_path):
    good = make_plan(tmp_path, cell=(1, 2))
    bad = make_plan(tmp_path, name="RACMO_-67.0_-61.486_1979_2023.csv", action="CONFLICT", cell=(3, 4))
    assert migrate.copy_all([good, bad]) == 1
    assert not (tmp_path / "cache").exists()
