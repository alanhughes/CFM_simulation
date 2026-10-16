"""Tests for the site table sites.yaml and its lock file sites.lock (FEAT-002, BUG-005)."""

import datetime

import pytest
import yaml

from cfm_config import SITES_LOCK_FILE, batches, decimal_year, get_site, sites


def _lock() -> dict:
    with open(SITES_LOCK_FILE) as file:
        return yaml.safe_load(file)


def test_sites_agree_with_lock_file():
    """The name, lat and lon of a site do not change. A new site must also be added to sites.lock."""
    table = sites(active_only=False)
    lock = _lock()
    missing = sorted(set(lock) - set(table))
    assert not missing, f"Sites in sites.lock but not in sites.yaml: {missing}. Do not remove a site."
    not_locked = sorted(set(table) - set(lock))
    assert not not_locked, f"Add these new sites to sites.lock in the same commit: {not_locked}"
    changed = {
        name: {"lock": lock[name], "sites.yaml": [site.lat, site.lon]}
        for name, site in table.items()
        if [site.lat, site.lon] != [float(v) for v in lock[name]]
    }
    assert not changed, (
        f"Site coordinates changed: {changed}. Do not change a site; add a new site with a new name, and set "
        "'active: false' and 'superseded_by' on the old one."
    )


def test_superseded_sites_point_to_known_inactive_sites():
    table = sites(active_only=False)
    for site in table.values():
        if site.superseded_by is not None:
            assert site.superseded_by in table, site
            assert not site.active, f"{site.name} is superseded, so set 'active: false'"


def test_active_only():
    active = sites()
    everything = sites(active_only=False)
    assert list(active) == ["CI-0", "CI-22", "CI-120", "WI-0", "WI-70"]
    assert {"J109", "Y208", "JP21", "SI47"} <= set(everything) - set(active)


def test_batches_contain_known_sites():
    table = sites(active_only=False)
    for name, members in batches().items():
        unknown = sorted(set(members) - set(table))
        assert not unknown, f"Batch {name} has sites that are not in sites.yaml: {unknown}"


def test_get_site_unknown_name():
    with pytest.raises(KeyError, match="NOPE"):
        get_site("NOPE")


def test_profile_default():
    assert get_site("CI-0").profile_file == "CI-0_depth-density.csv"


@pytest.mark.parametrize(
    "site, old_windows",
    [
        # the windows that scripts/plot_density_error.py used for each batch before FEAT-002
        ("CI-0", [(2014 + 11 / 12, 2015 + 2 / 12), (2015 + 11 / 12, 2016 + 2 / 12)]),
        ("WI-70", [(2014 + 11 / 12, 2015 + 2 / 12), (2015 + 11 / 12, 2016 + 2 / 12)]),
        ("J208", [(2009 + 11 / 12, 2010 + 1 / 12)]),
        ("SI47", [(2022 + 11 / 12, 2023 + 1 / 12)]),
    ],
)
def test_windows_reproduce_old_batch_windows(site, old_windows):
    assert get_site(site).decimal_year_windows() == pytest.approx(old_windows, abs=1e-12)


def test_site_without_windows_is_an_error():
    """BUG-005: a site with no windows must not use the windows of a different site."""
    with pytest.raises(ValueError, match="Y208 has no windows"):
        get_site("Y208").decimal_year_windows()


def test_decimal_year():
    assert decimal_year(datetime.date(2015, 1, 1)) == 2015.0
    assert decimal_year(datetime.date(2014, 12, 16)) == pytest.approx(2014 + 11 / 12 + 15 / (12 * 31))


def test_windows_load_as_dates(tmp_path):
    path = tmp_path / "sites.yaml"
    path.write_text("sites:\n  A:\n    lat: -67.0\n    lon: -63.0\n    windows:\n      - [2015-12-01, 2015-11-01]\n")
    with pytest.raises(ValueError, match="ends before it starts"):
        sites(path=path)


def test_old_config_with_borehole_sites_is_an_error(tmp_path):
    """config.yaml is read from the working directory when cfm_config is imported."""
    import subprocess
    import sys
    from pathlib import Path

    import cfm_config

    old = Path("config.yaml").read_text() + "\nborehole_sites:\n  CI-0: [-66.403, -63.376]\n"
    (tmp_path / "config.yaml").write_text(old)
    src = str(Path(cfm_config.__file__).parent)
    result = subprocess.run(
        [sys.executable, "-c", "import cfm_config"], cwd=tmp_path, env={"PYTHONPATH": src}, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "borehole_sites" in result.stderr and "sites.yaml" in result.stderr
