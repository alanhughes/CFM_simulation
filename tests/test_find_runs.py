"""Tests for find_runs (src/sim/runs.py) and scripts/link_sites.py (FEAT-001). Run from the repo root.

The run folders are made with manifests only. The model point of a site is a stub, so no RCM data is read. The site
coordinate check uses the sites of the real sites.yaml.
"""

import importlib.util
import os
from pathlib import Path

import pytest

import sim.runs
from cfm_config import config
from cfm_config import sites as site_table
from sim.manifest import COMPLETE, FAILED, run_folder, write_manifest
from sim.runs import find_run, find_runs, runs_table

_spec = importlib.util.spec_from_file_location(
    "link_sites", Path(__file__).resolve().parents[1] / "scripts" / "link_sites.py"
)
link_sites = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(link_sites)

SITE_TAGS = {"CI-0": {"RACMO": "RACMO_261_126", "MAR": "MAR_1_2"}, "CI-0-twin": {"RACMO": "RACMO_261_126", "MAR": "MAR_9_9"}}


def make_run(
    tag,
    physRho="GSFC2020",
    cfm="ab871c8",
    hash_="a" * 64,
    status=COMPLETE,
    finished="2026-10-16T10:00:00",
    sites=("CI-0",),
    **run_keys,
):
    """A run folder with a manifest. run_keys are more keys of manifest["run"], for example site_coords."""
    rcm = link_sites.rcm_of_tag(tag)
    folder = run_folder(tag, physRho, "bucket", 1979, 2024, cfm, hash_)
    os.makedirs(folder)
    manifest = {
        "status": status,
        "run_hash": hash_,
        "run": {
            "sites": list(sites),
            "model_point": {"tag": tag},
            "rcm_name": rcm,
            "physRho": physRho,
            "liquid": "bucket",
            "start_year": 1979,
            "end_year": 2024,
            "spin_end_year": 1989,
            **run_keys,
        },
        "cfm": {"label": cfm, "commit": "c"},
        "finished": finished,
    }
    write_manifest(folder, manifest)
    return folder


@pytest.fixture(autouse=True)
def no_warned_sites(monkeypatch):
    """Each test starts with no site warnings given in this process."""
    monkeypatch.setattr(sim.runs, "_warned_sites", set())


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setitem(config, "CFM_data_path", str(tmp_path))
    monkeypatch.setattr(sim.runs, "_site_model_point", lambda site, rcm: SITE_TAGS[site][rcm])
    return {
        "gsfc": make_run("RACMO_261_126"),
        "crocus": make_run("RACMO_261_126", physRho="Crocus", hash_="b" * 64),
        "old_cfm": make_run("RACMO_261_126", cfm="3e3bf7e-speersm", hash_="c" * 64),
        "failed": make_run("RACMO_261_126", physRho="HLdynamic", hash_="d" * 64, status=FAILED),
        "mar": make_run("MAR_1_2", hash_="e" * 64),
        "other_cell": make_run("RACMO_100_100", hash_="f" * 64),
    }


def folders(found):
    return sorted(run.folder for run in found)


def test_filters(runs):
    assert folders(find_runs(physRho="GSFC2020", cfm="ab871c8", rcm_name="RACMO")) == sorted(
        [runs["gsfc"], runs["other_cell"]]
    )
    assert folders(find_runs(model_point="RACMO_261_126", physRho="Crocus")) == [runs["crocus"]]
    assert folders(find_runs(cfm="3e3bf7e-speersm")) == [runs["old_cfm"]]
    assert folders(find_runs(run_hash="eee")) == [runs["mar"]]


def test_complete_runs_only_by_default(runs):
    assert runs["failed"] not in folders(find_runs())
    assert folders(find_runs(status=FAILED)) == [runs["failed"]]
    assert len(find_runs(status=None)) == 6


def test_site_uses_its_model_point(runs):
    assert folders(find_runs(site="CI-0", rcm_name="RACMO", physRho="GSFC2020")) == sorted(
        [runs["gsfc"], runs["old_cfm"]]
    )
    # two sites in one cell share the run
    assert find_runs(site="CI-0-twin", physRho="Crocus")[0].folder == runs["crocus"]
    # with no rcm_name, the model point of the site in each RCM
    assert runs["mar"] in folders(find_runs(site="CI-0", physRho="GSFC2020"))


def test_find_run_gives_the_last_finished(runs, caplog):
    later = make_run("RACMO_261_126", hash_="9" * 64, finished="2026-10-17T10:00:00")
    assert find_run(site="CI-0", rcm_name="RACMO", physRho="GSFC2020", cfm="ab871c8").folder == later
    assert "2 runs" in caplog.text
    assert find_run(site="CI-0", rcm_name="RACMO", physRho="GSFC2020", run_hash="aaaa").folder == runs["gsfc"]
    with pytest.raises(LookupError, match="No run"):
        find_run(physRho="Ligtenberg2011")


def test_results_file_and_table(runs):
    run = find_run(physRho="Crocus")
    assert run.results_file == os.path.join(runs["crocus"], "CFMresults.hdf5")
    table = runs_table(find_runs())
    assert len(table) == 5
    assert set(table["model_point"]) == {"RACMO_261_126", "MAR_1_2", "RACMO_100_100"}


def test_folder_without_run_hash_is_ignored(runs, caplog):
    write_manifest(runs["gsfc"], {"status": COMPLETE})
    assert runs["gsfc"] not in folders(find_runs())
    assert "no run hash" in caplog.text


def test_link_sites(runs, tmp_path):
    runs_root, links = str(tmp_path / "runs"), str(tmp_path / "sites")
    assert link_sites.rcms_with_runs(runs_root) == ["MAR", "RACMO"]
    expected = link_sites.expected_links(SITE_TAGS, runs_root, links)
    changes = link_sites.sync_links(expected, links, ["MAR", "RACMO"])
    assert len(changes) == 3
    for site, rcm in [("CI-0", "RACMO"), ("CI-0-twin", "RACMO"), ("CI-0", "MAR")]:
        link = os.path.join(links, site, rcm)
        assert os.readlink(link) == f"../../runs/{SITE_TAGS[site][rcm]}"
        assert os.path.samefile(link, os.path.join(runs_root, SITE_TAGS[site][rcm]))
    # a second sync changes nothing
    assert link_sites.sync_links(expected, links, ["MAR", "RACMO"]) == []


def test_link_sites_corrects_wrong_links(runs, tmp_path):
    runs_root, links = str(tmp_path / "runs"), str(tmp_path / "sites")
    os.makedirs(os.path.join(links, "CI-0"))
    os.makedirs(os.path.join(links, "gone"))
    os.symlink("../../runs/RACMO_100_100", os.path.join(links, "CI-0", "RACMO"))
    os.symlink("../../runs/RACMO_100_100", os.path.join(links, "gone", "RACMO"))
    expected = link_sites.expected_links(SITE_TAGS, runs_root, links)
    link_sites.sync_links(expected, links, ["RACMO"])
    assert os.readlink(os.path.join(links, "CI-0", "RACMO")) == "../../runs/RACMO_261_126"
    assert not os.path.exists(os.path.join(links, "gone"))
    assert not os.path.exists(os.path.join(links, "CI-0", "MAR"))  # MAR was not asked for


def test_link_sites_dry_run_and_no_data_deleted(runs, tmp_path):
    runs_root, links = str(tmp_path / "runs"), str(tmp_path / "sites")
    expected = link_sites.expected_links(SITE_TAGS, runs_root, links)
    assert link_sites.sync_links(expected, links, ["RACMO"], dry_run=True)
    assert not os.path.exists(links)
    os.makedirs(os.path.join(links, "CI-0", "RACMO"))  # a real folder, not a link
    with pytest.raises(RuntimeError, match="not a symbolic link"):
        link_sites.sync_links(expected, links, ["RACMO"])


def site_warnings(caplog) -> list:
    return [record.getMessage() for record in caplog.records if "sites.yaml" in record.getMessage()]


def test_site_coords_that_agree_give_no_warning(runs, caplog, monkeypatch):
    make_run("RACMO_100_100", hash_="1" * 64, site_coords={"CI-0": [-66.403, -63.376]})
    reads = []
    monkeypatch.setattr(sim.runs, "site_table", lambda **kwargs: reads.append(kwargs) or site_table(**kwargs))
    assert len(find_runs()) == 6
    assert site_warnings(caplog) == []
    assert reads == [{"active_only": False}]  # sites.yaml is read once for each call, not once for each run


def test_site_with_other_coords_gives_one_warning(runs, caplog):
    folder = make_run("RACMO_100_100", hash_="1" * 64, site_coords={"CI-0": [-66.5, -63.376]})
    assert folder in folders(find_runs())  # the run is still returned
    (warning,) = site_warnings(caplog)
    assert folder in warning and "CI-0" in warning
    assert "-66.5" in warning and "-66.403" in warning and "run hash does not depend" in warning
    caplog.clear()
    assert folder in folders(find_runs(model_point="RACMO_100_100"))
    assert site_warnings(caplog) == []  # one warning for each run folder and site in a process


def test_shared_run_warns_for_the_site_with_other_coords_only(runs, caplog):
    folder = make_run(
        "RACMO_100_100",
        hash_="1" * 64,
        sites=["J409", "J109"],
        site_coords={"J409": [-68.395, -64.713], "J109": [-68.395, -64.72]},
    )
    find_runs()
    (warning,) = site_warnings(caplog)
    assert folder in warning and "site J109" in warning and "J409" not in warning


def test_coords_are_compared_to_3_decimals(runs, caplog):
    """A migrated JP21 run at -68.009 (BUG-001) agrees with -68.0086 in sites.yaml."""
    make_run("RACMO_100_100", hash_="1" * 64, sites=["JP21"], site_coords={"JP21": [-68.009, -64.367]})
    find_runs()
    assert site_warnings(caplog) == []


def test_old_manifest_compares_the_first_site(runs, caplog):
    """A manifest from before run.site_coords has the coordinates of its first site only."""
    old = {"sites": ["J409", "J109"], "borehole_lat": -68.395, "borehole_lon": -64.72}
    folder = make_run("RACMO_100_100", hash_="1" * 64, **old)
    find_runs()
    (warning,) = site_warnings(caplog)
    assert folder in warning and "site J409" in warning
    assert find_runs(model_point="RACMO_100_100")[0].site_coords == {"J409": (-68.395, -64.72)}


def test_site_not_in_sites_yaml_gives_a_warning(runs, caplog):
    folder = make_run("RACMO_100_100", hash_="1" * 64, sites=["CI-0", "NOPE"], site_coords={"NOPE": [-70.0, 0.0]})
    assert folder in folders(find_runs())
    (warning,) = site_warnings(caplog)
    assert folder in warning and "NOPE is not in sites.yaml" in warning


def test_only_the_runs_found_are_checked(runs, caplog):
    make_run("RACMO_100_100", hash_="1" * 64, status=FAILED, site_coords={"CI-0": [-66.5, -63.376]})
    make_run("MAR_5_5", hash_="2" * 64, site_coords={"CI-0": [-66.5, -63.376]})
    find_runs(rcm_name="RACMO")
    assert site_warnings(caplog) == []


def test_spin_end_year(runs, caplog):
    spin_1995 = make_run("RACMO_261_126", cfm="3e3bf7e-speersm", hash_="1" * 64, spin_end_year=1995)
    migrated_1 = make_run("RACMO_261_126", cfm="3e3bf7e-speersm", hash_="2" * 64, spin_end_year=None)
    old = dict(model_point="RACMO_261_126", physRho="GSFC2020", cfm="3e3bf7e-speersm")
    assert folders(find_runs(spin_end_year=1995, **old)) == [spin_1995]
    assert folders(find_runs(spin_end_year=1989, **old)) == [runs["old_cfm"]]
    assert folders(find_runs(**old)) == sorted([runs["old_cfm"], spin_1995, migrated_1])
    assert find_run(spin_end_year=1995, **old).spin_end_year == 1995
    # the warning of find_run gives the spin_end_year of each run
    find_run(**old)
    assert "spin_end_year 1995" in caplog.text and "spin_end_year 1989" in caplog.text
    assert "spin_end_year None" in caplog.text
    with pytest.raises(LookupError, match="spin_end_year=2000"):
        find_run(spin_end_year=2000, **old)
    table = runs_table(find_runs(**old))
    assert sorted(table["spin_end_year"].fillna(0)) == [0, 1989, 1995]
