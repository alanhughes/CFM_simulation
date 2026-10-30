"""Tests for find_runs (src/sim/runs.py) and scripts/link_sites.py (FEAT-001). Run from the repo root.

The run folders are made with manifests only. The model point of a site is a stub, so no RCM data is read. The site
coordinate check uses the sites of the real sites.yaml.

The tests at the end use two run roots, the local root and a share root in tmp_path (FEAT-008). The real share is never
read.
"""

import errno
import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

import sim.manifest
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

_spec = importlib.util.spec_from_file_location(
    "find_runs_script", Path(__file__).resolve().parents[1] / "scripts" / "find_runs.py"
)
find_runs_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(find_runs_script)

SITE_TAGS = {"CI-0": {"RACMO": "RACMO_261_126", "MAR": "MAR_1_2"}, "CI-0-twin": {"RACMO": "RACMO_261_126", "MAR": "MAR_9_9"}}


def make_run(
    tag,
    physRho="GSFC2020",
    cfm="ab871c8",
    hash_="a" * 64,
    status=COMPLETE,
    finished="2026-10-16T10:00:00",
    sites=("CI-0",),
    cfm_config=None,
    **run_keys,
):
    """A run folder with a manifest. run_keys are more keys of manifest["run"], for example site_coords. cfm_config is
    the cfm_config of the manifest; None gives a manifest without it, as a migrated run."""
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
    if cfm_config is not None:
        manifest["cfm_config"] = cfm_config
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


def test_meltwater_solver(runs):
    ncz = make_run("RACMO_261_126", hash_="8" * 64, finished="2026-10-17T10:00:00",
                   cfm_config={"meltwater_solver": "ncz"})
    no_key = make_run("RACMO_261_126", hash_="7" * 64, cfm_config={"physRho": "GSFC2020"})
    filters = dict(model_point="RACMO_261_126", physRho="GSFC2020", cfm="ab871c8")
    # a run without the key (no cfm_config, or a cfm_config without meltwater_solver) used the default, enthalpy
    assert folders(find_runs(meltwater_solver="enthalpy", **filters)) == sorted([runs["gsfc"], no_key])
    assert folders(find_runs(meltwater_solver="ncz", **filters)) == [ncz]
    assert len(find_runs(**filters)) == 3
    # the runs differ in meltwater_solver, as the runs without the key are enthalpy (FEAT-009), so find_run stops
    with pytest.raises(LookupError, match="cfm_config key meltwater_solver"):
        find_run(**filters)
    assert find_run(meltwater_solver="ncz", **filters).folder == ncz
    table = runs_table(find_runs(**filters))
    assert sorted(table["meltwater_solver"]) == ["enthalpy", "enthalpy", "ncz"]


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


def test_no_run_gives_the_spin_end_year(runs):
    with pytest.raises(LookupError, match="Only the runs with spin_end_year 1995 agree"):
        find_run(physRho="Crocus", spin_end_year=1995)


def test_distance_km_for_each_site(runs):
    """The distance from each site of a run to its cell centre (FEAT-007), with the real values of the RACMO cell
    (259, 137), which WI-70 and Y208 share before the BUG-002 fix."""
    folder = make_run(
        "RACMO_259_137",
        hash_="d" * 64,
        sites=("WI-70", "Y208"),
        model_point={"tag": "RACMO_259_137", "cell_lat": -67.55758973191408, "cell_lon": -63.36961198220811},
        site_coords={"WI-70": [-67.5, -63.336], "Y208": [-67.564, -63.257]},
    )
    run = sim.runs.Run(folder, sim.runs.read_manifest(folder))
    assert run.distance_km("WI-70") == pytest.approx(6.56, abs=0.01)
    assert run.distance_km("Y208") == pytest.approx(4.8, abs=0.1)
    assert run.distance_km("CI-0") is None


def test_distance_km_without_a_cell_centre(runs):
    folder = make_run("RACMO_259_137", hash_="e" * 64, site_coords={"CI-0": [-66.403, -63.376]})
    run = sim.runs.Run(folder, sim.runs.read_manifest(folder))
    assert run.distance_km("CI-0") is None


# two run roots: the local runs and the copy of the runs on the share (FEAT-008). The share is a folder in tmp_path; a
# share that is not available is a stub of os.stat that raises OSError


@pytest.fixture
def share(tmp_path, monkeypatch, runs):
    """CFM_share_data_path in tmp_path, with a runs/ folder. The local runs are those of the runs fixture."""
    data = tmp_path / "share"
    (data / "runs").mkdir(parents=True)
    monkeypatch.setitem(config, "CFM_share_data_path", str(data))
    return data


def make_share_run(share, tag, **keys):
    """A run folder with a manifest in the share root, as make_run makes it in the local root."""
    local = config["CFM_data_path"]
    config["CFM_data_path"] = str(share)
    try:
        return make_run(tag, **keys)
    finally:
        config["CFM_data_path"] = local


def by_folder(found):
    return {run.folder: run for run in found}


def test_local_run_only(share, runs):
    found = by_folder(find_runs())
    assert len(found) == 5
    assert {run.location for run in found.values()} == {"local"}
    assert found[runs["crocus"]].results_file == os.path.join(runs["crocus"], "CFMresults.hdf5")


def test_share_run_only(share, runs):
    folder = make_share_run(share, "RACMO_300_300", hash_="1" * 64, cfm_config={"resultsFileName": "out.hdf5"})
    assert folder.startswith(str(share / "runs"))
    (run,) = find_runs(model_point="RACMO_300_300")
    assert run.location == "share"
    assert run.folder == folder
    assert run.results_file == os.path.join(folder, "out.hdf5")
    assert len(find_runs()) == 6


def test_run_in_both_roots(share, runs):
    folder = make_share_run(share, "RACMO_261_126", physRho="Crocus", hash_="b" * 64)
    found = find_runs(physRho="Crocus")
    (run,) = found  # no duplicate
    assert run.location == "both"
    assert run.folder == runs["crocus"] != folder
    assert run.results_file == os.path.join(runs["crocus"], "CFMresults.hdf5")
    assert find_run(physRho="Crocus").location == "both"
    assert len(find_runs()) == 5
    assert sorted(run.location for run in find_runs()) == ["both", "local", "local", "local", "local"]


def test_share_copy_when_the_local_copy_does_not_agree(share, runs):
    """A run hash that is failed in the local root and complete on the share: the share copy."""
    folder = make_share_run(share, "RACMO_261_126", physRho="HLdynamic", hash_="d" * 64)
    (run,) = find_runs(physRho="HLdynamic")
    assert (run.folder, run.location) == (folder, "share")
    # with all the statuses, the two copies agree: the local copy
    (run,) = find_runs(physRho="HLdynamic", status=None)
    assert (run.folder, run.location) == (runs["failed"], "both")


def test_no_local_runs_folder(tmp_path, monkeypatch):
    monkeypatch.setitem(config, "CFM_data_path", str(tmp_path / "local"))  # no runs/ folder
    monkeypatch.setitem(config, "CFM_share_data_path", str(tmp_path / "share"))
    monkeypatch.setattr(sim.runs, "_site_model_point", lambda site, rcm: SITE_TAGS[site][rcm])
    folder = make_share_run(tmp_path / "share", "RACMO_261_126")
    assert not os.path.exists(tmp_path / "local" / "runs")
    (run,) = find_runs(site="CI-0")
    assert (run.folder, run.location) == (folder, "share")


def test_filters_across_both_roots(share, runs):
    share_only = {
        "gsfc": make_share_run(share, "RACMO_261_126", hash_="1" * 64),
        "old_cfm": make_share_run(share, "RACMO_261_126", cfm="3e3bf7e-speersm", hash_="2" * 64),
        "mar": make_share_run(share, "MAR_1_2", hash_="3" * 64),
        "failed": make_share_run(share, "RACMO_100_100", hash_="4" * 64, status=FAILED),
    }
    make_share_run(share, "RACMO_261_126", physRho="Crocus", hash_="b" * 64)  # also local
    assert folders(find_runs(site="CI-0", rcm_name="RACMO", physRho="GSFC2020")) == sorted(
        [runs["gsfc"], runs["old_cfm"], share_only["gsfc"], share_only["old_cfm"]]
    )
    assert folders(find_runs(model_point="MAR_1_2")) == sorted([runs["mar"], share_only["mar"]])
    assert folders(find_runs(cfm="3e3bf7e-speersm")) == sorted([runs["old_cfm"], share_only["old_cfm"]])
    assert folders(find_runs(run_hash="333")) == [share_only["mar"]]
    assert folders(find_runs(status=FAILED)) == sorted([runs["failed"], share_only["failed"]])
    assert folders(find_runs(physRho="Crocus")) == [runs["crocus"]]
    assert len(find_runs(status=None)) == 6 + 4


def test_explicit_root_is_one_root(share, runs):
    folder = make_share_run(share, "RACMO_300_300", hash_="1" * 64)
    assert folders(find_runs(root=str(share / "runs"))) == [folder]
    assert {run.location for run in find_runs(root=str(share / "runs"))} == {"share"}
    assert folder not in folders(find_runs(root=str(Path(config["CFM_data_path"]) / "runs")))
    assert {run.location for run in find_runs(root=str(Path(config["CFM_data_path"]) / "runs"))} == {"local"}


def test_share_not_available(share, runs, monkeypatch, caplog):
    """A "soft" mount that stops gives an OSError after a timeout: one check, one warning, the local runs only."""
    make_share_run(share, "RACMO_300_300", hash_="1" * 64)
    real_stat = os.stat
    attempts = []

    def stat(path, *args, **kwargs):
        if str(path).startswith(str(share)):
            attempts.append(path)
            raise OSError(errno.EIO, "Input/output error", str(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(sim.manifest.os, "stat", stat)
    for _ in range(3):
        found = find_runs()
        assert len(found) == 5
        assert {run.location for run in found} == {"local"}
    assert len(attempts) == 1
    warnings = [record.getMessage() for record in caplog.records if "not available" in record.getMessage()]
    assert len(warnings) == 1 and str(share / "runs") in warnings[0]
    # the check is made again after reset_share_check (for example in the next test)
    monkeypatch.setattr(sim.manifest.os, "stat", real_stat)
    sim.manifest.reset_share_check()
    assert len(find_runs()) == 6


def test_table_has_location(share, runs):
    make_share_run(share, "RACMO_300_300", hash_="1" * 64)
    make_share_run(share, "RACMO_261_126", physRho="Crocus", hash_="b" * 64)
    table = runs_table(find_runs())
    assert list(table.columns).index("location") == list(table.columns).index("folder") - 1
    assert table.set_index("model_point")["location"]["RACMO_300_300"] == "share"
    assert table.set_index("physRho")["location"]["Crocus"] == "both"
    assert table.set_index("run_hash")["location"]["a" * 64] == "local"


# the --index option of scripts/find_runs.py: the default path, a share that is not available, the safe write


def local_index(tmp_path) -> Path:
    return tmp_path / "index.csv"  # CFM_data_path of the runs fixture is tmp_path


def test_default_index_without_share(runs, tmp_path, capsys):
    assert find_runs_script.main(["--index"]) == 0
    assert f"Wrote {local_index(tmp_path)}." in capsys.readouterr().out
    table = pd.read_csv(local_index(tmp_path))
    assert len(table) == 5 and set(table["location"]) == {"local"}
    assert not list(tmp_path.glob("*.tmp"))


def test_default_index_on_the_share(share, runs, tmp_path):
    make_share_run(share, "RACMO_300_300", hash_="1" * 64)
    assert find_runs_script.main(["--index"]) == 0
    table = pd.read_csv(share / "index.csv")
    assert len(table) == 6 and set(table["location"]) == {"local", "share"}
    assert "RACMO_300_300" in set(table["model_point"])
    assert not local_index(tmp_path).exists()
    assert not list(share.glob("*.tmp"))


def test_explicit_index_path(share, runs, tmp_path):
    assert find_runs_script.main(["--index", str(tmp_path / "x.csv")]) == 0
    assert len(pd.read_csv(tmp_path / "x.csv")) == 5
    assert not (share / "index.csv").exists()


def no_query(**filters):
    raise AssertionError("the query must not start")


def test_index_with_a_bad_extension(share, runs, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(find_runs_script, "find_runs", no_query)
    assert find_runs_script.main(["--index", str(tmp_path / "x.txt")]) == 1
    assert ".csv or .parquet" in capsys.readouterr().err
    assert not (tmp_path / "x.txt").exists() and not (tmp_path / "x.txt.tmp").exists()


def test_index_stops_if_the_share_is_not_available(share, runs, tmp_path, monkeypatch, capsys):
    real_stat = os.stat

    def stat(path, *args, **kwargs):
        if str(path).startswith(str(share)):
            raise OSError(errno.EIO, "Input/output error", str(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(sim.manifest.os, "stat", stat)  # the os module of cfm_config.share_available too
    explicit = tmp_path / "x.csv"
    with monkeypatch.context() as patch:
        patch.setattr(find_runs_script, "find_runs", no_query)  # the check is before the query
        for argv in (["--index"], ["--index", str(explicit)]):
            assert find_runs_script.main(argv) == 1
            assert "not available" in capsys.readouterr().err
    # without --index, the query still gives the local runs
    assert find_runs_script.main([]) == 0
    assert "5 runs." in capsys.readouterr().out
    monkeypatch.setattr(sim.manifest.os, "stat", real_stat)
    assert not (share / "index.csv").exists() and not local_index(tmp_path).exists() and not explicit.exists()


def test_failed_write_keeps_the_old_index(runs, tmp_path, monkeypatch, capsys):
    path = local_index(tmp_path)
    path.write_text("old\n")

    def replace(src, dst):
        raise OSError(errno.EIO, "Input/output error", dst)

    monkeypatch.setattr(find_runs_script.os, "replace", replace)
    assert find_runs_script.main(["--index"]) == 1
    assert "Cannot write" in capsys.readouterr().err
    assert path.read_text() == "old\n"
    assert not (tmp_path / "index.csv.tmp").exists()


def test_parquet_without_pyarrow(runs, tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "pyarrow", None)  # import pyarrow raises ImportError
    monkeypatch.setattr(find_runs_script, "find_runs", no_query)
    assert find_runs_script.main(["--index", str(tmp_path / "x.parquet")]) == 1
    assert "pyarrow is not installed" in capsys.readouterr().err
    assert not list(tmp_path.glob("x.parquet*"))


def test_share_that_stops_during_the_query(runs, monkeypatch, capsys):
    def stops(**filters):
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(find_runs_script, "find_runs", stops)
    assert find_runs_script.main([]) == 1
    assert "cannot read the runs" in capsys.readouterr().err


# the cfm_config filter (FEAT-009): two runs that differ only in a cfm_config key, for example SEB

SEB_TRUE = {"SEB": True, "MELT": True, "SEB_TL_thick": 0.05}
SEB_FALSE = {"SEB": False, "MELT": True, "SEB_TL_thick": 0.05}


def migrate(folder, inferred=None):
    """Make the run a migrated run with no cfm_config, and with migrated.inferred_cfm_config if inferred is given."""
    manifest = sim.manifest.read_manifest(folder)
    manifest["cfm_config"] = None
    manifest["migrated"] = {"unknown": ["cfm_config"]}
    if inferred is not None:
        manifest["migrated"]["inferred_cfm_config"] = inferred
    write_manifest(folder, manifest)


def test_cfm_config_value_sources():
    run = sim.runs.Run("x", {"cfm_config": {"SEB": True, "MELT": True}})
    assert run.cfm_config_value("SEB") is True  # 1. cfm_config
    assert sim.runs.Run("x", {"cfm_config": {"MELT": True}}).cfm_config_value("SEB") is False  # 2. the CFM default
    assert run.cfm_config_value("SEB_TL_thick") is sim.runs.UNKNOWN  # no default
    migrated = {"cfm_config": None, "migrated": {"inferred_cfm_config": {"SEB": True}}}
    assert sim.runs.Run("x", migrated).cfm_config_value("SEB") is True  # 3. inferred
    assert sim.runs.Run("x", migrated).cfm_config_value("MELT") is sim.runs.UNKNOWN  # 4. not known
    # a migrated run gives no SEB default: its CFM config is not known
    assert sim.runs.Run("x", {"cfm_config": None, "migrated": {}}).cfm_config_value("SEB") is sim.runs.UNKNOWN
    assert sim.runs.Run("x", {}).cfm_config_value("SEB") is sim.runs.UNKNOWN


def test_cfm_config_values_are_compared_after_a_json_round_trip():
    same = sim.runs.same_cfm_config_value
    assert same(1, 1.0) and same(0.05, 0.05) and same([1, 2], (1, 2)) and same(False, False)
    assert not same(True, 1) and not same(True, "true") and not same(False, 0) and not same(1, 2)
    assert not same(sim.runs.UNKNOWN, sim.runs.UNKNOWN)


def test_cfm_config_filter(runs):
    seb_true = make_run("RACMO_261_126", hash_="1" * 64, cfm_config=SEB_TRUE)
    seb_false = make_run("RACMO_261_126", hash_="2" * 64, cfm_config=SEB_FALSE)
    default = make_run("RACMO_261_126", hash_="3" * 64, cfm_config={"MELT": True})  # SEB false by default
    filters = dict(model_point="RACMO_261_126", physRho="GSFC2020", cfm="ab871c8")
    assert folders(find_runs(cfm_config={"SEB": True}, **filters)) == [seb_true]
    assert folders(find_runs(cfm_config={"SEB": False}, **filters)) == sorted([seb_false, default])
    assert folders(find_runs(cfm_config={"SEB": False, "SEB_TL_thick": 0.05}, **filters)) == [seb_false]
    assert folders(find_runs(cfm_config={"SEB_TL_thick": 0.05}, **filters)) == sorted([seb_true, seb_false])
    # no filter: all the runs, as before; runs["gsfc"] has no cfm_config, so it agrees with no cfm_config filter
    assert folders(find_runs(**filters)) == sorted([runs["gsfc"], seb_true, seb_false, default])
    assert folders(find_runs(cfm_config={}, **filters)) == folders(find_runs(**filters))


def test_cfm_config_filter_in_the_share_root(share, runs):
    local = make_run("RACMO_261_126", hash_="1" * 64, cfm_config=SEB_TRUE)
    on_share = make_share_run(share, "RACMO_261_126", hash_="2" * 64, cfm_config=SEB_FALSE)
    make_share_run(share, "RACMO_261_126", hash_="1" * 64, cfm_config=SEB_TRUE)  # also local
    (run,) = find_runs(cfm_config={"SEB": False})
    assert (run.folder, run.location) == (on_share, "share")
    (run,) = find_runs(cfm_config={"SEB": True})
    assert (run.folder, run.location) == (local, "both")


def test_cfm_config_filter_of_migrated_runs(runs):
    inferred = make_run("RACMO_261_126", cfm="3e3bf7e-speersm", hash_="1" * 64)
    migrate(inferred, {"SEB": True})
    migrate(runs["old_cfm"])  # no inferred value
    old = dict(model_point="RACMO_261_126", cfm="3e3bf7e-speersm")
    assert folders(find_runs(cfm_config={"SEB": True}, **old)) == [inferred]
    assert find_runs(cfm_config={"SEB": False}, **old) == []
    assert find_runs(cfm_config={"MELT": True}, **old) == []  # MELT is not known
    assert len(find_runs(**old)) == 2


def test_a_run_without_meltwater_solver_is_enthalpy(runs):
    # as Run.meltwater_solver: the cfm_config filter and find_run use the default of the CFM for a run without the key
    for manifest in ({"cfm_config": {"MELT": True}}, {"cfm_config": None, "migrated": {}}):
        run = sim.runs.Run("x", manifest)
        assert run.cfm_config_value("meltwater_solver") == run.meltwater_solver == "enthalpy"
    enthalpy = make_run("RACMO_261_126", hash_="1" * 64, cfm_config=SEB_TRUE)
    ncz = make_run("RACMO_261_126", hash_="2" * 64, cfm_config={**SEB_TRUE, "meltwater_solver": "ncz"})
    filters = dict(model_point="RACMO_261_126", physRho="GSFC2020", cfm="ab871c8", cfm_config={"SEB": True})
    assert folders(find_runs(**{**filters, "cfm_config": {"SEB": True, "meltwater_solver": "enthalpy"}})) == [enthalpy]
    with pytest.raises(LookupError, match=r"cfm_config key meltwater_solver \(11111111: 'enthalpy', 22222222: 'ncz'\)"):
        find_run(**filters)
    assert find_run(**{**filters, "cfm_config": {"SEB": True, "meltwater_solver": "ncz"}}).folder == ncz


def test_find_run_stops_if_the_runs_differ_in_a_cfm_config_key(runs):
    make_run("RACMO_261_126", hash_="1" * 64, cfm_config=SEB_TRUE)
    seb_false = make_run("RACMO_261_126", hash_="2" * 64, cfm_config=SEB_FALSE, finished="2026-10-17T10:00:00")
    filters = dict(model_point="RACMO_261_126", physRho="GSFC2020", cfm="ab871c8")
    with pytest.raises(LookupError, match=r"cfm_config key SEB \(11111111: True, aaaaaaaa: unknown, 22222222: False\)"):
        find_run(**filters)
    assert find_run(cfm_config={"SEB": False}, **filters).folder == seb_false


def test_find_run_gives_the_newest_if_no_known_key_differs(runs, caplog):
    make_run("RACMO_261_126", hash_="1" * 64, cfm_config=SEB_TRUE)
    later = make_run("RACMO_261_126", hash_="2" * 64, cfm_config=dict(SEB_TRUE), finished="2026-10-17T10:00:00")
    migrated = make_run("RACMO_261_126", hash_="3" * 64, cfm_config=SEB_FALSE)
    migrate(migrated)  # SEB not known: it is not compared
    assert find_run(model_point="RACMO_261_126", physRho="GSFC2020", cfm="ab871c8").folder == later
    assert "4 runs" in caplog.text


def test_cfm_config_differences_ignores_the_runs_that_do_not_know_a_key():
    a = sim.runs.Run("a", {"run_hash": "a" * 64, "cfm_config": {"SEB": True, "MELT": True, "resultsFolder": "a"}})
    b = sim.runs.Run("b", {"run_hash": "b" * 64, "cfm_config": {"SEB": True, "MELT": False, "resultsFolder": "b"}})
    m = sim.runs.Run("m", {"run_hash": "c" * 64, "cfm_config": None, "migrated": {"inferred_cfm_config": {"SEB": False}}})
    assert sim.runs.cfm_config_differences([a, b]) == {"MELT": [True, False]}  # not resultsFolder
    unknown = sim.runs.UNKNOWN
    assert sim.runs.cfm_config_differences([a, b, m]) == {"MELT": [True, False, unknown], "SEB": [True, True, False]}
    assert sim.runs.cfm_config_differences([a, m]) == {"SEB": [True, False]}  # MELT is known for a only
    assert sim.runs.cfm_config_differences([a, sim.runs.Run("n", {"cfm_config": None})]) == {}
    assert sim.runs.cfm_config_differences([a]) == {}
    assert "not known" in sim.runs.describe_cfm_config_differences([a, a])


def test_find_runs_script_cfm_config(runs, capsys):
    make_run("RACMO_261_126", hash_="1" * 64, cfm_config=SEB_TRUE)
    make_run("RACMO_261_126", hash_="2" * 64, cfm_config=SEB_FALSE)
    assert find_runs_script.main(["--cfm-config", "SEB=false"]) == 0
    out = capsys.readouterr().out
    assert "22222222" in out and "11111111" not in out and "1 runs." in out
    assert find_runs_script.main(["--cfm-config", "SEB=true", "--cfm-config", "SEB_TL_thick=0.05"]) == 0
    assert "11111111" in capsys.readouterr().out
    assert find_runs_script.cfm_config_item("meltwater_solver=ncz") == ("meltwater_solver", "ncz")
    assert find_runs_script.cfm_config_item("SEB_TL_thick=0.05") == ("SEB_TL_thick", 0.05)
    with pytest.raises(SystemExit):
        find_runs_script.main(["--cfm-config", "SEB"])
