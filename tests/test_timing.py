"""Tests for the timing block of the manifest and --profile (FEAT-013). Run from the repo root.

The CFM is a fake: makeSpinFiles and FirnDensityNoSpin are stubs, as in tests/test_run_status.py, so the real _run_cfm
runs and the stage logic is tested. Only the machine data is read from the real machine.
"""

import hashlib
import importlib.util
import logging
import os
import pstats
import sys
import time
from pathlib import Path

import pytest

import cfm_config
import sim.run
import sim.runs
import sim.timing
from cfm_config import ConfigError
from sim.manifest import COMPLETE, FAILED, read_manifest, run_hash
from sim.run import CFMRun
from sim.timing import clock

from conftest import SITE, VERSION, folder, new_run, overrides_file  # noqa: F401

pytestmark = pytest.mark.usefixtures("identities", "drift_not_warned")

STAGES_WITH_LOOP = [
    "prepare",
    "read_forcing",
    "setup",
    "make_spin_files",
    "cfm_init",
    "loop_before_switch",
    "loop_after_switch",
    "write_hdf5",
    "record_settings",
]


def pause() -> None:
    time.sleep(0.02)


def fake_cfm(monkeypatch, loop_stages="with_switch"):
    """Stub the CFM for the real _run_cfm. The fake firn sets step_timing_stages like the new CFM, with a switch
    ("with_switch"), without a switch ("no_switch"), or not at all ("old_cfm")."""
    seen = {}

    class Firn:
        c = {"liquid": "bucket"}

        def __init__(self, config_path, **kwargs):
            seen["config"] = dict(__import__("json").load(open(config_path)))
            pause()

        def time_evolve(self):
            if loop_stages == "old_cfm":
                pause()
                return
            if loop_stages == "crash":
                self.step_timing_stages = {"loop_start": clock(), "switch": None, "loop_end": None, "write_end": None}
                pause()
                self.step_timing_stages["switch"] = clock()
                pause()
                raise RuntimeError("crash in the loop")
            if loop_stages.startswith("bad:"):
                value = {"nan": float("nan"), "text": "x", "bool": True}[loop_stages[4:]]
                self.step_timing_stages = {
                    "loop_start": {"wall": value, "cpu": 0.0},
                    "switch": None,
                    "loop_end": clock(),
                    "write_end": clock(),
                }
                return
            start = clock()
            pause()
            switch = clock() if loop_stages == "with_switch" else None
            pause()
            end = clock()
            pause()
            self.step_timing_stages = {"loop_start": start, "switch": switch, "loop_end": end, "write_end": clock()}

    def make_spin_files(*args, **kwargs):
        pause()
        return {"SUBLIM": 1.0}, 12.0, 5.0, 10.0, 100.0, None

    monkeypatch.setattr(sim.run, "import_cfm", lambda repo: (make_spin_files, Firn))
    return seen


def run_and_read(**kwargs) -> dict:
    new_run(**kwargs).run()
    return read_manifest(folder())


# 1. the run hash


def test_step_timing_is_set_but_not_in_the_hash(data_path, monkeypatch):
    seen = fake_cfm(monkeypatch)
    before = new_run()
    before.prepare()
    manifest = run_and_read()
    assert seen["config"]["step_timing"] is True
    assert manifest["cfm_config"]["step_timing"] is True
    assert "step_timing" not in manifest["hash_inputs"]["cfm_config"]
    assert manifest["run_hash"] == before._run_hash == run_hash(manifest["hash_inputs"])
    assert folder() == before.prepare()


# 3. the stages


def test_the_stages_add_up_to_the_run(data_path, monkeypatch):
    fake_cfm(monkeypatch)
    start = time.perf_counter()
    manifest = run_and_read()
    process_wall = time.perf_counter() - start
    stages = manifest["timing"]["stages"]
    assert list(stages) == STAGES_WITH_LOOP
    for stage in stages.values():
        assert stage["wall_s"] >= 0 and stage["cpu_s"] >= 0
    total = sum(stage["wall_s"] for stage in stages.values())
    assert total <= process_wall + 0.01
    assert total == pytest.approx(process_wall, abs=1.0)
    for name in ("cfm_init", "loop_before_switch", "loop_after_switch"):
        assert stages[name]["wall_s"] >= 0.015  # each has a pause of 0.02 s
    assert manifest["status"] == COMPLETE and manifest["runtime_s"] >= 0.1


def test_the_loop_stages_are_the_loop_of_the_cfm(data_path, monkeypatch):
    """loop_before_switch plus loop_after_switch is loop_start to loop_end, write_hdf5 is loop_end to write_end."""
    fake_cfm(monkeypatch)
    recorded = {}
    original = sim.run.CFMRun._read_step_timing_stages

    def spy(firn):
        recorded.update(firn.step_timing_stages)
        return original(firn)

    monkeypatch.setattr(CFMRun, "_read_step_timing_stages", staticmethod(spy))
    stages = run_and_read()["timing"]["stages"]
    loop = recorded["loop_end"]["wall"] - recorded["loop_start"]["wall"]
    assert stages["loop_before_switch"]["wall_s"] + stages["loop_after_switch"]["wall_s"] == pytest.approx(loop, abs=0.002)
    assert stages["write_hdf5"]["wall_s"] == pytest.approx(recorded["write_end"]["wall"] - recorded["loop_end"]["wall"], abs=0.002)


def test_a_loop_without_a_switch_has_no_after_switch_stage(data_path, monkeypatch):
    fake_cfm(monkeypatch, "no_switch")
    stages = run_and_read()["timing"]["stages"]
    assert "loop_before_switch" in stages and "loop_after_switch" not in stages and "write_hdf5" in stages


# 4. an old CFM


def test_an_old_cfm_has_one_time_evolve_stage(data_path, monkeypatch):
    fake_cfm(monkeypatch, "old_cfm")
    manifest = run_and_read()
    stages = manifest["timing"]["stages"]
    assert list(stages) == ["prepare", "read_forcing", "setup", "make_spin_files", "cfm_init", "time_evolve", "record_settings"]
    assert stages["time_evolve"]["wall_s"] >= 0.015
    assert manifest["status"] == COMPLETE


def test_a_bad_step_timing_stages_is_a_warning_not_a_failure(data_path, monkeypatch, caplog):
    class Firn:
        step_timing_stages = {"loop_start": 1}

    with caplog.at_level(logging.WARNING):
        assert CFMRun._read_step_timing_stages(Firn()) is None
    assert "not as expected" in caplog.text


def test_a_failed_run_has_the_stages_that_were_reached(data_path, monkeypatch):
    monkeypatch.setattr(sim.run, "import_cfm", lambda repo: (lambda *a, **k: 1 / 0, None))
    with pytest.raises(ZeroDivisionError):
        new_run().run()
    manifest = read_manifest(folder())
    assert manifest["status"] == FAILED
    assert list(manifest["timing"]["stages"]) == ["prepare", "read_forcing", "setup"]


# 5. the machine data


def test_the_machine_data_is_filled_or_null(data_path, monkeypatch):
    fake_cfm(monkeypatch)
    block = run_and_read()["timing"]
    assert block["profiled"] is False
    assert block["cpu_s_total"] is not None and block["cpu_s_total"] > 0
    assert block["max_rss_mb"] is not None and 5 < block["max_rss_mb"] < 100_000
    for key in ("loadavg_start", "loadavg_end"):
        assert block[key] is None or len(block[key]) == 3
    assert block["cpu_count"] == os.cpu_count()
    assert block["cpus_allowed"] is None or block["cpus_allowed"] >= 1
    assert block["cpu_model"] is None or isinstance(block["cpu_model"], str)
    assert set(block["slurm"]) <= set(sim.timing.SLURM_VARIABLES)
    assert set(block["threads"]) <= set(sim.timing.THREAD_VARIABLES)


def test_slurm_and_thread_variables_are_recorded_when_set(monkeypatch):
    for name in sim.timing.SLURM_VARIABLES + sim.timing.THREAD_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    data = sim.timing.machine_data()
    assert data["slurm"] == {"SLURM_JOB_ID": "123"}
    assert data["threads"] == {"OMP_NUM_THREADS": "1"}


@pytest.mark.parametrize("platform, divisor", [("darwin", 1024 * 1024), ("linux", 1024)])
def test_max_rss_unit(monkeypatch, platform, divisor):
    """ru_maxrss is in bytes on macOS and in kB on Linux."""

    class Usage:
        ru_maxrss = 512 * divisor

    monkeypatch.setattr(sim.timing.sys, "platform", platform)
    monkeypatch.setattr(sim.timing.resource, "getrusage", lambda who: Usage)
    assert sim.timing.max_rss_mb() == 512.0


def test_a_value_that_cannot_be_read_is_null_with_a_warning(monkeypatch, caplog):
    def fail(who):
        raise OSError("no rusage")

    monkeypatch.setattr(sim.timing.resource, "getrusage", fail)
    monkeypatch.setattr(sim.timing.os, "getloadavg", lambda: 1 / 0)
    with caplog.at_level(logging.WARNING):
        assert sim.timing.max_rss_mb() is None
        assert sim.timing.cpu_time_total() is None
        assert sim.timing.loadavg() is None
    assert caplog.text.count("is not in the manifest") == 3


def test_a_failure_of_the_timing_block_does_not_fail_the_run(data_path, monkeypatch, caplog):
    fake_cfm(monkeypatch)
    monkeypatch.setattr(CFMRun, "_stages", lambda self: 1 / 0)
    with caplog.at_level(logging.WARNING):
        manifest = run_and_read()
    assert manifest["status"] == COMPLETE and manifest["timing"] is None
    assert "The timing is not in the manifest" in caplog.text


def test_cpu_model_falls_back_to_platform_processor(monkeypatch):
    monkeypatch.setattr("builtins.open", lambda *a, **k: (_ for _ in ()).throw(OSError("no /proc")))
    monkeypatch.setattr(sim.timing.sys, "platform", "linux")
    monkeypatch.setattr(sim.timing.platform, "processor", lambda: "arm")
    assert sim.timing.cpu_model() == "arm"
    monkeypatch.setattr(sim.timing.platform, "processor", lambda: "")
    assert sim.timing.cpu_model() is None


def test_cpu_model_on_macos_uses_sysctl_then_platform_processor(monkeypatch):
    monkeypatch.setattr("builtins.open", lambda *a, **k: (_ for _ in ()).throw(OSError("no /proc")))
    monkeypatch.setattr(sim.timing.sys, "platform", "darwin")
    monkeypatch.setattr(sim.timing.platform, "processor", lambda: "arm")

    class Result:
        stdout = "Apple M2 Pro\n"

    monkeypatch.setattr(sim.timing.subprocess, "run", lambda *a, **k: Result)
    assert sim.timing.cpu_model() == "Apple M2 Pro"

    def fail(*a, **k):
        raise OSError("no sysctl")

    monkeypatch.setattr(sim.timing.subprocess, "run", fail)
    assert sim.timing.cpu_model() == "arm"


def test_cpus_allowed_warns_where_it_is_not_available(monkeypatch, caplog):
    monkeypatch.delattr(sim.timing.os, "sched_getaffinity", raising=False)
    with caplog.at_level(logging.WARNING):
        assert sim.timing.cpus_allowed() is None
    assert "cpus_allowed" in caplog.text


# 6. the profile


def test_profile_writes_a_profile_that_pstats_reads(data_path, monkeypatch):
    def run_cfm(self):
        sorted(range(1000), key=lambda value: -value)
        open(os.path.join(self._cfm_config["resultsFolder"], "CFMresults.hdf5"), "w").close()

    monkeypatch.setattr(CFMRun, "_run_cfm", run_cfm)
    manifest = run_and_read(profile=True)
    path = os.path.join(folder(), sim.run.PROFILE_NAME)
    assert os.path.getsize(path) > 0
    assert pstats.Stats(path).total_calls > 0
    assert manifest["timing"]["profiled"] is True
    assert os.path.basename(folder()) == manifest["run_hash"][:8]  # the profile is not in the hash


def test_without_profile_there_is_no_profile_file(data_path, monkeypatch):
    fake_cfm(monkeypatch)
    manifest = run_and_read()
    assert not os.path.exists(os.path.join(folder(), sim.run.PROFILE_NAME))
    assert manifest["timing"]["profiled"] is False


def test_a_failed_profiled_run_keeps_the_profile(data_path, monkeypatch):
    def run_cfm(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(CFMRun, "_run_cfm", run_cfm)
    with pytest.raises(RuntimeError):
        new_run(profile=True).run()
    assert pstats.Stats(os.path.join(folder(), sim.run.PROFILE_NAME)).total_calls > 0
    assert read_manifest(folder())["status"] == FAILED


def test_profile_goes_from_run_cfm_to_the_worker():
    spec = importlib.util.spec_from_file_location(
        "run_cfm", Path(__file__).resolve().parents[1] / "scripts" / "run_cfm.py"
    )
    run_cfm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_cfm)
    from cfm_config import Site
    from force.years import RunYears

    site = Site("CI-0", -66.403, -63.376)
    years = RunYears(1979, 2023, 2024, "config")
    args = (VERSION, site, "GSFC2020", "RACMO", "bucket", years)
    assert "--profile" not in run_cfm.worker_command(*args)
    assert "--profile" in run_cfm.worker_command(*args, profile=True)


# 7. the overrides


def test_step_timing_is_refused_in_an_overrides_file(data_path):
    assert "step_timing" in sim.run.REFUSED_CFM_OVERRIDE_KEYS
    with pytest.raises(ConfigError, match="sets step_timing"):
        new_run(cfm_overrides=overrides_file(data_path, "step_timing: false\n"))


# the review of FEAT-013

REFERENCE_HASH = "51a14e6aacff1ba30b41caf49d7f848d62f507841785bd3099193aefbd240571"


def test_the_hash_of_fixed_inputs_is_the_hash_before_feat_013(data_path):
    """Computed with the code of bc6e941 (before FEAT-013) for these inputs: the test config, SITE, VERSION,
    CFM_IDENTITY and the forcing CSV of data_path. Change this value only for a change that is meant to change hashes."""
    run = new_run()
    run.prepare()
    assert run._run_hash == REFERENCE_HASH


def test_a_plain_rerun_removes_the_profile_of_an_earlier_run(data_path, monkeypatch):
    def run_cfm(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(CFMRun, "_run_cfm", run_cfm)
    with pytest.raises(RuntimeError):
        new_run(profile=True).run()
    path = os.path.join(folder(), sim.run.PROFILE_NAME)
    assert os.path.exists(path)
    with pytest.raises(RuntimeError):
        new_run().run()
    assert not os.path.exists(path)
    assert read_manifest(folder())["timing"]["profiled"] is False


def test_the_stages_cover_the_whole_run(data_path, monkeypatch):
    """The import of the CFM and the setup of the run folder are in a stage: no gap of more than a few ms."""
    fake_cfm(monkeypatch)
    wall = {}
    original = CFMRun.run

    def timed_run(self):
        start = time.perf_counter()
        original(self)
        wall["run"] = time.perf_counter() - start

    monkeypatch.setattr(CFMRun, "run", timed_run)
    manifest = run_and_read()
    total = sum(stage["wall_s"] for stage in manifest["timing"]["stages"].values())
    assert wall["run"] - 0.25 <= total <= wall["run"]


def test_the_import_of_the_cfm_is_in_the_make_spin_files_stage(data_path, monkeypatch):
    fake_cfm(monkeypatch)
    original = sim.run.import_cfm
    monkeypatch.setattr(sim.run, "import_cfm", lambda repo: (time.sleep(0.2), original(repo))[1])
    assert run_and_read()["timing"]["stages"]["make_spin_files"]["wall_s"] >= 0.2


def test_a_crash_in_the_loop_keeps_the_loop_times(data_path, monkeypatch):
    fake_cfm(monkeypatch, "crash")
    with pytest.raises(RuntimeError, match="crash in the loop"):
        new_run().run()
    manifest = read_manifest(folder())
    stages = manifest["timing"]["stages"]
    assert manifest["status"] == FAILED
    assert stages["loop_before_switch"]["wall_s"] >= 0.015
    assert "loop_after_switch" not in stages and "write_hdf5" not in stages
    assert "time_evolve" not in stages and "record_settings" not in stages


@pytest.mark.parametrize("bad", ["nan", "text", "bool"])
def test_a_bad_loop_time_gives_the_time_evolve_stage_and_keeps_the_machine_data(data_path, monkeypatch, bad, caplog):
    fake_cfm(monkeypatch, "bad:" + bad)
    with caplog.at_level(logging.WARNING):
        manifest = run_and_read()
    block = manifest["timing"]
    assert manifest["status"] == COMPLETE
    assert "time_evolve" in block["stages"] and "loop_before_switch" not in block["stages"]
    assert block["cpu_s_total"] > 0 and block["cpu_count"] == os.cpu_count()
    assert "not as expected" in caplog.text


def test_cfm_config_differences_leaves_out_step_timing():
    class FakeRun:
        def __init__(self, config):
            self.config = config

        def cfm_config_keys(self):
            return set(self.config)

        def cfm_config_value(self, key):
            return self.config.get(key, sim.runs.UNKNOWN)

    runs = [FakeRun({"IrrVal": 0.02, "step_timing": True}), FakeRun({"IrrVal": 0.02, "step_timing": False})]
    assert sim.runs.cfm_config_differences(runs) == {}
    assert sim.runs.cfm_config_differences([runs[0], FakeRun({"IrrVal": 0.05})]) == {"IrrVal": [0.02, 0.05]}


def test_the_worker_gives_profile_to_cfm_run(monkeypatch):
    import runpy

    seen = []

    class FakeRun:
        def __init__(self, *args, **kwargs):
            seen.append(kwargs)

        def run(self):
            pass

    monkeypatch.setattr(sim.run, "CFMRun", FakeRun)
    monkeypatch.setattr(cfm_config, "cfm_version", lambda label: VERSION)
    base = ["sim.worker", "--lat", "-66.4", "--lon", "-63.4", "--physrho", "GSFC2020", "--rcm", "RACMO"]
    base += ["--liquid", "bucket", "--cfm", "test", "--start-year", "1979", "--end-year", "2023"]
    for extra, expected in (([], False), (["--profile"], True)):
        monkeypatch.setattr(sys, "argv", base + extra)
        runpy.run_module("sim.worker", run_name="__main__")
        assert seen[-1]["profile"] is expected
