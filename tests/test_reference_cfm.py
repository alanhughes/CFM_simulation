"""The CFM reference test (FEAT-001 step 7): a real CFM run that must agree bit for bit with a run in speersm/.

The run is CI-0, GSFC2020, bucket, RACMO (ANT-12), 1979-2024, with the CFM version 3e3bf7e-speersm (3e3bf7e with the
SEB and melt patch of speersm) and the old CI-0 forcing CSV from speersm/. That CSV holds the data of the cell (261, 125),
which the code selected before the BUG-002 fix, so the test runs at the centre of that cell, not at the CI-0 site.

The run takes about 7 minutes and reads the luna share, so the test runs only with CFM_REFERENCE_TEST=1:

    CFM_REFERENCE_TEST=1 PYTHONPATH=src python -m pytest -q tests/test_reference_cfm.py

It needs config.yaml with the cfm_versions entry 3e3bf7e-speersm, and committed code in src/ (sim.run refuses
uncommitted run code). It writes only to tmp_path; it reads speersm/ and does not change it.
"""

import importlib.util
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from cfm_config import config

REPO = Path(__file__).resolve().parents[1]
LABEL = "3e3bf7e-speersm"
SPEERSM = Path(config["cpom_mount"]) / "speersm" / "CFM_data"
REFERENCE_RUN = SPEERSM / "cfm_output" / "CFMoutput_-66.403_-63.376_1979_2024_GSFC2020_bucket_RACMO"
REFERENCE_CSV = SPEERSM / "cfm_input" / "RACMO_-66.403_-63.376_1979_2024.csv"
OLD_CELL = (261, 125)

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("CFM_REFERENCE_TEST") != "1", reason="a 7 minute CFM run; set CFM_REFERENCE_TEST=1"
    ),
    pytest.mark.real_config,  # the real data, not the test config of tests/conftest.py
]


def load_compare():
    spec = importlib.util.spec_from_file_location("compare_cfm_runs", REPO / "scripts" / "compare_cfm_runs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_run_is_bit_for_bit(tmp_path):
    from cfm_config import cfm_version
    from force.model_point import _grid, rcm_data_path

    for path in (REFERENCE_RUN / "CFMresults.hdf5", REFERENCE_CSV):
        if not path.exists():
            pytest.skip(f"{path} is not available")
    version = cfm_version(LABEL)

    # a config.yaml for a CFM_data_path in tmp_path, with no share and the years of the speersm runs
    text = (REPO / "config.yaml").read_text()
    text = re.sub(r"(?m)^CFM_data_path:.*$", f"CFM_data_path: {tmp_path / 'data'}", text)
    text = re.sub(r"(?m)^CFM_share_data_path:.*$", "", text)
    text = re.sub(r"(?m)^forcing_data_path:.*$", "", text)
    text = re.sub(r"(?m)^start_year:.*$", "start_year: 1979", text)
    text = re.sub(r"(?m)^end_year:.*$", "end_year: 2024", text)
    (tmp_path / "config.yaml").write_text(text)

    # the old CSV in the forcing cache, as the forcing of the old cell
    release, _, lat2d, lon2d = _grid("RACMO", rcm_data_path("RACMO"))
    i, j = OLD_CELL
    forcing = tmp_path / "data" / "forcing" / "RACMO" / release / f"{i}_{j}_1979_2024.csv"
    forcing.parent.mkdir(parents=True)
    shutil.copyfile(REFERENCE_CSV, forcing)

    command = [
        version["python"],
        "-m",
        "sim.worker",
        "--lat",
        repr(float(lat2d[i, j])),
        "--lon",
        repr(float(lon2d[i, j])),
    ]
    command += ["--site", "CI-0", "--physrho", "GSFC2020", "--rcm", "RACMO", "--liquid", "bucket", "--cfm", LABEL]
    command += ["--start-year", "1979", "--end-year", "2024"]  # the years of the speersm runs
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]

    (manifest,) = (tmp_path / "data" / "runs" / f"RACMO_{i}_{j}").rglob("manifest.json")
    row = load_compare().summary(str(manifest.parent), str(REFERENCE_RUN))
    assert row["bit_for_bit"], row
