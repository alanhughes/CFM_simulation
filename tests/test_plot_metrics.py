"""Tests of plot.metrics and plot.output on a small synthetic run (FEAT-007). No real run is read."""

import h5py
import numpy as np
import pytest

import plot.output
from plot import metrics
from plot.output import RunOutput, exactly_one

DEPTH = np.arange(0.0, 10.0, 0.5)  # 20 depths
TIME = 1980.0 + np.arange(1, 25) / 4  # 24 steps, 1980.25 - 1986.0


def density_at(t: float) -> np.ndarray:
    """A profile that gets denser with depth and, by 1 kg/m3 per year, with time."""
    return 400.0 + 50.0 * DEPTH + (t - 1980.0)


@pytest.fixture
def run_folder(tmp_path):
    """A run folder with the layout of the gridded CFM output: row 0 is the initial state, column 0 is the time."""
    n, m = len(TIME), len(DEPTH)
    density = np.zeros((n + 1, m + 1))
    density[0] = -1  # the initial state: must never be used
    density[1:, 0] = TIME
    density[1:, 1:] = [density_at(t) for t in TIME]
    temperature = np.full((n + 1, m + 1), 260.0)
    temperature[1:, 0] = TIME
    dip = np.zeros((n + 1, 8))
    dip[1:, 0] = TIME
    dip[1:, 1] = 20.0 - 0.1 * np.arange(n)  # FAC
    dip[1:, 2] = 0.05  # surface height change of each step
    per_step = np.zeros((n + 1, 2))
    per_step[1:, 0] = TIME
    per_step[1:, 1] = 0.001  # m w.e. per step
    with h5py.File(tmp_path / "CFMresults.hdf5", "w") as results:
        results["depth"] = np.concatenate([[1700.0], DEPTH])
        results["density"] = density
        results["temperature"] = temperature
        results["DIP"] = dip
        for name in ("meltvol", "refreeze", "runoff"):
            results[name] = per_step if name != "runoff" else np.zeros_like(per_step)
    spin = np.arange(1979.0, 1980.0, 0.25)
    forcing_time = np.concatenate([spin, 1980.0 + np.arange(24) / 4])
    bdot = np.where(forcing_time >= 1980.0, 0.6, 0.4)
    with h5py.File(tmp_path / "CFMforcing.hdf5", "w") as forcing:
        forcing["main/time"] = forcing_time
        forcing["main/BDOT"] = bdot
        forcing["main/T2m"] = np.full(forcing_time.shape, 255.0)
        forcing["main/RAIN"] = np.zeros(forcing_time.shape)
    return str(tmp_path)


def test_run_output_removes_the_initial_row_and_the_time_column(run_folder):
    run = RunOutput.from_folder(run_folder)
    np.testing.assert_array_equal(run.time, TIME)
    np.testing.assert_array_equal(run.depth, DEPTH)
    assert run.density.shape == (len(TIME), len(DEPTH))
    assert run.density.min() > 0
    assert run.fac[0] == 20.0


def test_depth_of_density(run_folder):
    run = RunOutput.from_folder(run_folder)
    # final profile 406 + 50 z: reaches 830 kg/m3 at z = 8.48 m, thus at the grid depth 8.5 m
    assert run.depth_of_density(830) == 8.5
    # never reached: the first depth, as compare_cfm_runs.py
    assert metrics.depth_of_density(DEPTH, density_at(1986.0), 2000) == 0.0


def test_surface_height_and_trend(run_folder):
    run = RunOutput.from_folder(run_folder)
    assert run.surface_height[-1] == pytest.approx(0.05 * len(TIME))
    assert run.dhdt() == pytest.approx(0.05 * 4)  # 4 steps per year


def test_window_mask_includes_both_ends():
    mask = metrics.window_mask(TIME, [(1981.0, 1981.5), (1985.75, 1990.0)])
    np.testing.assert_array_equal(TIME[mask], [1981.0, 1981.25, 1981.5, 1985.75, 1986.0])


def test_compare_with_profile(run_folder):
    run = RunOutput.from_folder(run_folder)
    depth = np.linspace(1.0, 7.5, 66)  # within the model depths, so no value is extrapolated
    observed = 400.0 + 50.0 * depth  # the model at 1980.0
    profile = metrics.Profile("X", depth, observed)
    comparison = run.compare_with_profile(profile, [(1982.0, 1982.5)])
    assert comparison.steps == 3  # 1982.0, 1982.25, 1982.5
    assert comparison.bias == pytest.approx(2.25)  # the mean of 2.0, 2.25 and 2.5
    assert comparison.rmse == pytest.approx(2.25)
    np.testing.assert_allclose(comparison.lower - observed, 2.0)
    np.testing.assert_allclose(comparison.upper - observed, 2.5)


def test_compare_with_profile_with_no_step_in_the_windows(run_folder):
    run = RunOutput.from_folder(run_folder)
    with pytest.raises(ValueError, match="No model time step"):
        run.compare_with_profile(metrics.Profile("X", DEPTH, DEPTH), [(2000.0, 2001.0)])


def test_running_mean_and_fraction_above():
    depth = np.arange(0.0, 10.0, 0.001)
    density = np.where(np.arange(depth.size) % 2 == 0, 800.0, 900.0)
    profile = metrics.Profile("X", depth, density)
    assert profile.running_mean(1.0)[5000] == pytest.approx(850.0, abs=0.1)
    assert profile.fraction_above(830, 0.0, 10.0) == pytest.approx(0.5)


def test_forcing_summary_and_water_budget(run_folder):
    run = RunOutput.from_folder(run_folder)
    assert run.forcing.spin_start == 1979.0
    assert run.forcing.bdot_since_1980 == pytest.approx(0.6)
    assert run.forcing.iceout == pytest.approx((4 * 0.4 + 24 * 0.6) / 28)
    assert run.forcing.iceout_excess > 0
    budget = run.water_budget()
    assert budget["accumulation"] == pytest.approx(0.6 * 917)
    # 24 steps of 1 kg/m2 over 5.75 years
    assert budget["melt"] == pytest.approx(24 / 5.75)
    assert budget["runoff"] == 0
    assert budget["refreeze_fraction"] == pytest.approx(budget["refreeze"] / budget["accumulation"])


def test_largest_difference_by_time():
    a = np.array([[1.0, 5.0, 2.0], [0.0, 0.0, 0.0]])
    b = np.array([[1.0, 2.0], [0.0, -3.0]])
    np.testing.assert_array_equal(metrics.largest_difference_by_time(a, b), [3.0, 3.0])


def test_exactly_one(monkeypatch):
    import types

    import sim.runs

    found = []
    monkeypatch.setattr(sim.runs, "find_runs", lambda **filters: found)
    with pytest.raises(LookupError, match="found 0"):
        exactly_one(model_point="RACMO_1_1")
    found.extend([types.SimpleNamespace(folder="a"), types.SimpleNamespace(folder="b")])
    with pytest.raises(LookupError, match="found 2"):
        exactly_one(model_point="RACMO_1_1")
    found.pop()
    assert exactly_one(model_point="RACMO_1_1").folder == "a"


def test_missing_results_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        plot.output.RunOutput.from_folder(str(tmp_path))
