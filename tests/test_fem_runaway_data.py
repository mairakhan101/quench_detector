"""
test_fem_runaway_data.py -- FEA sweep loader for time-to-runaway labels.

Synthetic-HDF5 tests run everywhere; the real-sweep checks skip when the
local FreeFEM sweep isn't present on this host.
"""

import h5py
import numpy as np
import pytest

from hts_tape_env.fem_data import RTD_CHANNELS
from hts_tape_env.fem_runaway_data import (
    DEFAULT_PATH,
    FEATURES,
    load_runaway_sweep,
    runaway_onset_time,
)

DT = 0.01


def _write_run(f, label, ratio, status, t, tmax, i0):
    g = f.create_group(label)
    g.attrs["ratio"] = ratio
    g.attrs["status"] = status
    d = g.create_group("diagnostics")
    d["t"] = t
    d["Tmax"] = tmax
    d["I0"] = i0
    d["V_CL_plus"] = np.zeros_like(t)
    d["V_CL_minus"] = 1e-3 * (tmax - 77.0)
    for c in RTD_CHANNELS:
        d[c] = tmax - 1.0 if c == "RTD1_4" else np.full_like(t, 70.0)


@pytest.fixture
def sweep_path(tmp_path):
    path = tmp_path / "sweep.h5"
    with h5py.File(path, "w") as f:
        # Written out of ratio order to check the loader sorts by ratio.
        _write_run(f, "r1p3", 1.3, "runaway",
                   t=np.array([10.0, 10.5, 11.0, 11.02, 11.04]),
                   tmax=np.array([77.0, 78.0, 85.0, 95.0, 200.0]),
                   i0=np.array([311.0, 321.0, 331.0, 331.4, 331.8]))
        _write_run(f, "r0p9", 0.9, "reached_end_no_deviation",
                   t=np.array([10.0, 10.3, 10.6]),
                   tmax=np.array([77.0, 77.0, 77.0]),
                   i0=np.array([280.0, 280.0, 280.0]))
    return path


@pytest.fixture
def df(sweep_path):
    return load_runaway_sweep(sweep_path, dt_s=DT)


def _episode(df, label):
    return df[df["label"] == label].sort_values("step")


def test_onset_time_interpolates_between_solver_steps():
    t = np.array([0.0, 1.0, 2.0])
    assert runaway_onset_time(t, np.array([80.0, 85.0, 95.0])) == pytest.approx(1.5)
    assert runaway_onset_time(t, np.array([95.0, 96.0, 97.0])) == 0.0
    assert np.isnan(runaway_onset_time(t, np.array([77.0, 80.0, 89.9])))


def test_episodes_sorted_by_ratio(df):
    assert list(df.groupby("episode_id")["label"].first()) == ["r0p9", "r1p3"]


def test_grid_is_uniform_and_starts_at_zero(df):
    for _, g in df.groupby("episode_id"):
        t = g.sort_values("step")["t_s"].to_numpy()
        assert t[0] == 0.0
        assert np.allclose(np.diff(t), DT)


def test_runaway_episode_truncated_at_onset(df):
    g = _episode(df, "r1p3")
    t_run = g["t_runaway_s"].iloc[0]
    assert g["runaway"].iloc[0] == 1
    assert t_run == pytest.approx(1.01)
    assert g["t_end_s"].iloc[0] == t_run
    assert 0.0 <= t_run - g["t_s"].iloc[-1] < DT
    assert np.allclose(g["time_to_runaway_s"], t_run - g["t_s"])
    assert (g["Tmax_K"] < 90.0 + 1e-9).all()


def test_censored_episode_has_no_time_to_runaway(df):
    g = _episode(df, "r0p9")
    assert g["runaway"].iloc[0] == 0
    assert g["time_to_runaway_s"].isna().all()
    assert np.isnan(g["t_runaway_s"].iloc[0])
    assert g["t_end_s"].iloc[0] == pytest.approx(0.6)
    assert g["t_s"].iloc[-1] == pytest.approx(0.6)


def test_inputs_are_interpolated_instrument_channels(df):
    g = _episode(df, "r1p3")
    t = g["t_s"].to_numpy()
    on_ramp = t <= 1.0
    assert np.allclose(g["I_A"].to_numpy()[on_ramp], 311.0 + 20.0 * t[on_ramp])
    # Surface temperature comes from the hottest RTD, not the mesh Tmax.
    assert np.allclose(g["T_surface_K"], g["Tmax_K"] - 1.0)
    assert set(FEATURES) <= set(df.columns)


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason=f"FEA sweep not present at {DEFAULT_PATH}")
def test_real_sweep_labels():
    real = load_runaway_sweep()
    ep = real.groupby("episode_id").first()
    assert len(ep) == 801
    assert ep["runaway"].sum() == 236
    assert ep.loc[ep["runaway"] == 1, "ratio"].min() == pytest.approx(1.265)
    # The solver's status misses ratios 1.265-1.268 (past Tc when the hold ended).
    falsely_stable = ep[(ep["status"] == "reached_end_no_deviation") & (ep["runaway"] == 1)]
    assert sorted(falsely_stable["ratio"].round(3)) == [1.265, 1.266, 1.267, 1.268]
    assert real[FEATURES].notna().all().all()
