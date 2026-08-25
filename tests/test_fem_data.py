"""
test_fem_data.py -- FEM sweep loader (fem_data.py).

Skipped whole-file if the //eagle mount isn't reachable from this host --
this is real network-mounted data, not something to fixture/commit.
"""

import numpy as np
import pytest

from hts_tape_env.fem_data import DEFAULT_PATH, QUENCH_T_K, load_fem_positions, load_fem_sweep

pytestmark = pytest.mark.skipif(
    not __import__("os").path.exists(DEFAULT_PATH),
    reason=f"FEM sweep not reachable at {DEFAULT_PATH} (network mount)",
)


@pytest.fixture(scope="module")
def df():
    return load_fem_sweep(DEFAULT_PATH)


@pytest.fixture(scope="module")
def pos():
    return load_fem_positions(DEFAULT_PATH)


def test_six_episodes_in_ascending_ratio_order(df):
    ratios = df.groupby("episode_id")["ratio"].first().to_numpy()
    assert len(ratios) == 6
    assert np.all(np.diff(ratios) > 0)


def test_runaway_flag_matches_status(df):
    for _, g in df.groupby("episode_id"):
        assert g["runaway"].iloc[0] == int(g["status"].iloc[0] == "runaway")
        assert g["runaway"].nunique() == 1  # constant within an episode


def test_quench_step_consistent_with_Tmax_threshold(df):
    for _, g in df.groupby("episode_id"):
        qs = g["quench_step"].iloc[0]
        tmax = g["T_surface_K"].to_numpy()
        if np.isfinite(qs):
            assert tmax[int(qs)] >= QUENCH_T_K
            assert np.all(tmax[: int(qs)] < QUENCH_T_K)
        else:
            assert np.all(tmax < QUENCH_T_K)


def test_undefined_columns_are_all_nan(df):
    for col in ("lam", "Istar_A", "Tstar_K", "margin_norm", "time_margin_ms"):
        assert df[col].isna().all(), f"{col} should be all-NaN (see fem_data.py docstring)"


def test_true_feature_columns_present_for_surrogate_reuse(df):
    """seq_data.TRUE_FEATURES columns must exist so a REBCO-trained
    surrogate can run against this table with zero adapter code."""
    from hts_tape_env.seq_data import TRUE_FEATURES

    for col in TRUE_FEATURES:
        assert col in df.columns


def test_positions_cover_every_sensor_channel(pos, df):
    from hts_tape_env.fem_data import SENSOR_CHANNELS

    for label in df["ratio_label"].unique():
        channels = set(pos[pos["ratio_label"] == label]["channel"])
        assert set(SENSOR_CHANNELS).issubset(channels)
