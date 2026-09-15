"""
test_bi2223_rollout.py -- Bi-2223 trajectory generator (bi2223_rollout.py).
"""

import numpy as np

from hts_tape_env.bi2223_rollout import generate_bi2223_grid_dataset
from hts_tape_env.seq_data import TRUE_FEATURES


def test_grid_dataset_has_true_feature_columns():
    df = generate_bi2223_grid_dataset(np.linspace(77.16, 79.0, 3))
    for col in TRUE_FEATURES:
        assert col in df.columns


def test_margin_decreases_monotonically_within_an_episode():
    df = generate_bi2223_grid_dataset(np.array([77.16]))
    g = df[df.episode_id == 0].sort_values("step")
    assert np.all(np.diff(g["margin_norm"].to_numpy()) < 0)


def test_out_of_range_T0_skipped_not_crashed():
    """T0 well past the validated range (see physics_bi2223.py) should be
    silently skipped (no episode), not raise."""
    df = generate_bi2223_grid_dataset(np.array([85.0]))
    assert len(df) == 0
