"""Tests for hts_tape_env.noise_models."""

import numpy as np
import pandas as pd
import pytest

from hts_tape_env.noise_models import (
    GaussianNoise,
    RealisticSensorNoise,
    apply_noise_to_rollout,
    NOISE_CONFIGS,
)


def _make_rollout_df(n_episodes=5, steps_per_ep=20, seed=0):
    """Minimal rollout DataFrame with TRUE_FEATURES and episode structure."""
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(n_episodes):
        T0 = rng.uniform(20, 70)
        for step in range(steps_per_ep):
            I = 100 + step * 5 + rng.normal(0, 0.1)
            V = 1e-4 * step + rng.normal(0, 1e-7)
            T_surf = T0 + step * 0.2 + rng.normal(0, 0.01)
            rows.append(dict(
                episode_id=ep, step=step, T0_K=T0,
                I_true_A=I, voltage_true_V=V, T_surface_true_K=T_surf,
                I_A=I, voltage_V=V, T_surface_K=T_surf,
            ))
    return pd.DataFrame(rows)


class TestGaussianNoise:
    def test_scale_zero_identity(self):
        df = _make_rollout_df()
        out = apply_noise_to_rollout(df, GaussianNoise(scale=0.0))
        np.testing.assert_array_equal(out["I_A"].values, out["I_true_A"].values)
        np.testing.assert_array_equal(out["voltage_V"].values, out["voltage_true_V"].values)
        np.testing.assert_array_equal(out["T_surface_K"].values, out["T_surface_true_K"].values)

    def test_scale_nonzero_adds_noise(self):
        df = _make_rollout_df()
        out = apply_noise_to_rollout(df, GaussianNoise(scale=1.0))
        assert not np.allclose(out["I_A"].values, out["I_true_A"].values)

    def test_noise_scales_with_multiplier(self):
        df = _make_rollout_df(n_episodes=50, steps_per_ep=100)
        out_1x = apply_noise_to_rollout(df, GaussianNoise(scale=1.0), seed=7)
        out_3x = apply_noise_to_rollout(df, GaussianNoise(scale=3.0), seed=7)
        std_1x = (out_1x["I_A"] - out_1x["I_true_A"]).std()
        std_3x = (out_3x["I_A"] - out_3x["I_true_A"]).std()
        assert std_3x > 2.5 * std_1x

    def test_seeded_reproducibility(self):
        df = _make_rollout_df()
        out1 = apply_noise_to_rollout(df, GaussianNoise(scale=2.0), seed=42)
        out2 = apply_noise_to_rollout(df, GaussianNoise(scale=2.0), seed=42)
        np.testing.assert_array_equal(out1["I_A"].values, out2["I_A"].values)

    def test_T0_unchanged(self):
        df = _make_rollout_df()
        out = apply_noise_to_rollout(df, GaussianNoise(scale=5.0))
        np.testing.assert_array_equal(out["T0_K"].values, df["T0_K"].values)

    def test_true_features_preserved(self):
        df = _make_rollout_df()
        out = apply_noise_to_rollout(df, GaussianNoise(scale=2.0))
        np.testing.assert_array_equal(out["I_true_A"].values, df["I_true_A"].values)


class TestRealisticSensorNoise:
    def test_adds_noise(self):
        df = _make_rollout_df()
        out = apply_noise_to_rollout(df, RealisticSensorNoise())
        assert not np.allclose(out["I_A"].values, out["I_true_A"].values)
        assert not np.allclose(out["voltage_V"].values, out["voltage_true_V"].values)
        assert not np.allclose(out["T_surface_K"].values, out["T_surface_true_K"].values)

    def test_drift_bounded(self):
        df = _make_rollout_df(n_episodes=20, steps_per_ep=200)
        params = dict(T_drift_sigma=0.1, T_drift_max=0.3,
                      T_quantization=0.0, T_white_sigma=0.0,
                      I_gain_sigma=0.0, I_white_sigma=0.0,
                      V_rel_sigma=0.0, V_floor_sigma=0.0)
        out = apply_noise_to_rollout(df, RealisticSensorNoise(params=params))
        residual = (out["T_surface_K"] - out["T_surface_true_K"]).abs()
        assert residual.max() <= 0.3 + 1e-10

    def test_voltage_signal_proportional(self):
        df = _make_rollout_df(n_episodes=50, steps_per_ep=100)
        params = dict(T_drift_sigma=0, T_drift_max=0, T_quantization=0,
                      T_white_sigma=0, I_gain_sigma=0, I_white_sigma=0,
                      V_rel_sigma=0.1, V_floor_sigma=0.0)
        out = apply_noise_to_rollout(df, RealisticSensorNoise(params=params))
        V_err = (out["voltage_V"] - out["voltage_true_V"]).abs()
        V_abs = out["voltage_true_V"].abs()
        big = V_abs > V_abs.median()
        small = V_abs <= V_abs.median()
        if small.sum() > 0 and big.sum() > 0:
            assert V_err[big].mean() > V_err[small].mean()

    def test_quantization(self):
        df = _make_rollout_df()
        params = dict(T_drift_sigma=0, T_drift_max=0, T_quantization=0.5,
                      T_white_sigma=0, I_gain_sigma=0, I_white_sigma=0,
                      V_rel_sigma=0, V_floor_sigma=0)
        out = apply_noise_to_rollout(df, RealisticSensorNoise(params=params))
        residuals = out["T_surface_K"].values % 0.5
        np.testing.assert_allclose(residuals, np.round(residuals / 0.5) * 0.5, atol=1e-10)

    def test_seeded_reproducibility(self):
        df = _make_rollout_df()
        out1 = apply_noise_to_rollout(df, RealisticSensorNoise(), seed=7)
        out2 = apply_noise_to_rollout(df, RealisticSensorNoise(), seed=7)
        np.testing.assert_array_equal(out1["I_A"].values, out2["I_A"].values)


class TestNoiseConfigs:
    def test_all_configs_present(self):
        expected = {"noiseless", "gauss_0.5x", "gauss_1x", "gauss_2x", "gauss_5x", "realistic"}
        assert set(NOISE_CONFIGS.keys()) == expected

    def test_all_configs_apply(self):
        df = _make_rollout_df()
        for name, model in NOISE_CONFIGS.items():
            out = apply_noise_to_rollout(df, model)
            assert len(out) == len(df), f"Config {name} changed row count"
            assert "I_A" in out.columns
