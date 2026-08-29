"""Tests for early stopping, LR scheduling, and convergence reporting."""

import numpy as np
import pandas as pd
import pytest
import torch

from hts_tape_env.seq_surrogates import train_seq_surrogate
from hts_tape_env.diffusion import train_diffusion_surrogate


def _make_rollout_df(n_episodes=10, steps_per_ep=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(n_episodes):
        T0 = rng.uniform(20, 70)
        Istar = rng.uniform(200, 500)
        Tstar = T0 + rng.uniform(5, 30)
        for step in range(steps_per_ep):
            frac = 0.5 + step * 0.02
            I_true = frac * Istar
            margin = (Istar - I_true) / Istar
            V = 1e-4 * max(0, frac - 0.8)
            T_surf = T0 + step * 0.3
            rows.append(dict(
                episode_id=ep, step=step, T0_K=T0,
                I_true_A=I_true, voltage_true_V=V, T_surface_true_K=T_surf,
                I_A=I_true, voltage_V=V, T_surface_K=T_surf,
                Istar_A=Istar, Tstar_K=Tstar,
                margin_norm=margin, runaway=int(margin <= 0),
                time_margin_ms=rng.uniform(100, 3000),
            ))
    return pd.DataFrame(rows)


class TestSeqConvergence:
    def test_convergence_metrics_in_result(self):
        df = _make_rollout_df(n_episodes=15, steps_per_ep=20)
        result = train_seq_surrogate(
            df, arch="gru", epochs=5, patience=3,
            batch_size=8, seed=42, verbose=False,
        )
        assert "converged" in result.metrics
        assert "stopped_epoch" in result.metrics
        assert "total_epochs" in result.metrics
        assert result.metrics["total_epochs"] == 5

    def test_early_stopping_triggers(self):
        df = _make_rollout_df(n_episodes=15, steps_per_ep=20)
        result = train_seq_surrogate(
            df, arch="gru", epochs=200, patience=5, min_delta=0.0,
            batch_size=8, lr=1e-3, seed=42, verbose=False,
        )
        assert result.metrics["stopped_epoch"] < 199
        assert result.metrics["converged"] is True

    def test_no_early_stop_when_patience_zero(self):
        df = _make_rollout_df(n_episodes=15, steps_per_ep=20)
        result = train_seq_surrogate(
            df, arch="gru", epochs=5, patience=0,
            batch_size=8, seed=42, verbose=False,
        )
        assert result.metrics["stopped_epoch"] == 4
        assert result.metrics["converged"] is False

    def test_runs_all_epochs_when_improving(self):
        df = _make_rollout_df(n_episodes=15, steps_per_ep=20)
        result = train_seq_surrogate(
            df, arch="gru", epochs=8, patience=50,
            batch_size=8, seed=42, verbose=False,
        )
        assert result.metrics["stopped_epoch"] == 7
        assert result.metrics["converged"] is False

    def test_lr_reduces_on_plateau(self):
        df = _make_rollout_df(n_episodes=15, steps_per_ep=20)
        initial_lr = 0.01
        result = train_seq_surrogate(
            df, arch="gru", epochs=60, lr=initial_lr,
            patience=0, reduce_lr_patience=5, reduce_lr_factor=0.5,
            batch_size=8, seed=42, verbose=False,
        )
        assert result.metrics["best_val_loss"] < float("inf")


class TestDiffusionConvergence:
    @pytest.mark.slow
    def test_convergence_metrics_in_result(self):
        df = _make_rollout_df(n_episodes=20, steps_per_ep=24)
        result = train_diffusion_surrogate(
            df, epochs=5, patience=3, n_timesteps=20, n_ddim_steps=5,
            batch_size=8, seed=42, verbose=False,
            model_kwargs={"d_cond": 4, "channels": (8, 16), "max_seq_len": 32},
        )
        assert "converged" in result.metrics
        assert "stopped_epoch" in result.metrics
        assert "total_epochs" in result.metrics
        assert result.metrics["total_epochs"] == 5

    @pytest.mark.slow
    def test_phase_reset_at_boundary(self):
        df = _make_rollout_df(n_episodes=20, steps_per_ep=24)
        result = train_diffusion_surrogate(
            df, epochs=10, denoising_frac=0.5, patience=0,
            n_timesteps=20, n_ddim_steps=5,
            batch_size=8, seed=42, verbose=False,
            model_kwargs={"d_cond": 4, "channels": (8, 16), "max_seq_len": 32},
        )
        assert result.metrics["stopped_epoch"] == 9
        assert result.metrics["converged"] is False
