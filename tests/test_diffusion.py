"""Tests for hts_tape_env.diffusion (Conditional DDPM surrogate)."""

import numpy as np
import pandas as pd
import pytest
import torch

from hts_tape_env.diffusion import (
    cosine_beta_schedule,
    linear_beta_schedule,
    SinusoidalPosEmb,
    ConditionalUNet1D,
    DiffusionSurrogate,
    train_diffusion_surrogate,
)


def _make_rollout_df(n_episodes=10, steps_per_ep=30, seed=0):
    """Minimal rollout DataFrame matching the rollout.py schema."""
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


class TestNoiseSchedule:
    def test_cosine_shape(self):
        betas = cosine_beta_schedule(200)
        assert betas.shape == (200,)
        assert (betas > 0).all()
        assert (betas < 1).all()

    def test_linear_shape(self):
        betas = linear_beta_schedule(100)
        assert betas.shape == (100,)
        assert betas[0] < betas[-1]

    def test_alpha_bar_decreasing(self):
        betas = cosine_beta_schedule(200)
        alpha_bar = torch.cumprod(1 - betas, dim=0)
        diffs = alpha_bar[1:] - alpha_bar[:-1]
        assert (diffs <= 0).all()


class TestSinusoidalPosEmb:
    def test_shape(self):
        emb = SinusoidalPosEmb(64)
        t = torch.tensor([0, 50, 100, 199])
        out = emb(t)
        assert out.shape == (4, 64)


class TestConditionalUNet1D:
    def test_forward_shape(self):
        unet = ConditionalUNet1D(in_ch=1, cond_dim=4, channels=(16, 32))
        x = torch.randn(2, 1, 32)
        t = torch.tensor([0, 100])
        cond = torch.randn(2, 4)
        out = unet(x, t, cond)
        assert out.shape == (2, 1, 32)


class TestDiffusionSurrogate:
    def test_forward_shapes(self):
        model = DiffusionSurrogate(n_features=4, d_cond=4, channels=(16, 32),
                                    n_timesteps=50, n_ddim_steps=10, max_seq_len=64)
        X = torch.randn(2, 32, 4)
        lengths = torch.tensor([32, 20])
        out = model(X, lengths)
        assert out["margin"].shape == (2, 32)
        assert out["breach_logit"].shape == (2, 32)
        assert out["istar_log"].shape == (2,)
        assert out["tstar"].shape == (2,)
        assert out["time_margin_log"].shape == (2,)

    def test_training_step_scalar(self):
        model = DiffusionSurrogate(n_features=4, d_cond=4, channels=(16, 32),
                                    n_timesteps=50)
        x_0 = torch.randn(3, 1, 32)
        cond = torch.randn(3, 4)
        loss = model.training_step(x_0, cond)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_q_sample_noise_at_max(self):
        model = DiffusionSurrogate(n_features=4, d_cond=4, channels=(16, 32),
                                    n_timesteps=100)
        x_0 = torch.ones(2, 1, 16)
        noise = torch.randn(2, 1, 16)
        t = torch.tensor([99, 99])
        x_t = model.q_sample(x_0, t, noise)
        # At t=T-1, alpha_bar is small, so x_t should be mostly noise
        assert x_t.std() > 0.5


class TestTrainDiffusionSurrogate:
    @pytest.mark.slow
    def test_smoke_train(self):
        df = _make_rollout_df(n_episodes=20, steps_per_ep=24)
        result = train_diffusion_surrogate(
            df, epochs=3, n_timesteps=20, n_ddim_steps=5,
            batch_size=8, seed=42, verbose=False,
            model_kwargs={"d_cond": 4, "channels": (8, 16), "max_seq_len": 32},
        )
        assert result.arch == "diffusion"
        assert "istar_mae_A" in result.metrics
        assert "breach_accuracy" in result.metrics
        assert result.model is not None
        assert result.normalizer is not None

    @pytest.mark.slow
    def test_save_load_roundtrip(self, tmp_path):
        from hts_tape_env.seq_surrogates import load_seq_surrogate
        df = _make_rollout_df(n_episodes=15, steps_per_ep=24)
        result = train_diffusion_surrogate(
            df, epochs=2, n_timesteps=20, n_ddim_steps=5,
            batch_size=8, seed=42, verbose=False,
            model_kwargs={"d_cond": 4, "channels": (8, 16), "max_seq_len": 32},
        )
        path = tmp_path / "test_diffusion.pt"
        result.save(path)
        loaded = load_seq_surrogate(path)
        assert loaded.arch == "diffusion"
        assert loaded.metrics["istar_mae_A"] == result.metrics["istar_mae_A"]
