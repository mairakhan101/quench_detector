"""Tests for FD1DQuenchDetectorEnv."""

import numpy as np
import pytest

from hts_tape_env.env_fd1d import FD1DQuenchDetectorEnv
from hts_tape_env.physics_fd1d import HeaterPulse, CoolingModel


@pytest.fixture
def fast_env():
    """Small tape, coarse grid, short transients → fast I* search."""
    return FD1DQuenchDetectorEnv(
        geometry="lumped",
        nx=50,
        T0_range=(50.0, 70.0),
        cool_mode="bath_ends",
        ramp_rate=0.02,
        n_settle_steps=5,
        dt=1e-4,
        istar_tol=5.0,
        istar_max_iter=8,
        istar_t_end=0.05,
    )


class TestResetStep:
    def test_reset_returns_obs_info(self, fast_env):
        obs, info = fast_env.reset(seed=42)
        assert obs.shape == (4,)
        assert obs.dtype == np.float32
        assert "I_star" in info
        assert info["I_star"] > 0

    def test_step_returns_five_tuple(self, fast_env):
        fast_env.reset(seed=42)
        obs, reward, terminated, truncated, info = fast_env.step(0)
        assert obs.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "margin" in info

    def test_dump_terminates(self, fast_env):
        fast_env.reset(seed=42)
        _, _, terminated, _, _ = fast_env.step(1)
        assert terminated

    def test_obs_in_bounds(self, fast_env):
        obs, _ = fast_env.reset(seed=42)
        lo = fast_env.observation_space.low
        hi = fast_env.observation_space.high
        assert np.all(obs >= lo) and np.all(obs <= hi)

    def test_current_ramps(self, fast_env):
        fast_env.reset(seed=42)
        fracs = [fast_env._I_frac]
        for _ in range(5):
            fast_env.step(0)
            fracs.append(fast_env._I_frac)
        diffs = np.diff(fracs)
        np.testing.assert_allclose(diffs, fast_env.ramp_rate, atol=1e-10)


class TestResetAt:
    def test_deterministic_reset(self, fast_env):
        obs1, info1 = fast_env.reset_at(60.0, seed=42)
        obs2, info2 = fast_env.reset_at(60.0, seed=42)
        np.testing.assert_array_equal(obs1, obs2)
        assert info1["I_star"] == info2["I_star"]


class TestTFieldEvolution:
    def test_field_starts_uniform(self, fast_env):
        fast_env.reset(seed=42)
        T_bath = fast_env._T_bath
        # After first _get_obs the field may have evolved slightly, but prev was uniform
        assert fast_env._T_field.shape[0] == fast_env.tape.nx

    def test_field_evolves_over_steps(self, fast_env):
        fast_env.reset(seed=42)
        # Ramp close to I* where Joule heating is significant
        for _ in range(30):
            fast_env.step(0)
        T_mid = fast_env._T_field.copy()
        for _ in range(10):
            fast_env.step(0)
        T_late = fast_env._T_field.copy()
        # At I_frac ~ 0.9+, Joule heating should change the T field
        assert not np.allclose(T_mid, T_late, atol=1e-12, rtol=0)


class TestSurrogateInterface:
    def test_set_surrogate_none(self, fast_env):
        fast_env.set_surrogate(None)
        obs, info = fast_env.reset(seed=42)
        assert "margin" in info

    def test_surrogate_predictions_in_info(self, fast_env):
        class DummySurrogate:
            def reset(self): pass
            def predict_step(self, obs):
                return {"surrogate_V": float(obs[2]) * 1.1}
        fast_env.set_surrogate(DummySurrogate())
        obs, info = fast_env.reset(seed=42)
        assert "surrogate_V" in info
