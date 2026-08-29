"""
noise_models.py

Offline noise models for surrogate benchmarking: apply synthetic sensor noise
to rollout DataFrames so surrogates can be trained and evaluated at various
noise levels without regenerating trajectory data.

Two tiers:
  1. GaussianNoise — additive Gaussian scaled from the env defaults
  2. RealisticSensorNoise — channel-specific physically motivated noise
     (thermocouple drift, current transducer gain error, voltage tap noise)

Usage:
    from hts_tape_env.noise_models import NOISE_CONFIGS, apply_noise_to_rollout

    df_noisy = apply_noise_to_rollout(df, NOISE_CONFIGS["gauss_2x"], seed=42)
    # Train on df_noisy with feature_cols=FEATURES (noisy columns)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


DEFAULT_GAUSSIAN_STD = {"I": 0.5, "V": 1e-6, "T": 0.05}


@dataclass
class GaussianNoise:
    """Additive Gaussian noise scaled from the env defaults.

    scale=0 → noiseless, scale=1 → env default, scale=2 → 2x noise.
    """
    scale: float = 1.0
    base_std: dict = field(default_factory=lambda: dict(DEFAULT_GAUSSIAN_STD))

    def apply(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        out = df.copy()
        n = len(out)
        if self.scale == 0.0:
            out["I_A"] = out["I_true_A"]
            out["voltage_V"] = out["voltage_true_V"]
            out["T_surface_K"] = out["T_surface_true_K"]
            return out

        noise = rng.standard_normal((n, 3))
        out["I_A"] = out["I_true_A"] + noise[:, 0] * self.base_std["I"] * self.scale
        out["voltage_V"] = out["voltage_true_V"] + noise[:, 1] * self.base_std["V"] * self.scale
        out["T_surface_K"] = out["T_surface_true_K"] + noise[:, 2] * self.base_std["T"] * self.scale
        return out


REALISTIC_DEFAULTS = {
    "T_drift_sigma": 0.01,
    "T_drift_max": 0.5,
    "T_quantization": 0.1,
    "T_white_sigma": 0.02,
    "I_gain_sigma": 0.001,
    "I_white_sigma": 0.3,
    "V_rel_sigma": 0.05,
    "V_floor_sigma": 5e-7,
}


@dataclass
class RealisticSensorNoise:
    """Channel-specific physically motivated noise model.

    - T_surface: random-walk drift (bounded) + quantization + white noise
    - I: multiplicative gain error (per-episode calibration) + white noise
    - V: signal-proportional noise + nanovoltmeter floor noise
    """
    params: dict = field(default_factory=lambda: dict(REALISTIC_DEFAULTS))

    def apply(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        out = df.copy()
        p = self.params

        I_true = out["I_true_A"].values
        V_true = out["voltage_true_V"].values
        T_true = out["T_surface_true_K"].values

        n = len(out)
        episode_ids = out["episode_id"].values

        # --- Current: per-episode gain error + white noise ---
        unique_eps = np.unique(episode_ids)
        gain_map = {eid: rng.normal(0, p["I_gain_sigma"]) for eid in unique_eps}
        gains = np.array([gain_map[eid] for eid in episode_ids])
        I_obs = I_true * (1.0 + gains) + rng.normal(0, p["I_white_sigma"], n)

        # --- Voltage: signal-proportional + floor noise ---
        V_obs = V_true * (1.0 + rng.normal(0, p["V_rel_sigma"], n)) + \
                rng.normal(0, p["V_floor_sigma"], n)

        # --- Temperature: random-walk drift + quantization + white noise ---
        T_obs = np.empty(n)
        for eid in unique_eps:
            mask = episode_ids == eid
            idx = np.where(mask)[0]
            T_ep = T_true[idx]

            drift = np.zeros(len(idx))
            for i in range(1, len(idx)):
                drift[i] = drift[i - 1] + rng.normal(0, p["T_drift_sigma"])
                drift[i] = np.clip(drift[i], -p["T_drift_max"], p["T_drift_max"])

            T_noisy = T_ep + drift + rng.normal(0, p["T_white_sigma"], len(idx))
            q = p["T_quantization"]
            if q > 0:
                T_noisy = np.round(T_noisy / q) * q

            T_obs[idx] = T_noisy

        out["I_A"] = I_obs
        out["voltage_V"] = V_obs
        out["T_surface_K"] = T_obs
        return out


def apply_noise_to_rollout(
    df: pd.DataFrame,
    noise_model: GaussianNoise | RealisticSensorNoise,
    seed: int = 0,
) -> pd.DataFrame:
    """Apply a noise model to a rollout DataFrame.

    Reads from TRUE_FEATURES columns (I_true_A, voltage_true_V,
    T_surface_true_K), applies noise, writes to FEATURES columns
    (I_A, voltage_V, T_surface_K). T0_K is never noised.
    """
    rng = np.random.default_rng(seed)
    return noise_model.apply(df, rng)


NOISE_CONFIGS: dict[str, GaussianNoise | RealisticSensorNoise] = {
    "noiseless": GaussianNoise(scale=0.0),
    "gauss_0.5x": GaussianNoise(scale=0.5),
    "gauss_1x": GaussianNoise(scale=1.0),
    "gauss_2x": GaussianNoise(scale=2.0),
    "gauss_5x": GaussianNoise(scale=5.0),
    "realistic": RealisticSensorNoise(),
}
