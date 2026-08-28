"""
adaptive_sampling.py

Boundary-aware adaptive sampling strategies for the thermal runaway
parameter space.  Concentrates training samples where the runaway boundary
is hardest to learn: near I/I* ≈ 1 and in high-curvature regions of the
I*(T₀) surface.

Two composable strategies:

  1. **Boundary-concentrated I/I* sampling** — truncated normal centered at 1.0
     instead of the uniform safe/unsafe coin flip.  Eliminates the gap at the
     boundary and naturally concentrates density where the surrogate must be
     most accurate.

  2. **Curvature-weighted T₀ sampling** — precompute |d²I*/dT₀²| on a coarse
     grid, then sample T₀ from a mixture of curvature-weighted + uniform.
     Concentrates samples at material-property transitions (30-40 K for REBCO,
     79-80 K for Bi-2223) where the I*(T₀) surface bends fastest.

Both integrate into existing datagen as drop-in replacements via
`generate_adaptive_dataset` (static data) and
`generate_adaptive_rollout_dataset` (trajectory data).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


# ── Strategy 1: Boundary-concentrated I/I* ──────────────────────────────────

def boundary_I_fractions(
    rng: np.random.Generator,
    n: int,
    sigma: float = 0.15,
    lo: float = 0.3,
    hi: float = 1.3,
) -> np.ndarray:
    """
    Draw n I/I* fractions from a truncated normal centered at 1.0.

    ~68% of samples fall within ±σ of the boundary (I/I* = 1),
    vs the current approach which has a dead zone between 0.99 and 1.01.
    """
    samples = []
    while len(samples) < n:
        batch = rng.normal(1.0, sigma, size=n - len(samples))
        valid = batch[(batch >= lo) & (batch <= hi)]
        samples.extend(valid.tolist())
    return np.array(samples[:n])


# ── Strategy 2: Curvature-weighted T₀ ───────────────────────────────────────

def _build_curvature_cdf(
    T0_range: tuple[float, float],
    solve_fn: Callable[[float], tuple[float, float]],
    n_grid: int = 100,
    dT0: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a CDF for sampling T₀ proportional to |d²I*/dT₀²|.

    Parameters
    ----------
    T0_range : (lo, hi)
    solve_fn : T0 -> (Istar, Tstar), the conductor's solve_runaway
    n_grid : coarse grid size for curvature estimation
    dT0 : finite-difference step; defaults to 2% of the range
    """
    lo, hi = T0_range
    if dT0 is None:
        dT0 = max(0.02 * (hi - lo), 0.1)

    T0s = np.linspace(lo + dT0, hi - dT0, n_grid)
    curvatures = np.zeros(n_grid)

    for i, T0 in enumerate(T0s):
        i_lo = solve_fn(T0 - dT0)[0]
        i_mid = solve_fn(T0)[0]
        i_hi = solve_fn(T0 + dT0)[0]
        if all(np.isfinite(v) and v > 0 for v in [i_lo, i_mid, i_hi]):
            curvatures[i] = abs(i_hi - 2 * i_mid + i_lo) / dT0 ** 2

    # Ensure non-zero everywhere for coverage
    curvatures = np.maximum(curvatures, curvatures[curvatures > 0].min() * 0.01
                            if np.any(curvatures > 0) else 1.0)

    # Build piecewise-constant CDF
    weights = curvatures / curvatures.sum()
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0  # numerical safety
    return T0s, cdf


def curvature_weighted_T0(
    rng: np.random.Generator,
    n: int,
    T0_range: tuple[float, float],
    solve_fn: Callable[[float], tuple[float, float]],
    blend: float = 0.5,
    n_grid: int = 100,
) -> np.ndarray:
    """
    Draw T₀ values from a mixture distribution:
      blend × curvature-weighted  +  (1 - blend) × uniform.

    The curvature-weighted component concentrates samples where |d²I*/dT₀²|
    is largest.  The uniform component ensures coverage across the full range.
    """
    n_curv = int(n * blend)
    n_unif = n - n_curv

    # Uniform component
    T0_uniform = rng.uniform(*T0_range, size=n_unif)

    if n_curv == 0:
        return T0_uniform

    # Curvature component via inverse CDF
    T0_grid, cdf = _build_curvature_cdf(T0_range, solve_fn, n_grid=n_grid)
    u = rng.uniform(size=n_curv)
    indices = np.searchsorted(cdf, u)
    indices = np.clip(indices, 0, len(T0_grid) - 1)

    # Add jitter within the grid cell
    dx = T0_grid[1] - T0_grid[0] if len(T0_grid) > 1 else 1.0
    T0_curv = T0_grid[indices] + rng.uniform(-dx / 2, dx / 2, size=n_curv)
    T0_curv = np.clip(T0_curv, *T0_range)

    combined = np.concatenate([T0_uniform, T0_curv])
    rng.shuffle(combined)
    return combined


# ── Adaptive static dataset (REBCO) ─────────────────────────────────────────

def generate_adaptive_dataset_rebco(
    n_samples: int = 5000,
    *,
    mode: str = "stab",
    T0_range: tuple[float, float] = (5.0, 75.0),
    lam_range: tuple[float, float] = (20, 200),
    sigma: float = 0.15,
    blend: float = 0.5,
    seed: int = 7,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Adaptive replacement for datagen.generate_dataset (REBCO conductors).

    Uses curvature-weighted T₀ sampling and boundary-concentrated I/I*.
    """
    from hts_tape_env.physics import (
        Tc as TC_DEFAULT,
        solve_runaway,
        stable_surface_T,
        tape_voltage,
    )
    from hts_tape_env.thermal import time_margin_to_safe

    n, Ic0, Tc = 20.0, 500.0, TC_DEFAULT
    rng = np.random.default_rng(seed)

    # Build curvature-weighted T₀ sampler at fixed λ=100 (median of range)
    lam_mid = 0.5 * (lam_range[0] + lam_range[1])
    solve_at_lam_mid = lambda T0: solve_runaway(T0, n, mode=mode, lam_override=lam_mid)

    if verbose:
        print(f"  Building curvature CDF (mode={mode})...")
    T0_vals = curvature_weighted_T0(
        rng, n_samples, T0_range, solve_at_lam_mid, blend=blend,
    )
    lam_vals = rng.uniform(*lam_range, size=n_samples)
    I_fracs = boundary_I_fractions(rng, n_samples, sigma=sigma)

    rows = []
    for k in range(n_samples):
        if verbose and k and k % 1000 == 0:
            print(f"  generated {k}/{n_samples} attempts, {len(rows)} kept")

        T0 = float(T0_vals[k])
        lam = float(lam_vals[k])

        Istar, Tstar = solve_runaway(T0, n, mode=mode, lam_override=lam, Ic0=Ic0, Tc=Tc)
        if not np.isfinite(Istar) or Istar <= 0:
            continue

        frac = float(I_fracs[k])
        I = frac * Istar
        runaway = int(I > Istar)

        if runaway:
            tm_s = time_margin_to_safe(I, Tstar)
            if not np.isfinite(tm_s):
                continue
            Tsurf = Tstar
            V = tape_voltage(I, min(Tstar, Tc - 1e-4), n, mode=mode, Ic0=Ic0, Tc=Tc)
            time_margin_ms = tm_s * 1000.0
        else:
            Tsurf = stable_surface_T(I, T0, n, mode=mode, lam_override=lam,
                                     Ic0=Ic0, Tc=Tc, T_hi=Tstar)
            if not np.isfinite(Tsurf):
                continue
            V = tape_voltage(I, min(Tsurf, Tc - 1e-4), n, mode=mode, Ic0=Ic0, Tc=Tc)
            time_margin_ms = np.nan

        rows.append(dict(
            T0_K=T0, I_A=I, voltage_V=V, T_surface_K=Tsurf,
            lam=lam, Istar_A=Istar, margin_norm=(Istar - I) / Istar,
            Tstar_K=Tstar, runaway=runaway, time_margin_ms=time_margin_ms,
        ))

    df = pd.DataFrame(rows)
    if verbose:
        frac_unsafe = df["runaway"].mean() if len(df) else float("nan")
        near_boundary = (df["margin_norm"].abs() < 0.05).mean() if len(df) else 0
        print(f"Generated {len(df)} rows; {frac_unsafe:.0%} unsafe, "
              f"{near_boundary:.0%} within 5% of boundary.")
    return df


# ── Adaptive static dataset (Bi-2223) ───────────────────────────────────────

def generate_adaptive_dataset_bi2223(
    n_samples: int = 3000,
    *,
    mode: str = "no_stab",
    T0_range: tuple[float, float] = (77.16, 81.0),
    sigma: float = 0.15,
    blend: float = 0.5,
    seed: int = 7,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Adaptive replacement for datagen_bi2223.generate_dataset.
    """
    from hts_tape_env.physics_bi2223 import (
        T_X_K,
        solve_runaway,
        stable_surface_T,
        tape_voltage,
    )
    from hts_tape_env.thermal_bi2223 import time_margin_to_safe

    rng = np.random.default_rng(seed)

    solve_fn = lambda T0: solve_runaway(T0, mode=mode)

    if verbose:
        print(f"  Building curvature CDF (mode={mode})...")
    T0_vals = curvature_weighted_T0(
        rng, n_samples, T0_range, solve_fn, blend=blend,
    )
    I_fracs = boundary_I_fractions(rng, n_samples, sigma=sigma)

    rows = []
    for k in range(n_samples):
        if verbose and k and k % 1000 == 0:
            print(f"  generated {k}/{n_samples} attempts, {len(rows)} kept")

        T0 = float(T0_vals[k])
        Istar, Tstar = solve_runaway(T0, mode=mode)
        if not np.isfinite(Istar) or Istar <= 0:
            continue

        frac = float(I_fracs[k])
        I = frac * Istar
        runaway = int(I > Istar)

        if runaway:
            tm_s = time_margin_to_safe(I, Tstar)
            if not np.isfinite(tm_s):
                continue
            Tsurf = Tstar
            V = tape_voltage(I, min(Tstar, T_X_K - 1e-4), mode=mode)
            time_margin_ms = tm_s * 1000.0
        else:
            Tsurf = stable_surface_T(I, T0, T_hi=Tstar, mode=mode)
            if not np.isfinite(Tsurf):
                continue
            V = tape_voltage(I, min(Tsurf, T_X_K - 1e-4), mode=mode)
            time_margin_ms = np.nan

        rows.append(dict(
            T0_K=T0, I_A=I, voltage_V=V, T_surface_K=Tsurf,
            Istar_A=Istar, margin_norm=(Istar - I) / Istar,
            Tstar_K=Tstar, runaway=runaway, time_margin_ms=time_margin_ms,
        ))

    df = pd.DataFrame(rows)
    if verbose:
        frac_unsafe = df["runaway"].mean() if len(df) else float("nan")
        near_boundary = (df["margin_norm"].abs() < 0.05).mean() if len(df) else 0
        print(f"Generated {len(df)} rows; {frac_unsafe:.0%} unsafe, "
              f"{near_boundary:.0%} within 5% of boundary.")
    return df


# ── Adaptive rollout dataset ────────────────────────────────────────────────

def generate_adaptive_rollout_dataset(
    n_episodes: int,
    *,
    env,
    T0_sampler: Callable,
    lam_sampler: Callable | None = None,
    seed: int = 7,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate trajectory data with adaptive T₀ (and optionally λ) sampling.

    Uses env.reset_at(T0, lam) for each episode instead of env.reset(),
    falling back to env.reset() on invalid (T0, lam) pairs.

    Parameters
    ----------
    env : QuenchDetectorEnv or similar with reset_at()
    T0_sampler : callable(rng, n) -> array of T₀ values
    lam_sampler : callable(rng, n) -> array of λ values, or None for uniform
    """
    from hts_tape_env.rollout import rollout_episode

    rng = np.random.default_rng(seed)
    T0_vals = T0_sampler(rng, n_episodes)
    if lam_sampler is not None:
        lam_vals = lam_sampler(rng, n_episodes)
    else:
        lam_vals = [None] * n_episodes

    rows = []
    n_runaway = 0
    n_fallback = 0

    for ep in range(n_episodes):
        if verbose and ep and ep % 100 == 0:
            print(f"  rolled out {ep}/{n_episodes} episodes, {len(rows)} steps")

        T0 = float(T0_vals[ep])
        lam = float(lam_vals[ep]) if lam_vals[ep] is not None else None

        try:
            if hasattr(env, "reset_at") and lam is not None:
                obs, info = env.reset_at(T0, lam, seed=seed + ep)
            elif hasattr(env, "reset_at"):
                obs, info = env.reset_at(T0, seed=seed + ep)
            else:
                obs, info = env.reset(seed=seed + ep)

            if not np.isfinite(info.get("I_star", float("nan"))) or info.get("I_star", 0) <= 0:
                raise ValueError("invalid I*")
        except (ValueError, KeyError):
            obs, info = env.reset(seed=seed + ep)
            n_fallback += 1

        ep_rows = rollout_episode(env, ep, obs=obs, info=info)
        rows.extend(ep_rows)
        n_runaway += int(ep_rows[-1]["runaway"])

    df = pd.DataFrame(rows)
    if verbose:
        print(f"Generated {len(df)} steps over {n_episodes} episodes "
              f"({n_runaway}/{n_episodes} runaway, {n_fallback} fallbacks)")
    return df
