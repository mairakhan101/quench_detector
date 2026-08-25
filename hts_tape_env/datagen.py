"""
datagen.py

Generate labelled surrogate-training datasets by sweeping the analytic solver in
physics.py. One row per operating point; every row carries *all* labels so a
single table can feed several surrogates (see surrogates.py).

Columns
-------
    T0_K            bath temperature [K]                       (input)
    I_A             transport current [A]                      (input)
    voltage_V       tape voltage at the stable surface T [V]   (input)
    T_surface_K     stable surface temperature [K]             (input)
    lam             cooling lambda [W/m/K] (None -> lambda(T))  (input)
    Istar_A         runaway current at this T0, lambda [A]     (label: A / 2D / raw)
    margin_norm     (I* - I) / I*                              (label: B)
    Tstar_K         runaway temperature [K]
    runaway         1 if I > I*, else 0
    time_margin_ms  adiabatic time-to-300K at runaway [ms]     (label: time-margin)

Typical use
-----------
    from hts_tape_env.datagen import generate_dataset
    df = generate_dataset(n_samples=5000, seed=7)
    df.to_parquet("data/surrogate_main.parquet")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hts_tape_env.physics import (
    Tc as TC_DEFAULT,
    solve_runaway,
    stable_surface_T,
    tape_voltage,
)
from hts_tape_env.thermal import time_margin_to_safe

# Fixed conductor for the default dataset (one conductor, matching the paper)
N_FIXED = 20.0
IC0_FIXED = 500.0


def generate_dataset(
    n_samples: int = 5000,
    *,
    n: float = N_FIXED,
    Ic0: float = IC0_FIXED,
    Tc: float = TC_DEFAULT,
    mode: str = "stab",
    T0_range: tuple[float, float] = (5.0, 75.0),
    lam_range: tuple[float, float] | None = None,
    frac_unsafe: float = 0.30,
    seed: int = 7,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Sweep (T0, [lambda], I) and label each operating point.

    Parameters
    ----------
    n_samples : int
        Number of operating points to attempt (rows dropped where the solver
        finds no interior runaway, so the returned frame may be shorter).
    lam_range : (float, float) or None
        If given, lambda is sampled uniformly from this range and stored in the
        `lam` column (enables the (T0, lambda) -> I* surrogate). If None, the
        temperature-dependent NIST brass lambda(T) is used and `lam` is NaN.
    frac_unsafe : float
        Fraction of rows sampled above I* (I/I* in (1.01, 1.20)); the rest are
        below (I/I* in (0.50, 0.99)). Spans the runaway boundary.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for k in range(n_samples):
        if verbose and k and k % 1000 == 0:
            print(f"  generated {k}/{n_samples} attempts, {len(rows)} kept")

        T0 = float(rng.uniform(*T0_range))
        lam = float(rng.uniform(*lam_range)) if lam_range is not None else None

        Istar, Tstar = solve_runaway(
            T0, n, mode=mode, lam_override=lam, Ic0=Ic0, Tc=Tc
        )
        if not np.isfinite(Istar) or Istar <= 0:
            continue

        if rng.random() >= frac_unsafe:
            # Safe branch: below runaway, stable surface temperature exists
            frac = rng.uniform(0.50, 0.99)
            I = frac * Istar
            Tsurf = stable_surface_T(I, T0, n, mode=mode, lam_override=lam,
                                     Ic0=Ic0, Tc=Tc, T_hi=Tstar)
            if not np.isfinite(Tsurf):
                continue
            V = tape_voltage(I, min(Tsurf, Tc - 1e-4), n, mode=mode,
                             Ic0=Ic0, Tc=Tc)
            runaway = 0
            time_margin_ms = np.nan
        else:
            # Unsafe branch: above runaway, use adiabatic time margin
            frac = rng.uniform(1.01, 1.20)
            I = frac * Istar
            tm_s = time_margin_to_safe(I, Tstar)
            if not np.isfinite(tm_s):
                continue
            Tsurf = Tstar
            V = tape_voltage(I, min(Tstar, Tc - 1e-4), n, mode=mode,
                             Ic0=Ic0, Tc=Tc)
            runaway = 1
            time_margin_ms = tm_s * 1000.0

        rows.append(dict(
            T0_K=T0,
            I_A=I,
            voltage_V=V,
            T_surface_K=Tsurf,
            lam=(np.nan if lam is None else lam),
            Istar_A=Istar,
            margin_norm=(Istar - I) / Istar,
            Tstar_K=Tstar,
            runaway=runaway,
            time_margin_ms=time_margin_ms,
        ))

    df = pd.DataFrame(rows)
    if verbose:
        frac = df["runaway"].mean() if len(df) else float("nan")
        print(f"Generated {len(df)} rows; {frac:.0%} unsafe.")
    return df
