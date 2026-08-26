"""
datagen_bi2223.py

Generate labelled surrogate-training datasets for the Bi-2223 conductor
by sweeping the analytic solver in physics_bi2223.py. Mirrors datagen.py
but for the Bi-2223 conductor's narrower validated range and constant-
cooling physics (see physics_bi2223.py docstring).

Columns
-------
    T0_K            bath temperature [K]                       (input)
    I_A             transport current [A]                      (input)
    voltage_V       tape voltage at the stable surface T [V]   (input)
    T_surface_K     stable surface temperature [K]             (input)
    Istar_A         runaway current at this T0 [A]             (label)
    margin_norm     (I* - I) / I*                              (label)
    Tstar_K         runaway temperature [K]
    runaway         1 if I > I*, else 0
    time_margin_ms  adiabatic time-to-300K at runaway [ms]     (label)

Note: no `lam` column -- Bi-2223 cooling is a measured constant
(eta_c = 4.56 W/K), not a swept parameter. The `two_d` surrogate spec
from surrogates.py is therefore inapplicable to Bi-2223 data.

Typical use
-----------
    from hts_tape_env.datagen_bi2223 import generate_dataset
    df = generate_dataset(n_samples=3000, seed=7)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hts_tape_env.physics_bi2223 import (
    T_X_K,
    solve_runaway,
    stable_surface_T,
    tape_voltage,
)
from hts_tape_env.thermal_bi2223 import time_margin_to_safe


def generate_dataset(
    n_samples: int = 3000,
    *,
    T0_range: tuple[float, float] = (77.16, 80.5),
    frac_unsafe: float = 0.30,
    seed: int = 7,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Sweep (T0, I) and label each operating point for the Bi-2223 conductor.

    The narrower T0 range and solve_runaway's sanity cap mean more points
    are dropped than REBCO datagen; the returned frame may be significantly
    shorter than n_samples.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for k in range(n_samples):
        if verbose and k and k % 1000 == 0:
            print(f"  generated {k}/{n_samples} attempts, {len(rows)} kept")

        T0 = float(rng.uniform(*T0_range))
        Istar, Tstar = solve_runaway(T0)
        if not np.isfinite(Istar) or Istar <= 0:
            continue

        if rng.random() >= frac_unsafe:
            frac = rng.uniform(0.50, 0.99)
            I = frac * Istar
            Tsurf = stable_surface_T(I, T0, T_hi=Tstar)
            if not np.isfinite(Tsurf):
                continue
            V = tape_voltage(I, min(Tsurf, T_X_K - 1e-4))
            runaway_flag = 0
            time_margin_ms = np.nan
        else:
            frac = rng.uniform(1.01, 1.20)
            I = frac * Istar
            tm_s = time_margin_to_safe(I, Tstar)
            if not np.isfinite(tm_s):
                continue
            Tsurf = Tstar
            V = tape_voltage(I, min(Tstar, T_X_K - 1e-4))
            runaway_flag = 1
            time_margin_ms = tm_s * 1000.0

        rows.append(dict(
            T0_K=T0,
            I_A=I,
            voltage_V=V,
            T_surface_K=Tsurf,
            Istar_A=Istar,
            margin_norm=(Istar - I) / Istar,
            Tstar_K=Tstar,
            runaway=runaway_flag,
            time_margin_ms=time_margin_ms,
        ))

    df = pd.DataFrame(rows)
    if verbose:
        frac = df["runaway"].mean() if len(df) else float("nan")
        print(f"Generated {len(df)} rows; {frac:.0%} unsafe.")
    return df
