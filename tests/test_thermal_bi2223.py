"""
test_thermal_bi2223.py -- Bi-2223 adiabatic time-margin (thermal_bi2223.py).
"""

import numpy as np

from hts_tape_env import thermal_bi2223 as tb


def test_cp_ag_increases_with_temperature():
    Ts = np.array([4.0, 20.0, 77.0, 300.0])
    assert np.all(np.diff(tb.cp_ag(Ts)) > 0)


def test_rho_ag_increases_with_temperature():
    Ts = np.array([4.0, 20.0, 77.0, 300.0])
    assert np.all(np.diff(tb.rho_ag(Ts)) > 0)


def test_C_per_m_positive():
    assert tb.C_per_m(77.0) > 0.0


def test_R_ag_positive_and_physical():
    """Ag resistance should be small but positive at LN2 temperature."""
    R = tb.R_ag(77.0)
    assert R > 0.0
    assert R < 1.0


def test_time_margin_finite_for_bi2223_runaway():
    """At the Bi-2223's own runaway point (T0=77.16K), the time margin
    should be a finite positive value."""
    from hts_tape_env.physics_bi2223 import solve_runaway
    Istar, Tstar = solve_runaway(77.16)
    tm = tb.time_margin_to_safe(Istar, Tstar)
    assert np.isfinite(tm)
    assert 0.0 < tm < 10.0


def test_time_margin_nan_when_unreachable():
    tm = tb.time_margin_to_safe(I=0.1, Tstar=10.0, T_target=300.0,
                                 dt=2e-4, t_max=1e-3)
    assert np.isnan(tm)
