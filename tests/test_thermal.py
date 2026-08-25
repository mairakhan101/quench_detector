"""
test_thermal.py -- adiabatic time-margin / heat capacity (thermal.py).
"""

import numpy as np

from hts_tape_env import thermal


def test_cp_increases_with_temperature():
    Ts = np.array([4.0, 20.0, 77.0, 300.0])
    assert np.all(np.diff(thermal.cp_cu(Ts)) > 0)
    assert np.all(np.diff(thermal.cp_ss(Ts)) > 0)


def test_C_per_m_positive():
    assert thermal.C_per_m(77.0) > 0.0


def test_time_margin_finite_for_a_hot_runaway_point():
    """A conductor already at a high runaway temperature with substantial
    current should reach T_target well within t_max."""
    tm = thermal.time_margin_to_safe(I=600.0, Tstar=60.0, T_target=300.0, dt=2e-4, t_max=1.0)
    assert np.isfinite(tm)
    assert 0.0 < tm < 1.0


def test_time_margin_nan_when_unreachable():
    """A tiny current / tiny t_max shouldn't reach T_target -- should report
    NaN, not silently return t_max or crash."""
    tm = thermal.time_margin_to_safe(I=1.0, Tstar=10.0, T_target=300.0, dt=2e-4, t_max=1e-3)
    assert np.isnan(tm)
