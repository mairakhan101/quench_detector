"""
test_physics_bi2223.py -- Bi-2223 analytic solver (physics_bi2223.py).
"""

import numpy as np
import pytest

from hts_tape_env import physics_bi2223 as bi


def test_Ic_zero_at_Tx():
    assert bi.Ic(bi.T_X_K) == pytest.approx(0.0, abs=1e-6)


def test_Ic_monotonic_decreasing_in_valid_range():
    Ts = np.linspace(bi.T_ANCHOR_K, bi.T_X_K, 30)
    Ic = bi.Ic(Ts)
    assert np.all(np.diff(Ic) <= 1e-9)


def test_n_value_floored_at_one_and_decreasing():
    Ts = np.linspace(bi.T_ANCHOR_K, bi.T_X_K, 30)
    n = bi.n_value(Ts)
    assert np.all(n >= 1.0)
    assert n[0] > n[-1]  # n decreases toward T_x, per fig 6(e)


def test_solve_runaway_matches_paper_within_reconstruction_tolerance():
    """Cross-check vs the paper's own reported ~260 A runaway at T0~77.16 K
    (Sec 3, p8). The Ic0/n(T) anchors are an approximate reconstruction (see
    physics_bi2223.py's docstring) -- 15% is the documented tolerance, not
    an exact-match expectation."""
    Istar, Tstar = bi.solve_runaway(bi.T_ANCHOR_K)
    assert np.isfinite(Istar)
    assert Istar == pytest.approx(260.0, rel=0.15)
    assert bi.T_ANCHOR_K < Tstar < bi.T_X_K


def test_solve_runaway_decreases_with_T0_in_validated_range():
    """I*(T0) should fall smoothly as T0 rises -- the physically expected
    trend (less thermal margin left before Ic(T0) itself is used up)."""
    T0s = np.linspace(bi.T_ANCHOR_K, 80.5, 8)
    Istars = [bi.solve_runaway(T0)[0] for T0 in T0s]
    assert all(np.isfinite(I) for I in Istars)
    assert np.all(np.diff(Istars) < 0)


def test_solve_runaway_guards_against_reconstruction_artifact():
    """Above ~81 K the crude n(T) anchors degenerate near T_x and produce a
    nonphysical current spike (see solve_runaway's docstring) -- this should
    be caught by the sanity cap and reported as NaN, not returned as a huge
    number a caller might mistake for a real runaway current."""
    Istar, Tstar = bi.solve_runaway(85.0)
    assert np.isnan(Istar) and np.isnan(Tstar)


def test_tape_voltage_nonnegative():
    T0 = bi.T_ANCHOR_K
    Istar, Tstar = bi.solve_runaway(T0)
    Tsurf = bi.stable_surface_T(0.8 * Istar, T0, T_hi=Tstar)
    assert np.isfinite(Tsurf)
    V = bi.tape_voltage(0.8 * Istar, Tsurf)
    assert V >= 0.0
