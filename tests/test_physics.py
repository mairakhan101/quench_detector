"""
test_physics.py -- REBCO analytic solver (physics.py).
"""

import numpy as np
import pytest

from hts_tape_env import physics


def test_Ic_monotonic_decreasing_with_T():
    Ts = np.linspace(1.0, 84.0, 50)
    Ic = physics.Ic(Ts)
    assert np.all(np.diff(Ic) <= 0), "Ic(T) should be non-increasing in T"


def test_Ic_zero_at_Tc():
    assert physics.Ic(physics.Tc - 1e-6) < 1.0  # near-zero, per Ic0*(1-(T/Tc)^1.7)


def test_solve_runaway_matches_known_notebook_point():
    """Regression check from docs/PROGRESS.md's original validation: at
    T0=59 K, lambda=99 W/m/K, solve_runaway should give I* ~= 435 A."""
    Istar, Tstar = physics.solve_runaway(59.0, 20.0, mode="stab", lam_override=99.0)
    assert np.isfinite(Istar)
    assert Istar == pytest.approx(435.0, rel=0.05)
    assert 59.0 < Tstar < physics.Tc


def test_solve_runaway_no_stab_below_stab():
    """No-stabilizer conductor should runaway at a lower (or equal) current
    than the same conductor with a Cu/SS stabilizer sharing current, at the
    same operating point -- the stabilizer's whole purpose is to raise I*."""
    T0, lam = 40.0, 80.0
    Istar_stab, _ = physics.solve_runaway(T0, 20.0, mode="stab", lam_override=lam)
    Istar_no_stab, _ = physics.solve_runaway(T0, 20.0, mode="no_stab", lam_override=lam)
    assert np.isfinite(Istar_stab) and np.isfinite(Istar_no_stab)
    assert Istar_stab >= Istar_no_stab


def test_stable_surface_T_below_runaway_current():
    """Below I*, stable_surface_T (given the T_hi=T* hint -- see its
    docstring's bug note) should find a root close to T0, not diverge."""
    T0, lam = 40.0, 80.0
    Istar, Tstar = physics.solve_runaway(T0, 20.0, mode="stab", lam_override=lam)
    I = 0.9 * Istar
    Tsurf = physics.stable_surface_T(I, T0, 20.0, mode="stab", lam_override=lam, T_hi=Tstar)
    assert np.isfinite(Tsurf)
    assert T0 < Tsurf < Tstar


def test_stable_surface_T_is_a_heat_balance_root():
    """The returned T should actually satisfy Qin(I,T) ~= Qout(T,T0) --
    i.e. stable_surface_T is solving the equation its docstring claims."""
    T0, lam = 30.0, 100.0
    Istar, Tstar = physics.solve_runaway(T0, 20.0, mode="stab", lam_override=lam)
    I = 0.9 * Istar
    Tsurf = physics.stable_surface_T(I, T0, 20.0, mode="stab", lam_override=lam, T_hi=Tstar)
    assert np.isfinite(Tsurf)
    qin = physics.Qin_stab(I, Tsurf, 20.0)
    qout = physics.Qout(Tsurf, T0, lam_override=lam)
    assert qin == pytest.approx(qout, rel=1e-3, abs=1e-6)


def test_tape_voltage_nonnegative_and_increases_with_current():
    T0, lam = 40.0, 80.0
    Istar, Tstar = physics.solve_runaway(T0, 20.0, mode="stab", lam_override=lam)
    Vs = []
    for frac in (0.3, 0.6, 0.9):
        I = frac * Istar
        T = physics.stable_surface_T(I, T0, 20.0, mode="stab", lam_override=lam, T_hi=Tstar)
        V = physics.tape_voltage(I, T if np.isfinite(T) else T0, 20.0, mode="stab")
        assert V >= 0.0
        Vs.append(V)
    assert Vs[0] < Vs[1] < Vs[2]


def test_stable_surface_T_Tstar_bound_fixes_bracket_bug():
    """Regression test for the bracket bug fixed in stable_surface_T (see its
    docstring): bounding brentq's search at Tc instead of T* leaves it unable
    to bracket either root ~40% of the time across a realistic (T0, lambda,
    I/I*) sweep in mode='stab', because Qin(I,T) diverges (not -> 0) as
    T -> Tc, so f(T0) and f(Tc-eps) often share the same sign. Passing the
    known T* as T_hi fixes the sign mismatch and should drop that well
    below 10%."""
    rng = np.random.default_rng(0)
    n_bad_old, n_bad_new, n_total = 0, 0, 0
    for _ in range(200):
        T0 = float(rng.uniform(5, 75))
        lam = float(rng.uniform(20, 200))
        Istar, Tstar = physics.solve_runaway(T0, 20.0, mode="stab", lam_override=lam)
        if not np.isfinite(Istar):
            continue
        I = float(rng.uniform(0.5, 0.99)) * Istar
        T_old = physics.stable_surface_T(I, T0, 20.0, mode="stab", lam_override=lam)
        T_new = physics.stable_surface_T(I, T0, 20.0, mode="stab", lam_override=lam, T_hi=Tstar)
        n_total += 1
        n_bad_old += int(not np.isfinite(T_old))
        n_bad_new += int(not np.isfinite(T_new))
    assert n_total > 50, "sanity: enough valid (T0, lambda) draws to make this test meaningful"
    assert n_bad_old / n_total > 0.25, "expected the pre-fix (T_hi=Tc) bracket to fail often -- if this fails, the underlying Qin(T->Tc) behavior changed"
    assert n_bad_new / n_total < 0.10, "T_hi=T* should resolve the large majority of those brentq bracket failures"


def test_solve_runaway_nan_when_no_interior_max():
    """At a base temperature right at Tc, there's no room for a stable
    branch -- should report (nan, nan), not crash or return nonsense."""
    Istar, Tstar = physics.solve_runaway(physics.Tc - 1e-3, 20.0, mode="stab", lam_override=80.0)
    assert np.isnan(Istar) and np.isnan(Tstar)
