"""Tests for the 1D Du Fort-Frankel FD solver."""

import numpy as np
import pytest

from hts_tape_env.materials import lumped_rebco_tape, TapeGeometry
from hts_tape_env.physics_fd1d import (
    Tape1D, HeaterPulse, CoolingModel,
    solve_current_sharing_1d,
    _solve_current_sharing_vec,
    du_fort_frankel_step,
    _forward_euler_step,
    run_transient,
    find_Istar_fd,
)


@pytest.fixture
def tape():
    geom = lumped_rebco_tape(L=1.0, w=0.004)
    return Tape1D(geom=geom, nx=100)


@pytest.fixture
def short_tape():
    geom = lumped_rebco_tape(L=0.2, w=0.004)
    return Tape1D(geom=geom, nx=100)


class TestCurrentSharing:
    def test_zero_current(self, tape):
        T = np.full(tape.nx, 77.0)
        E, Q = solve_current_sharing_1d(0.0, T, tape.geom)
        assert np.allclose(E, 0.0)
        assert np.allclose(Q, 0.0)

    def test_positive_current(self, tape):
        T = np.full(tape.nx, 77.0)
        E, Q = solve_current_sharing_1d(200.0, T, tape.geom)
        assert np.all(E > 0)
        assert np.all(Q > 0)

    def test_vec_matches_scalar(self, tape):
        T = np.linspace(50.0, 80.0, tape.nx)
        E_s, Q_s = solve_current_sharing_1d(200.0, T, tape.geom)
        E_v, Q_v = _solve_current_sharing_vec(200.0, T, tape.geom)
        np.testing.assert_allclose(E_s, E_v, rtol=1e-3)
        np.testing.assert_allclose(Q_s, Q_v, rtol=1e-3)

    def test_higher_current_higher_E(self, tape):
        T = np.full(tape.nx, 77.0)
        E1, _ = solve_current_sharing_1d(100.0, T, tape.geom)
        E2, _ = solve_current_sharing_1d(300.0, T, tape.geom)
        assert np.all(E2 > E1)

    def test_higher_T_higher_E(self, tape):
        """At higher T, Ic drops so E must rise to maintain the same current."""
        T_lo = np.full(tape.nx, 40.0)
        T_hi = np.full(tape.nx, 75.0)
        E_lo, _ = solve_current_sharing_1d(200.0, T_lo, tape.geom)
        E_hi, _ = solve_current_sharing_1d(200.0, T_hi, tape.geom)
        assert np.mean(E_hi) > np.mean(E_lo)


class TestDuFortFrankel:
    def test_uniform_stays_uniform(self, tape):
        """Uniform temperature field with no sources should stay uniform."""
        T = np.full(tape.nx, 77.0)
        T_prev = T.copy()
        Q = np.zeros(tape.nx)
        T_next = du_fort_frankel_step(T, T_prev, Q, tape.geom, tape.dx, 1e-4, 77.0)
        np.testing.assert_allclose(T_next, 77.0, atol=1e-10)

    def test_hot_spot_diffuses(self, tape):
        """A hot spot should spread out (max decreases) under pure diffusion."""
        T = np.full(tape.nx, 77.0)
        mid = tape.nx // 2
        T[mid] = 90.0
        T_prev = T.copy()
        Q = np.zeros(tape.nx)

        for _ in range(100):
            T_next = du_fort_frankel_step(T, T_prev, Q, tape.geom, tape.dx, 1e-5, 77.0)
            T_prev = T
            T = T_next

        assert T[mid] < 90.0, "Hot spot should have cooled"
        assert T[mid] > 77.0, "Hot spot shouldn't have vanished completely"

    def test_symmetry(self):
        """Symmetric initial condition should produce symmetric result."""
        geom = lumped_rebco_tape(L=1.0, w=0.004)
        tape = Tape1D(geom=geom, nx=101)  # odd grid for clean center
        nx = tape.nx
        T = np.full(nx, 77.0)
        mid = nx // 2  # = 50
        T[mid] = 90.0
        T[mid - 1] = 85.0
        T[mid + 1] = 85.0
        T_prev = T.copy()
        Q = np.zeros(nx)

        for _ in range(50):
            T_next = du_fort_frankel_step(T, T_prev, Q, tape.geom, tape.dx, 1e-5, 77.0)
            T_prev = T
            T = T_next

        # Compare left half (0..mid-1) vs reversed right half (nx-1..mid+1)
        left = T[:mid]
        right = T[nx - 1:mid:-1]
        np.testing.assert_allclose(left, right, atol=0.01)


class TestTransient:
    def test_no_current_no_heater(self, tape):
        """Zero current, no heater → temperature stays at bath."""
        result = run_transient(tape, 0.0, 77.0, heater=None, dt=1e-4, t_end=0.01)
        np.testing.assert_allclose(result["Tmax"], 77.0, atol=0.01)
        assert not result["runaway"]

    def test_heater_raises_temperature(self, tape):
        heater = HeaterPulse(power=50.0, duration=0.005)
        result = run_transient(tape, 0.0, 77.0, heater=heater, dt=1e-4, t_end=0.02)
        assert result["Tmax"][-1] > 77.0

    def test_recovery_at_low_current(self, short_tape):
        """Low current with heater should recover."""
        heater = HeaterPulse(power=13.0, duration=0.01)
        cool = CoolingModel(mode="bath_ends", T_bath=77.0)
        result = run_transient(short_tape, 50.0, 77.0, heater=heater, cool=cool,
                               dt=5e-5, t_end=0.1)
        assert not result["runaway"]

    def test_runaway_at_high_current(self, short_tape):
        """Near-Ic current with heater should run away."""
        Ic77 = short_tape.geom.sc.Ic_func(77.0)
        heater = HeaterPulse(power=50.0, duration=0.02)
        cool = CoolingModel(mode="bath_ends", T_bath=77.0)
        result = run_transient(short_tape, Ic77 * 0.95, 77.0, heater=heater, cool=cool,
                               dt=5e-5, t_end=2.0, T_runaway=300.0)
        assert result["runaway"]


class TestIstarFinder:
    def test_Istar_is_finite(self, short_tape):
        """Should find a finite I* at 77K."""
        heater = HeaterPulse(power=13.0, duration=0.01)
        cool = CoolingModel(mode="bath_ends", T_bath=77.0)
        Istar, Tstar = find_Istar_fd(short_tape, 77.0, heater=heater, cool=cool,
                                      tol=5.0, max_iter=10, t_end=0.1, dt=5e-5)
        assert np.isfinite(Istar)
        assert Istar > 50.0
        Ic77 = short_tape.geom.sc.Ic_func(77.0)
        # With bath-end cooling, I* can exceed Ic (heat conducts to cooled ends)
        assert Istar < Ic77 * 5.0

    def test_Istar_decreases_with_temperature(self, short_tape):
        """I* should be lower at higher bath temperature."""
        heater = HeaterPulse(power=13.0, duration=0.01)
        cool = CoolingModel(mode="bath_ends", T_bath=40.0)
        I1, _ = find_Istar_fd(short_tape, 40.0, heater=heater, cool=cool,
                               tol=10.0, max_iter=8, t_end=0.1, dt=5e-5)

        cool2 = CoolingModel(mode="bath_ends", T_bath=70.0)
        I2, _ = find_Istar_fd(short_tape, 70.0, heater=heater, cool=cool2,
                               tol=10.0, max_iter=8, t_end=0.1, dt=5e-5)

        assert I1 > I2, f"I*(40K)={I1:.1f} should be > I*(70K)={I2:.1f}"
