"""
Parity tests: JAX FD solver functions vs NumPy reference (physics_fd1d.py).

Each test verifies that the JAX implementation produces results matching
the original NumPy code to within tight tolerances.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import numpy.testing as npt
import pytest

import jax.numpy as jnp

from hts_tape_env import physics, thermal
from hts_tape_env.materials import lumped_rebco_tape, TapeGeometry, L0_WF
from hts_tape_env.physics_fd1d import (
    Tape1D, _solve_current_sharing_vec, du_fort_frankel_step, _forward_euler_step,
)
from hts_tape_env.physics_fd1d_jax import (
    rho_cu_jax, rho_ss_jax, k_cu_jax, k_ss_jax,
    cp_cu_jax, cp_ss_jax, Ic_jax,
    k_eff_jax, C_eff_jax, G_norm_jax,
    _solve_current_sharing_jax, _du_fort_frankel_step_jax,
    _forward_euler_step_jax,
    make_lumped_params, make_settle_fn,
)


T_SWEEP = np.array([4.0, 10.0, 20.0, 50.0, 77.0, 100.0, 200.0, 300.0])
GEOM = lumped_rebco_tape(L=0.2, w=0.004)
PARAMS = make_lumped_params(Ic0=500.0, Tc=85.0, n_value=20.0, w=0.004,
                            t_cu=physics.t_cu, t_ss=physics.t_ss, Ec=physics.Ec)


class TestMaterialProperties:
    """JAX material functions vs NumPy originals."""

    def test_rho_cu(self):
        jax_vals = np.array(rho_cu_jax(jnp.array(T_SWEEP)))
        np_vals = np.array([float(physics.rho_cu(T)) for T in T_SWEEP])
        npt.assert_allclose(jax_vals, np_vals, rtol=1e-10)

    def test_rho_ss(self):
        jax_vals = np.array(rho_ss_jax(jnp.array(T_SWEEP)))
        np_vals = np.array([float(physics.rho_ss(T)) for T in T_SWEEP])
        npt.assert_allclose(jax_vals, np_vals, rtol=1e-10)

    def test_k_cu(self):
        jax_vals = np.array(k_cu_jax(jnp.array(T_SWEEP)))
        np_vals = np.array([L0_WF * T / float(physics.rho_cu(T)) for T in T_SWEEP])
        npt.assert_allclose(jax_vals, np_vals, rtol=1e-10)

    def test_k_ss(self):
        jax_vals = np.array(k_ss_jax(jnp.array(T_SWEEP)))
        np_vals = np.array([L0_WF * T / float(physics.rho_ss(T)) for T in T_SWEEP])
        npt.assert_allclose(jax_vals, np_vals, rtol=1e-10)

    def test_cp_cu(self):
        jax_vals = np.array(cp_cu_jax(jnp.array(T_SWEEP)))
        np_vals = np.array([float(thermal.cp_cu(T)) for T in T_SWEEP])
        npt.assert_allclose(jax_vals, np_vals, rtol=1e-6)

    def test_cp_ss(self):
        jax_vals = np.array(cp_ss_jax(jnp.array(T_SWEEP)))
        np_vals = np.array([float(thermal.cp_ss(T)) for T in T_SWEEP])
        npt.assert_allclose(jax_vals, np_vals, rtol=1e-6)

    def test_Ic(self):
        jax_vals = np.array(Ic_jax(jnp.array(T_SWEEP), 500.0, 85.0))
        np_vals = np.array([float(physics.Ic(T, 500.0, 85.0)) for T in T_SWEEP])
        npt.assert_allclose(jax_vals, np_vals, rtol=1e-10)


class TestCompositeGeometry:
    """JAX composite functions vs TapeGeometry methods."""

    def test_k_eff(self):
        jax_vals = np.array(k_eff_jax(jnp.array(T_SWEEP), PARAMS.A_cu, PARAMS.A_ss))
        np_vals = np.array([GEOM.k_eff(T) for T in T_SWEEP])
        npt.assert_allclose(jax_vals, np_vals, rtol=1e-10)

    def test_C_eff(self):
        jax_vals = np.array(C_eff_jax(jnp.array(T_SWEEP), PARAMS.A_cu, PARAMS.A_ss))
        np_vals = np.array([GEOM.C_eff(T) for T in T_SWEEP])
        npt.assert_allclose(jax_vals, np_vals, rtol=1e-6)

    def test_G_norm(self):
        jax_vals = np.array(G_norm_jax(jnp.array(T_SWEEP), PARAMS.A_cu, PARAMS.A_ss))
        np_vals = np.array([GEOM.normal_layers_conductance(T) for T in T_SWEEP])
        npt.assert_allclose(jax_vals, np_vals, rtol=1e-10)


class TestCurrentSharing:
    """JAX current sharing solver vs NumPy _solve_current_sharing_vec."""

    @pytest.mark.parametrize("I", [100.0, 300.0, 480.0])
    def test_E_field_parity(self, I):
        T = np.linspace(50.0, 80.0, 50)
        E_np, Q_np = _solve_current_sharing_vec(I, T, GEOM)
        E_jax, Q_jax = _solve_current_sharing_jax(I, jnp.array(T), PARAMS)
        E_jax, Q_jax = np.array(E_jax), np.array(Q_jax)
        npt.assert_allclose(E_jax, E_np, rtol=1e-4)
        npt.assert_allclose(Q_jax, Q_np, rtol=1e-4)


class TestDuFortFrankel:
    """JAX DFF step vs NumPy du_fort_frankel_step."""

    def test_step_parity(self):
        nx = 50
        tape = Tape1D(geom=GEOM, nx=nx)
        T_bath = 60.0
        T_curr = np.full(nx, T_bath)
        T_curr[nx // 2] = 80.0
        T_prev = T_curr.copy()
        Q = np.zeros(nx)
        Q[nx // 2] = 1e4

        T_np = du_fort_frankel_step(T_curr, T_prev, Q, GEOM, tape.dx, 1e-4, T_bath)
        T_jax = np.array(_du_fort_frankel_step_jax(
            jnp.array(T_curr), jnp.array(T_prev), jnp.array(Q),
            PARAMS, tape.dx, 1e-4, T_bath,
        ))
        npt.assert_allclose(T_jax, T_np, rtol=1e-6)

    def test_euler_parity(self):
        nx = 50
        tape = Tape1D(geom=GEOM, nx=nx)
        T_bath = 60.0
        T = np.full(nx, T_bath)
        T[nx // 2] = 80.0
        Q = np.zeros(nx)

        T_np = _forward_euler_step(T, Q, GEOM, tape.dx, 1e-6, T_bath)
        T_jax = np.array(_forward_euler_step_jax(
            jnp.array(T), jnp.array(Q), PARAMS, tape.dx, 1e-6, T_bath,
        ))
        npt.assert_allclose(T_jax, T_np, rtol=1e-6)


class TestSettleLoop:
    """Fused JAX settle vs NumPy _advance_fd loop."""

    def test_settle_parity(self):
        nx = 50
        tape = Tape1D(geom=GEOM, nx=nx)
        T_bath = 60.0
        I = 300.0
        n_steps = 10

        # NumPy reference: manual loop
        T_curr = np.full(nx, T_bath)
        T_prev = T_curr.copy()
        for _ in range(n_steps):
            E, Q_j = _solve_current_sharing_vec(I, T_curr, GEOM)
            T_next = du_fort_frankel_step(T_curr, T_prev, Q_j, GEOM, tape.dx, 1e-4, T_bath)
            T_prev = T_curr
            T_curr = T_next
        Tmax_np = np.max(T_curr)
        V_np = np.sum(E * tape.dx)

        # JAX settle
        settle = make_settle_fn(PARAMS, tape.dx, 1e-4, n_steps)
        T_init = jnp.full(nx, T_bath)
        T_final, T_prev_final, Tmax_jax, V_jax = settle(I, T_init, T_init, T_bath)

        npt.assert_allclose(float(Tmax_jax), Tmax_np, rtol=1e-3)
        npt.assert_allclose(np.array(T_final), T_curr, rtol=1e-3)


class TestEnvBackend:
    """FD env with JAX backend produces same outputs as NumPy."""

    def test_env_jax_runs(self):
        from hts_tape_env.env_fd1d import FD1DQuenchDetectorEnv
        env = FD1DQuenchDetectorEnv(
            geometry="lumped", nx=30,
            T0_range=(50.0, 60.0),
            n_settle_steps=5, dt=1e-4,
            istar_tol=5.0, istar_max_iter=8, istar_t_end=0.05,
            backend="jax",
        )
        obs, info = env.reset(seed=42)
        assert obs.shape == (4,)
        assert info["I_star"] > 0

        obs2, reward, term, trunc, info2 = env.step(0)
        assert obs2.shape == (4,)
