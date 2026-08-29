"""Tests for FD1d ground-truth solver dispatch in benchmark.py."""

import numpy as np
import pytest

from hts_tape_env.benchmark import solve_ground_truth


class TestSolveGroundTruth:
    def test_fd1d_returns_physical_values(self):
        Is, Ts = solve_ground_truth(
            50.0, "fd1d", {"nx": 20, "tol": 5.0, "max_iter": 8}
        )
        assert np.isfinite(Is) and Is > 100
        assert np.isfinite(Ts) and Ts >= 50.0

    def test_rebco_lumped_returns_physical_values(self):
        Is, Ts = solve_ground_truth(
            50.0, "rebco", {"n": 20, "mode": "no_stab", "lam_override": 100.0}
        )
        assert np.isfinite(Is) and Is > 50
        assert np.isfinite(Ts) and Ts > 50.0

    def test_bi2223_returns_physical_values(self):
        Is, Ts = solve_ground_truth(80.0, "bi2223", {"mode": "no_stab"})
        assert np.isfinite(Is) and Is > 0
        assert np.isfinite(Ts) and Ts > 80.0

    def test_fd1d_vs_lumped_differ(self):
        T0 = 50.0
        Is_fd, _ = solve_ground_truth(
            T0, "fd1d", {"nx": 20, "tol": 5.0, "max_iter": 8}
        )
        Is_lumped, _ = solve_ground_truth(
            T0, "rebco", {"n": 20, "mode": "no_stab", "lam_override": 100.0}
        )
        rel_diff = abs(Is_fd - Is_lumped) / max(Is_fd, Is_lumped)
        assert rel_diff > 0.05

    def test_dispatch_unknown_solver_uses_rebco(self):
        Is, Ts = solve_ground_truth(
            50.0, "rebco", {"n": 20, "mode": "no_stab", "lam_override": 100.0}
        )
        assert np.isfinite(Is)
