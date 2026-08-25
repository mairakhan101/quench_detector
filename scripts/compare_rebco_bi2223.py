"""
compare_rebco_bi2223.py

Side-by-side runaway-boundary comparison between the two conductors in
Marchevsky & Prestemon 2024: the theoretical REBCO model (physics.py,
mode='no_stab' -- the fair comparison, since physics_bi2223.py has no
stabilizer/current-redistribution model either, see its module docstring)
and the experimentally-characterized Bi-2223 tape (physics_bi2223.py).

These are NOT swept over the same T0 range: physics_bi2223's Ic(T)/n(T) are
only characterized near LN2 temperature (T_ANCHOR_K=77.16 .. T_X_K=91.35, see
its docstring) -- extrapolating that fit down to 5 K the way the REBCO model
is routinely swept would be unsupported by any data. Each conductor is swept
over its own physically-meaningful range.

Usage
-----
    python scripts/compare_rebco_bi2223.py
"""

import _bootstrap  # noqa: F401

import numpy as np

from hts_tape_env import physics as rebco
from hts_tape_env import physics_bi2223 as bi2223


def main():
    print("REBCO (physics.py, no_stab, n=20, Ic0=500A, Tc=85K):")
    print(f"{'T0_K':<8}{'I*_A':<10}{'T*_K':<10}{'I*/Ic(T0)'}")
    for T0 in [5, 20, 40, 60, 75, 80]:
        Istar, Tstar = rebco.solve_runaway(T0, n=20.0, mode="no_stab")
        ic_t0 = rebco.Ic(T0)
        ratio = Istar / ic_t0 if np.isfinite(Istar) else float("nan")
        print(f"{T0:<8}{Istar:<10.1f}{Tstar:<10.2f}{ratio:.3f}")

    print(f"\nBi-2223 (physics_bi2223.py, measured Ic(T)/n(T)/eta_c={bi2223.ETA_DEFAULT_W_PER_K} W/K):")
    print("(solve_runaway() is only well-behaved up to ~80-81K -- see physics_bi2223.py docstring)")
    print(f"{'T0_K':<8}{'I*_A':<10}{'T*_K':<10}{'I*/Ic(T0)'}")
    for T0 in [77.16, 78, 79, 79.5, 80, 80.5]:
        Istar, Tstar = bi2223.solve_runaway(T0)
        ic_t0 = bi2223.Ic(T0)
        ratio = Istar / ic_t0 if np.isfinite(Istar) else float("nan")
        print(f"{T0:<8.2f}{Istar:<10.1f}{Tstar:<10.2f}{ratio:.3f}")

    print("\nPaper cross-check (Sec. 3, p8): measured/simulated Bi-2223 runaway at T0=77.16-77.2K "
          "is ~260 A. This reconstruction (approximate Ic(T)/n(T) anchors, see physics_bi2223.py "
          "docstring) gives:")
    Istar, Tstar = bi2223.solve_runaway(77.16)
    print(f"  I* = {Istar:.1f} A ({(Istar/260.0 - 1)*100:+.1f}% vs paper's ~260 A), T* = {Tstar:.2f} K")

    try:
        import matplotlib.pyplot as plt

        T0_rebco = np.linspace(5, 80, 60)
        I_rebco = np.array([rebco.solve_runaway(T0, n=20.0, mode="no_stab")[0] for T0 in T0_rebco])

        T0_bi = np.linspace(bi2223.T_ANCHOR_K, 80.5, 40)  # trustworthy range only -- see module docstring
        I_bi = np.array([bi2223.solve_runaway(T0)[0] for T0 in T0_bi])

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].plot(T0_rebco, I_rebco, color="tab:blue")
        axes[0].set_title("REBCO (physics.py, no_stab)")
        axes[0].set_xlabel("T0 [K]"); axes[0].set_ylabel("I* [A]")

        axes[1].plot(T0_bi, I_bi, color="tab:orange")
        axes[1].set_title("Bi-2223 (physics_bi2223.py, measured Ic(T)/n(T))")
        axes[1].set_xlabel("T0 [K]"); axes[1].set_ylabel("I* [A]")

        fig.suptitle("Runaway current I*(T0) -- separate conductors, separate validated ranges")
        fig.tight_layout()
        fig.savefig("models/rebco_vs_bi2223_runaway.png", dpi=130)
        print("\nSaved models/rebco_vs_bi2223_runaway.png")
    except ImportError:
        print("\n(matplotlib not available -- skipped plot)")


if __name__ == "__main__":
    main()
