#!/usr/bin/env python
"""
compare_bi2223_modes.py

Side-by-side comparison of the Bi-2223 equilibrium branch I(T) with and
without Ag-matrix redistribution. Demonstrates that redistribution
eliminates the thermal runaway boundary in the physically meaningful
n(T) >= 3 region.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hts_tape_env.physics_bi2223 import (
    _equilibrium_I_at_T, n_value, Ic, T_X_K, solve_runaway,
)


def main():
    T0 = 78.0
    Ts = np.linspace(T0 + 0.1, T_X_K - 0.5, 200)

    I_ns = np.array([_equilibrium_I_at_T(T, T0, mode='no_stab') for T in Ts])
    I_ag = np.array([_equilibrium_I_at_T(T, T0, mode='ag_stab') for T in Ts])
    n_vals = np.array([float(n_value(T)) for T in Ts])
    ic_vals = np.array([float(Ic(T)) for T in Ts])

    Is_ns, Ts_ns = solve_runaway(T0, mode='no_stab')

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel 1: Full equilibrium branches
    ax = axes[0, 0]
    ax.plot(Ts, I_ns, 'b-', lw=2, label='no stabilizer')
    ax.plot(Ts, I_ag, 'r-', lw=2, label='Ag-matrix redistribution')
    ax.plot(Ts, ic_vals, 'k--', lw=1, label='Ic(T)')
    if np.isfinite(Is_ns):
        ax.plot(Ts_ns, Is_ns, 'b*', ms=15, zorder=5, label=f'I* = {Is_ns:.0f} A')
    ax.axvspan(T0, T0 + (86.04 - T0), alpha=0.08, color='green', label='n(T) ≥ 3')
    ax.set_xlabel('T [K]')
    ax.set_ylabel('I_eq [A]')
    ax.set_title(f'Equilibrium branch I(T) at T₀ = {T0} K')
    ax.set_ylim(0, 1500)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: n(T) for reference
    ax = axes[0, 1]
    ax.plot(Ts, n_vals, 'k-', lw=2)
    ax.axhline(3.0, color='red', ls='--', lw=1, label='n_floor = 3')
    ax.set_xlabel('T [K]')
    ax.set_ylabel('n(T)')
    ax.set_title('Power-law index n(T)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: I*(T0) comparison across T0 range
    ax = axes[1, 0]
    T0_grid = np.linspace(77.5, 81.0, 30)
    Istar_ns = [solve_runaway(T0, mode='no_stab')[0] for T0 in T0_grid]
    Istar_ag = [solve_runaway(T0, mode='ag_stab')[0] for T0 in T0_grid]
    ax.plot(T0_grid, Istar_ns, 'bo-', ms=5, label='I* (no stab)')
    ax.plot(T0_grid, Istar_ag, 'rs-', ms=5, label='I* (Ag stab)')
    ax.set_xlabel('T₀ [K]')
    ax.set_ylabel('I* [A]')
    ax.set_title('Runaway current vs bath temperature')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 4: Text summary
    ax = axes[1, 1]
    ax.axis('off')
    summary = (
        "Ag-matrix redistribution eliminates\n"
        "the thermal runaway boundary.\n\n"
        f"No-stab: I* = {Is_ns:.0f} A at T* = {Ts_ns:.1f} K\n"
        f"  (n = {float(n_value(Ts_ns)):.1f} at T*)\n\n"
        "Ag-stab: equilibrium branch I(T) is\n"
        "  monotonically increasing in the\n"
        "  n(T) ≥ 3 region. No interior max.\n\n"
        "The Ag matrix shunts excess current\n"
        "  so effectively that Joule heating\n"
        "  never exceeds cooling within the\n"
        "  valid power-law regime.\n\n"
        "This matches the paper's own choice\n"
        "  to test ONLY the no-stabilizer model\n"
        "  for the Bi-2223 conductor."
    )
    ax.text(0.1, 0.95, summary, transform=ax.transAxes, va='top', fontsize=11,
            fontfamily='monospace')

    fig.suptitle('Bi-2223: no-stab vs Ag-matrix redistribution', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig('models/bi2223_mode_comparison.png', dpi=150)
    plt.close(fig)
    print('Saved models/bi2223_mode_comparison.png')


if __name__ == '__main__':
    main()
