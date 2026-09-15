"""
plot_fem_spatial.py

Spatial diagnostics from the FEM sweep (fem_data.py) -- the thing neither
physics.py (lumped REBCO) nor physics_bi2223.py (lumped Bi-2223) can show at
all, since both assume a single (I, T) point for the whole tape. This is
the "far more spatially inhomogeneous" behavior the reference paper's
Discussion flags for REBCO and doesn't model (see physics_bi2223.py's
docstring) -- the FEM sweep is real x-resolved data for exactly that.

Three panels, comparing a stable run (r0p5) against the runaway run (r0p99):
    1. RTD temperature vs x-position at several snapshots in time -- shows
       whether/where a hot spot localizes, vs the lumped model's single T.
    2. Local voltage-tap segment drop (V_i - V_{i-1}, i.e. the actual
       per-segment resistive voltage, not the cumulative-from-V_CL+ value
       stored in V1_i/V2_i) vs x at the final recorded step -- the real
       detection signal a distributed voltage-tap array would see: which
       segment lights up first.
    3. TmaxLeft vs TmaxRight vs t for the runaway run -- the left/right
       asymmetry (fracLeft) driving which side reaches runaway.

Usage
-----
    python scripts/plot_fem_spatial.py --stable r0p5 --runaway r0p99
"""

import _bootstrap  # noqa: F401

import argparse

import numpy as np

from hts_tape_env.fem_data import DEFAULT_PATH, RTD_CHANNELS, VTAP_CHANNELS, load_fem_positions, load_fem_sweep


def _channel_x(pos_df, ratio_label, channels):
    ref = pos_df[pos_df["ratio_label"] == ratio_label].set_index("channel")
    return {c: float(ref.loc[c, "x_m"]) for c in channels if c in ref.index}


def _segment_voltage_profile(g, side_prefix, x_by_channel):
    """V_i - V_{i-1} (local segment drop) vs the segment's midpoint x, for one
    side's tap chain (V1_* or V2_*), with V_0 taken as V_CL_plus (=0 by
    construction, see fem_data.py's docstring)."""
    n_taps = 10
    chans = [f"{side_prefix}_{i}" for i in range(1, n_taps + 1)]
    x = [0.0] + [x_by_channel[c] for c in chans]  # V_CL_plus sits at x=0
    v = [0.0] + [float(g[c].iloc[-1]) for c in chans]
    seg_x = [(x[i] + x[i - 1]) / 2 for i in range(1, len(x))]
    seg_dv = [v[i] - v[i - 1] for i in range(1, len(v))]
    return np.array(seg_x), np.array(seg_dv)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stable", default="r0p5")
    ap.add_argument("--runaway", default="r0p99")
    ap.add_argument("--fem-path", default=DEFAULT_PATH)
    ap.add_argument("--plot", default="models/fem_spatial_diagnostics.png")
    args = ap.parse_args()

    df = load_fem_sweep(args.fem_path)
    pos = load_fem_positions(args.fem_path)

    rtd_x = _channel_x(pos, args.runaway, RTD_CHANNELS)  # same channel set both runs; x shifts slightly (see fem_data.py)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    # ── Panel 1: RTD temperature vs x, several time snapshots, both runs.
    # Left (RTD1_*) and right (RTD2_*) chains plotted as separate lines --
    # their x-positions interleave almost exactly, so merging them into one
    # line sorted by x produces a false zigzag rather than two smooth curves.
    rtd1_chans = [c for c in RTD_CHANNELS if c.startswith("RTD1_")]
    rtd2_chans = [c for c in RTD_CHANNELS if c.startswith("RTD2_")]
    for label, style in [(args.stable, "-"), (args.runaway, "--")]:
        g = df[df["ratio_label"] == label].sort_values("step")
        rtd_x_run = _channel_x(pos, label, RTD_CHANNELS)
        n = len(g)
        snap_idxs = sorted(set([0, n // 3, (2 * n) // 3, n - 1]))
        cmap = plt.cm.plasma(np.linspace(0.15, 0.9, len(snap_idxs)))
        for k, idx in enumerate(snap_idxs):
            row = g.iloc[idx]
            for chans, marker in [(rtd1_chans, "o"), (rtd2_chans, "^")]:
                xs = np.array([rtd_x_run[c] for c in chans])
                ys = np.array([row[c] for c in chans])
                order = np.argsort(xs)
                is_last = (k == len(snap_idxs) - 1) and (chans is rtd1_chans)
                axes[0].plot(xs[order], ys[order], style, marker=marker, ms=3,
                             color=cmap[k], alpha=0.85,
                             label=f"{label}  t={row['t_s']:.2f}s" if is_last else None)
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("RTD temperature [K]")
    axes[0].set_title("Spatial temperature profile (o=left/RTD1, ^=right/RTD2)\n"
                       "(solid=stable, dashed=runaway, darker=later)")
    axes[0].legend(fontsize=8)

    # ── Panel 2: local voltage-tap segment drop vs x, final step, both runs ─
    for label, color in [(args.stable, "tab:blue"), (args.runaway, "tab:red")]:
        g = df[df["ratio_label"] == label].sort_values("step")
        x_run = _channel_x(pos, label, VTAP_CHANNELS)
        for side, marker in [("V1", "o"), ("V2", "s")]:
            seg_x, seg_dv = _segment_voltage_profile(g, side, x_run)
            axes[1].plot(seg_x, seg_dv * 1e3, marker + "-", color=color, alpha=0.8,
                         label=f"{label} {side} (final t={g['t_s'].iloc[-1]:.2f}s)")
    axes[1].set_xlabel("segment midpoint x [m]"); axes[1].set_ylabel("local segment voltage [mV]")
    axes[1].set_title("Where the resistive voltage actually rises\n(per-tap-segment drop, not cumulative)")
    axes[1].legend(fontsize=7)

    # ── Panel 3: TmaxLeft vs TmaxRight vs t, runaway run ──────────────────
    g = df[df["ratio_label"] == args.runaway].sort_values("step")
    axes[2].plot(g["t_s"], g["TmaxLeft"], label="TmaxLeft", color="tab:blue")
    axes[2].plot(g["t_s"], g["TmaxRight"], label="TmaxRight", color="tab:red")
    ax2b = axes[2].twinx()
    ax2b.plot(g["t_s"], g["fracLeft"], label="fracLeft", color="gray", ls=":")
    ax2b.set_ylabel("fracLeft", color="gray")
    axes[2].set_xlabel("t [s]"); axes[2].set_ylabel("T [K]")
    axes[2].set_title(f"Left/right asymmetry during runaway ({args.runaway})")
    axes[2].legend(loc="upper left", fontsize=8)

    fig.suptitle("FEM spatial diagnostics -- what the lumped physics.py/physics_bi2223.py models can't see")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=130)
    print(f"Saved {args.plot}")

    # Console summary: where's the hottest RTD, and biggest voltage segment?
    for label in [args.stable, args.runaway]:
        g = df[df["ratio_label"] == label].sort_values("step")
        x_run = _channel_x(pos, label, RTD_CHANNELS)
        last = g.iloc[-1]
        hottest = max(RTD_CHANNELS, key=lambda c: last[c])
        print(f"{label}: hottest RTD at final step = {hottest} "
              f"(T={last[hottest]:.2f} K, x={x_run[hottest]:.4f} m); Tmax={last['T_surface_K']:.2f} K")


if __name__ == "__main__":
    main()
