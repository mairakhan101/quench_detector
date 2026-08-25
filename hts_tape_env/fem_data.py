"""
fem_data.py

Loader for the 3D-slit electrothermal FEM sweep at
//eagle/fanc_ldrd/magnets_fem/sweep.h5 (produced by
run_transient.py/sweep_transient.py, per its `source` root attribute).

The sweep has 6 runs (HDF5 groups r0p5 ... r0p99), each a transient solve at a
fixed *current-sharing ratio* (a local resistive-split fraction between the
tape's two parallel current paths, `left`/`right`) while the transport
current I0 is ramped to a plateau. Low ratios settle ("reached_end_deviated");
r0p95/r0p99 thermally run away ("runaway") -- see each group's `status`
attribute. This is a genuinely different sweep axis than physics.py's (T0,
lambda) -> I* boundary: there is no swept transport current *at fixed
geometry* here, so there is no FEM analog of Istar_A/Tstar_K/margin_norm/lam.
Those columns are kept (as NaN) purely so code written against
rollout.py's schema doesn't break on missing keys -- use `ratio`/`status`/
`runaway` for FEM episode outcome instead.

Each run also carries 40 real instrument channels with physical positions
(see load_fem_positions): 16 RTDs (8/side, surface-mounted), 20 voltage taps
(10/side, cumulative from V_CL_plus), 2 current-lead voltages, and 2 Hall
probes that are UNWIRED PLACEHOLDERS (constant -999 -- see HALL_CHANNELS).

Typical use
-----------
    from hts_tape_env.fem_data import load_fem_sweep, load_fem_positions
    df  = load_fem_sweep("/eagle/fanc_ldrd/magnets_fem/sweep.h5")
    pos = load_fem_positions("/eagle/fanc_ldrd/magnets_fem/sweep.h5")
    df.to_parquet("data/fem_sweep.parquet")
    pos.to_parquet("data/fem_positions.parquet")
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd

DEFAULT_PATH = "/eagle/fanc_ldrd/magnets_fem/sweep.h5"

# Quench/damage threshold [K] for locating a "runaway onset" step in Tmax(t).
# Matches thermal.time_margin_to_safe's default T_target -- the same
# "conductor reaches 300 K" convention already used elsewhere in this repo
# for the unsafe/damage boundary, not an FEM-specific choice.
QUENCH_T_K = 300.0

RTD_CHANNELS = [f"RTD1_{i}" for i in range(1, 9)] + [f"RTD2_{i}" for i in range(1, 9)]
VTAP_CHANNELS = [f"V1_{i}" for i in range(1, 11)] + [f"V2_{i}" for i in range(1, 11)]
HALL_CHANNELS = ["H1", "H2"]  # placeholders -- not yet wired to a B-field computation
LEAD_CHANNELS = ["V_CL_plus", "V_CL_minus"]
SENSOR_CHANNELS = RTD_CHANNELS + VTAP_CHANNELS + HALL_CHANNELS + LEAD_CHANNELS

# Columns kept for shape-compatibility with rollout.py's schema but not
# physically defined for an FEM ratio-sweep run (see module docstring).
_UNDEFINED_FOR_FEM = ["lam", "Istar_A", "Tstar_K", "margin_norm", "time_margin_ms"]


def _decode(arr) -> list[str]:
    """h5py object-dtype string arrays come back as bytes; decode to str."""
    return [v.decode() if isinstance(v, bytes) else str(v) for v in arr]


def load_fem_positions(path: str = DEFAULT_PATH) -> pd.DataFrame:
    """
    Sensor position/metadata table, one row per (ratio_label, channel).

    Positions are NOT identical across ratio groups (the mesh differs per
    run), so this is kept per-group rather than deduplicated -- join on
    ["ratio_label", "channel"] against load_fem_sweep()'s wide sensor columns
    if you need per-channel (x, y, z) for a given run.
    """
    rows = []
    with h5py.File(path, "r") as f:
        for label in f.keys():
            pos = f[label]["positions"]
            channel = _decode(pos["channel"][:])
            side = _decode(pos["side"][:])
            typ = _decode(pos["type"][:])
            notes = _decode(pos["notes"][:])
            x, y, z = pos["x_m"][:], pos["y_m"][:], pos["z_m"][:]
            for i in range(len(channel)):
                rows.append(dict(
                    ratio_label=label, channel=channel[i], type=typ[i], side=side[i],
                    x_m=float(x[i]), y_m=float(y[i]), z_m=float(z[i]), notes=notes[i],
                ))
    return pd.DataFrame(rows)


def load_fem_sweep(path: str = DEFAULT_PATH) -> pd.DataFrame:
    """
    Long-format telemetry table, one row per (ratio_label, timestep), columns
    matching rollout.py's names where a physical analog exists:

        episode_id      int    0..5, in ascending `ratio` order
        ratio_label     str    HDF5 group name, e.g. "r0p5"
        ratio           float  swept current-sharing ratio (0.5 .. 0.99)
        step            int    0-indexed step within the run
        t_s             float  simulation time [s]
        T0_K            float  bath temperature [K] (= Tmax at t=0, constant per episode)
        I_A, I_true_A   float  transport current I0 [A] (no noise model applied
                        here, so I_true_A duplicates I_A -- matches
                        seq_data.TRUE_FEATURES' column name for direct reuse
                        of a QuenchDetectorEnv-trained sequence surrogate)
        T_surface_K, T_surface_true_K   float  Tmax(t) [K] -- the hottest point in
                        the model, the closest FEM analog of the lumped env's
                        single surface-temperature channel (no noise model
                        applied here, so *_true duplicates the nominal value,
                        matching rollout.py's column pair for drop-in reuse)
        voltage_V, voltage_true_V       float  V_CL_minus - V_CL_plus [V], the
                        whole-tape voltage -- the FEM analog of tape_voltage()
        runaway         int    1 if this episode's status == "runaway"
        status          str    FEM solver's own outcome label
        quench_step     float  first step index where Tmax >= QUENCH_T_K, else NaN
        lam, Istar_A, Tstar_K, margin_norm, time_margin_ms
                        NaN -- not physically defined for this sweep, kept
                        only for schema compatibility (see module docstring)

    Plus every raw sensor channel in SENSOR_CHANNELS (RTD1_1..RTD2_8,
    V1_1..V2_10, H1/H2, V_CL_plus/V_CL_minus) and fracLeft/TmaxLeft/TmaxRight,
    for spatial/multi-sensor use (see fem_env.py).
    """
    rows_by_episode = []
    with h5py.File(path, "r") as f:
        labels = sorted(f.keys(), key=lambda k: float(f[k].attrs["ratio"]))
        for episode_id, label in enumerate(labels):
            g = f[label]
            attrs = dict(g.attrs)
            d = g["diagnostics"]
            t = d["t"][:]
            n = len(t)

            cols = {name: d[name][:] for name in SENSOR_CHANNELS}
            i0 = d["I0"][:]
            tmax = d["Tmax"][:]
            frac_left = d["fracLeft"][:]
            tmax_left = g["transient"]["TmaxLeft"][:]
            tmax_right = g["transient"]["TmaxRight"][:]

            quench_idx = np.where(tmax >= QUENCH_T_K)[0]
            quench_step = float(quench_idx[0]) if len(quench_idx) else np.nan

            episode_rows = []
            for step in range(n):
                row = dict(
                    episode_id=episode_id,
                    ratio_label=label,
                    ratio=float(attrs["ratio"]),
                    step=step,
                    t_s=float(t[step]),
                    T0_K=float(tmax[0]),
                    I_A=float(i0[step]),
                    I_true_A=float(i0[step]),
                    T_surface_K=float(tmax[step]),
                    T_surface_true_K=float(tmax[step]),
                    voltage_V=float(cols["V_CL_minus"][step] - cols["V_CL_plus"][step]),
                    voltage_true_V=float(cols["V_CL_minus"][step] - cols["V_CL_plus"][step]),
                    fracLeft=float(frac_left[step]),
                    TmaxLeft=float(tmax_left[step]),
                    TmaxRight=float(tmax_right[step]),
                    runaway=int(attrs["status"] == "runaway"),
                    status=str(attrs["status"]),
                    quench_step=quench_step,
                )
                for name in SENSOR_CHANNELS:
                    row[name] = float(cols[name][step])
                for name in _UNDEFINED_FOR_FEM:
                    row[name] = np.nan
                episode_rows.append(row)
            rows_by_episode.extend(episode_rows)

    return pd.DataFrame(rows_by_episode)
