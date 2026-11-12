"""
fem_runaway_data.py

Loads the FEA no-heater current sweep (local-FreeFEM 3D-slit electrothermal,
sweep_transient.py) as time-to-runaway / runaway-probability training data.
The sweep has 801 runs at target I/Ic = 0.700..1.500 (step 0.001, Ic ~ 311 A)
with T0 and cooling fixed, so every run shares one I* -- the learnable targets
are when (and whether) runaway happens, not I* itself.

Physics / labeling assumptions:
- Runaway onset t_run is the first time Tmax >= threshold_K (default Tc = 90 K),
  linearly interpolated between solver steps. The solver's own stop criterion
  (Tmax > 900 K) is not in the saved diagnostics, and its status attribute
  mislabels ratios 1.265-1.268 as stable although Tmax is already past Tc.
- Runs that never reach threshold_K are right-censored, not stable: from
  ~1.20 Ic upward they are still heating with dT/dt increasing when the 1.5 s
  hold window ends. time_to_runaway_s is NaN for them; t_end_s gives the
  observed window, so t_end_s - t_s is a lower bound on the time to runaway.
- Signals are linearly resampled onto a uniform dt_s grid: the solver's
  adaptive step shrinks as runaway approaches, which would leak the label to
  the model, and a real DAQ samples at a fixed rate.
- Runaway episodes are truncated at onset (the last grid point <= t_run); past
  that point the protection decision has already been missed.
- T_surface_K is the hottest of the 16 surface RTDs -- what an instrument
  sees. Tmax (the hottest point in the mesh, up to ~13 K above the RTDs
  during runaway) only defines the label and is kept as Tmax_K.
- voltage_V is the whole-tape voltage V_CL_minus - V_CL_plus, as in fem_data.py.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from hts_tape_env.fem_data import RTD_CHANNELS

DEFAULT_PATH = (Path.home() / "Downloads/local-FreeFEM/3D-slit/electrothermal/"
                "runs/noheater_sweep/sweep.h5")
T_RUNAWAY_K = 90.0
DT_S = 0.01
FEATURES = ["I_A", "voltage_V", "T_surface_K"]


def runaway_onset_time(t: np.ndarray, tmax: np.ndarray,
                       threshold_K: float = T_RUNAWAY_K) -> float:
    """First time tmax reaches threshold_K, linearly interpolated; NaN if never."""
    above = np.nonzero(tmax >= threshold_K)[0]
    if len(above) == 0:
        return np.nan
    i = above[0]
    if i == 0:
        return float(t[0])
    frac = (threshold_K - tmax[i - 1]) / (tmax[i] - tmax[i - 1])
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))


def _resample_run(diagnostics, dt_s: float, threshold_K: float) -> dict:
    t = diagnostics["t"][:]
    tmax = diagnostics["Tmax"][:]
    t_run = runaway_onset_time(t, tmax, threshold_K)
    t_stop = t_run if np.isfinite(t_run) else t[-1]
    n = int(np.floor((t_stop - t[0]) / dt_s + 1e-9)) + 1
    grid = t[0] + dt_s * np.arange(n)

    voltage = diagnostics["V_CL_minus"][:] - diagnostics["V_CL_plus"][:]
    rtd_max = np.max([diagnostics[c][:] for c in RTD_CHANNELS], axis=0)
    return dict(
        t_s=grid - t[0],
        I_A=np.interp(grid, t, diagnostics["I0"][:]),
        voltage_V=np.interp(grid, t, voltage),
        T_surface_K=np.interp(grid, t, rtd_max),
        Tmax_K=np.interp(grid, t, tmax),
        t_runaway_s=t_run - t[0],
        t_end_s=t_stop - t[0],
    )


def load_runaway_sweep(path: str | Path = DEFAULT_PATH, dt_s: float = DT_S,
                       threshold_K: float = T_RUNAWAY_K) -> pd.DataFrame:
    """
    Long-format table, one row per (episode, uniform grid step):

        episode_id, label, ratio, status   run identity (ascending ratio);
                                           status is the solver's own label
        step, t_s                          grid index / time since run start [s]
        I_A, voltage_V, T_surface_K        model input channels (FEATURES)
        Tmax_K                             hottest mesh point [K], label source only
        runaway                            1 if Tmax reached threshold_K
        t_runaway_s                        onset time since run start, NaN if censored
        t_end_s                            last observed time (= t_runaway_s if runaway)
        time_to_runaway_s                  t_runaway_s - t_s, NaN if censored
    """
    frames = []
    with h5py.File(path, "r") as f:
        labels = sorted(f.keys(), key=lambda k: float(f[k].attrs["ratio"]))
        for episode_id, label in enumerate(labels):
            g = f[label]
            r = _resample_run(g["diagnostics"], dt_s, threshold_K)
            frames.append(pd.DataFrame(dict(
                episode_id=episode_id,
                label=label,
                ratio=float(g.attrs["ratio"]),
                status=str(g.attrs["status"]),
                step=np.arange(len(r["t_s"])),
                **r,
                runaway=int(np.isfinite(r["t_runaway_s"])),
                time_to_runaway_s=r["t_runaway_s"] - r["t_s"],
            )))
    return pd.concat(frames, ignore_index=True)
