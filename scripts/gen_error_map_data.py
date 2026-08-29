#!/usr/bin/env python
"""Generate per-T0-point error data for all baseline benchmarks."""

from __future__ import annotations
import _bootstrap  # noqa: F401

import json
import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# ── Conductor configs ──────────────────────────────────────────────────
CONDUCTORS = {
    "bi2223_nostab": {
        "label": "Bi-2223 no-stab",
        "T0_range": (77.5, 81.0),
        "n_points": 60,
        "nonlinear_band": [79.5, 81.0],
        "point_specs": ["A", "B", "raw"],
        "best_point": "raw",
        "solver": "bi2223",
        "solver_kwargs": {"mode": "no_stab"},
    },
    "bi2223_agstab": {
        "label": "Bi-2223 Ag-stab",
        "T0_range": (77.26, 79.0),
        "n_points": 60,
        "nonlinear_band": [78.0, 79.0],
        "point_specs": ["A", "B", "raw"],
        "best_point": "raw",
        "solver": "bi2223",
        "solver_kwargs": {"mode": "ag_stab"},
    },
    "rebco_nostab": {
        "label": "REBCO no-redist",
        "T0_range": (10.0, 80.0),
        "n_points": 60,
        "nonlinear_band": [70.0, 80.0],
        "point_specs": ["A", "two_d", "B", "raw"],
        "best_point": "two_d",
        "solver": "rebco",
        "solver_kwargs": {"n": 20, "mode": "no_stab", "lam_override": 100.0},
    },
    "rebco_stab": {
        "label": "REBCO Cu/SS redist",
        "T0_range": (10.0, 80.0),
        "n_points": 60,
        "nonlinear_band": [70.0, 80.0],
        "point_specs": ["A", "two_d", "B", "raw"],
        "best_point": "two_d",
        "solver": "rebco",
        "solver_kwargs": {"n": 20, "mode": "stab", "lam_override": 100.0},
    },
    "rebco_fd1d": {
        "label": "REBCO 1D-FD",
        "T0_range": (10.0, 75.0),
        "n_points": 30,
        "nonlinear_band": [70.0, 80.0],
        "point_specs": ["A", "two_d", "B", "raw"],
        "best_point": "A",
        "solver": "fd1d",
        "solver_kwargs": {"nx": 50, "tol": 3.0, "max_iter": 10},
    },
}

def solve_istar(T0, solver, solver_kwargs):
    if solver == "bi2223":
        from hts_tape_env.physics_bi2223 import solve_runaway
        mode = solver_kwargs.get("mode", "no_stab")
        Is, Ts = solve_runaway(T0, mode=mode)
        return Is, Ts
    elif solver == "fd1d":
        from hts_tape_env.physics_fd1d import (
            Tape1D, HeaterPulse, CoolingModel, find_Istar_fd,
        )
        from hts_tape_env.materials import lumped_rebco_tape
        nx = solver_kwargs.get("nx", 50)
        tape = Tape1D(geom=lumped_rebco_tape(L=0.2, w=0.004), nx=nx)
        heater = HeaterPulse(power=13.0, duration=0.01)
        cool = CoolingModel(mode="bath_ends", T_bath=T0)
        Is, Ts = find_Istar_fd(
            tape, T0, heater=heater, cool=cool,
            tol=solver_kwargs.get("tol", 3.0),
            max_iter=solver_kwargs.get("max_iter", 10),
            t_end=0.10, dt=1e-4,
        )
        return Is, Ts
    else:
        from hts_tape_env.physics import solve_runaway
        Is, Ts = solve_runaway(T0, **solver_kwargs)
        return Is, Ts


def main():
    from hts_tape_env.surrogates import load_surrogate

    results = {}

    for cond_name, cfg in CONDUCTORS.items():
        print(f"\n=== {cfg['label']} ===")
        T0_lo, T0_hi = cfg["T0_range"]
        T0_grid = np.linspace(T0_lo, T0_hi, cfg["n_points"])

        Istar_analytic = np.full(len(T0_grid), np.nan)
        Tstar_analytic = np.full(len(T0_grid), np.nan)
        for i, T0 in enumerate(T0_grid):
            Is, Ts = solve_istar(T0, cfg["solver"], cfg["solver_kwargs"])
            if np.isfinite(Is) and Is > 0:
                Istar_analytic[i] = Is
                Tstar_analytic[i] = Ts

        valid = np.isfinite(Istar_analytic) & (Istar_analytic > 0)
        print(f"  T0 range: {T0_lo}–{T0_hi} K, {valid.sum()}/{len(T0_grid)} valid analytic points")

        surr_errors = {}
        for spec in cfg["point_specs"]:
            p = MODEL_DIR / f"{cond_name}_surr_{spec}.joblib"
            if not p.exists():
                print(f"  {spec}: not found")
                continue
            model_data = load_surrogate(p)
            model = model_data["model"]

            if spec == "A":
                X = T0_grid[valid].reshape(-1, 1)
            elif spec == "two_d":
                lam_val = cfg["solver_kwargs"].get("lam_override", 0.0)
                X = np.column_stack([T0_grid[valid], np.full(valid.sum(), lam_val)])
            elif spec == "B":
                X = np.column_stack([
                    T0_grid[valid],
                    Istar_analytic[valid] * 0.5,
                ])
            elif spec == "raw":
                X = np.column_stack([
                    T0_grid[valid],
                    Istar_analytic[valid] * 0.5,
                    np.zeros(valid.sum()),
                    T0_grid[valid] + 0.1,
                ])

            pred = model.predict(X)
            rel_err = np.abs(pred - Istar_analytic[valid]) / np.maximum(np.abs(Istar_analytic[valid]), 1e-6)
            surr_errors[spec] = {
                "T0": T0_grid[valid].tolist(),
                "Istar_pred": pred.tolist(),
                "rel_err": rel_err.tolist(),
                "median_rel_err": float(np.nanmedian(rel_err)),
                "p95_rel_err": float(np.nanpercentile(rel_err, 95)),
                "max_rel_err": float(np.nanmax(rel_err)),
            }
            print(f"  {spec}: median={np.nanmedian(rel_err):.4%}, 95th={np.nanpercentile(rel_err, 95):.4%}")

        metrics_path = MODEL_DIR / f"{cond_name}_eval_metrics.json"
        seq_metrics = {}
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            seq_raw = m.get("seq_surrogates", {})
            if not seq_raw:
                for k in ("seq_gru", "seq_transformer"):
                    if k in m:
                        arch = k.replace("seq_", "")
                        seq_raw[arch] = m[k]
            for arch, sm in seq_raw.items():
                seq_metrics[arch] = {
                    "istar_median_rel_err": sm.get("istar_median_rel_err") or sm.get("full_median_rel_err", None),
                    "istar_95th_rel_err": sm.get("istar_95th_rel_err") or sm.get("full_95th_rel_err", None),
                    "tstar_mae_K": sm.get("tstar_mae_K"),
                    "breach_accuracy": sm.get("breach_accuracy"),
                    "time_margin_median_rel_err": sm.get("time_margin_median_rel_err"),
                    "band_median_rel_err": sm.get("band_median_rel_err"),
                    "band_95th_rel_err": sm.get("band_95th_rel_err"),
                }

        results[cond_name] = {
            "label": cfg["label"],
            "T0_range": [T0_lo, T0_hi],
            "nonlinear_band": cfg["nonlinear_band"],
            "best_point_spec": cfg["best_point"],
            "T0_grid": T0_grid.tolist(),
            "Istar_analytic": [float(x) if np.isfinite(x) else None for x in Istar_analytic],
            "Tstar_analytic": [float(x) if np.isfinite(x) else None for x in Tstar_analytic],
            "point_surrogates": surr_errors,
            "seq_surrogates": seq_metrics,
        }

    out_path = MODEL_DIR / "thermal_runaway_error_map_data.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
