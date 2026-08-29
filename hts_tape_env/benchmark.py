"""
benchmark.py

Unified benchmark infrastructure for all HTS conductor surrogate environments.
Computes metrics both globally and restricted to nonlinear T0 bands, generates
diagnostic figures and error surface plots for each conductor type.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# ── Ground-truth solver dispatch ────────────────────────────────────────────

def solve_ground_truth(
    T0: float,
    solver: str,
    solver_kwargs: dict | None = None,
) -> tuple[float, float]:
    """Compute (I*, T*) at a single operating temperature using the specified solver.

    solver: "bi2223" | "rebco" | "fd1d"
    Returns (Istar, Tstar). Both may be NaN if no runaway is found.
    """
    solver_kwargs = solver_kwargs or {}
    if solver == "bi2223":
        from hts_tape_env.physics_bi2223 import solve_runaway
        return solve_runaway(T0, mode=solver_kwargs.get("mode", "no_stab"))
    elif solver == "fd1d":
        from hts_tape_env.physics_fd1d import (
            Tape1D, HeaterPulse, CoolingModel, find_Istar_fd,
        )
        from hts_tape_env.materials import lumped_rebco_tape
        nx = solver_kwargs.get("nx", 50)
        tape = Tape1D(geom=lumped_rebco_tape(L=0.2, w=0.004), nx=nx)
        heater = HeaterPulse(power=13.0, duration=0.01)
        cool = CoolingModel(mode="bath_ends", T_bath=T0)
        return find_Istar_fd(
            tape, T0, heater=heater, cool=cool,
            tol=solver_kwargs.get("tol", 3.0),
            max_iter=solver_kwargs.get("max_iter", 10),
            t_end=0.10, dt=1e-4,
        )
    else:
        from hts_tape_env.physics import solve_runaway
        return solve_runaway(T0, **solver_kwargs)


# ── Nonlinear-region metrics ─────────────────────────────────────────────────

def _rel_err(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    denom = np.where(np.abs(y_true) > 1e-12, np.abs(y_true), np.nan)
    return np.abs(y_pred - y_true) / denom


def nonlinear_region_metrics(
    T0_eval: np.ndarray,
    Istar_true: np.ndarray,
    Istar_pred: np.ndarray,
    band: tuple[float, float],
) -> dict:
    """
    Compute I* relative error restricted to a T0 band (the nonlinear region)
    and over the full range for comparison.

    Returns dict with keys: full_median_rel_err, full_95th_rel_err,
    band_median_rel_err, band_95th_rel_err, band_n_points.
    """
    T0 = np.asarray(T0_eval).ravel()
    yt = np.asarray(Istar_true).ravel()
    yp = np.asarray(Istar_pred).ravel()

    valid = np.isfinite(yt) & np.isfinite(yp) & (yt > 0)
    re = _rel_err(yt[valid], yp[valid])
    T0v = T0[valid] if len(T0) == len(yt) else T0[valid]

    mask_band = (T0v >= band[0]) & (T0v <= band[1])
    re_band = re[mask_band]

    return {
        "full_median_rel_err": float(np.nanmedian(re)),
        "full_95th_rel_err": float(np.nanpercentile(re, 95)),
        "band_median_rel_err": float(np.nanmedian(re_band)) if len(re_band) > 0 else np.nan,
        "band_95th_rel_err": float(np.nanpercentile(re_band, 95)) if len(re_band) > 0 else np.nan,
        "band_n_points": int(np.sum(mask_band)),
        "band": list(band),
    }


# ── Point surrogate benchmarking ─────────────────────────────────────────────

def benchmark_point_surrogates(
    df_test: pd.DataFrame,
    specs: dict,
    models: dict,
    *,
    T0_col: str = "T0_K",
    nonlinear_band: tuple[float, float] = (79.5, 80.5),
) -> dict:
    """
    Evaluate a dict of trained point surrogate models on a test DataFrame.
    Returns per-spec metrics with both global and nonlinear-band breakdowns.
    """
    from sklearn.metrics import mean_absolute_error, r2_score

    results = {}
    for name, spec in specs.items():
        if name not in models:
            continue
        model = models[name]
        try:
            X, y = spec.prepare(df_test)
        except (KeyError, ValueError):
            continue

        pred = model.predict(X)
        re = _rel_err(y, pred)

        T0_vals = df_test.loc[df_test.index[:len(y)], T0_col].values[:len(y)]
        band_mask = (T0_vals >= nonlinear_band[0]) & (T0_vals <= nonlinear_band[1])

        metrics = {
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)),
            "median_rel_err": float(np.nanmedian(re)),
            "95th_rel_err": float(np.nanpercentile(re, 95)),
        }
        if np.any(band_mask):
            re_band = re[band_mask]
            metrics["band_median_rel_err"] = float(np.nanmedian(re_band))
            metrics["band_95th_rel_err"] = float(np.nanpercentile(re_band, 95))
            metrics["band_n_points"] = int(np.sum(band_mask))
        metrics["nonlinear_band"] = list(nonlinear_band)

        results[name] = metrics
    return results


# ── Sequence surrogate benchmarking ──────────────────────────────────────────

def benchmark_seq_surrogates(
    seq_metrics: dict,
    T0_eval: np.ndarray,
    Istar_analytic: np.ndarray,
    Istar_pred: np.ndarray,
    *,
    Tstar_analytic: np.ndarray | None = None,
    Tstar_pred: np.ndarray | None = None,
    nonlinear_band: tuple[float, float] = (79.5, 80.5),
) -> dict:
    """
    Augment existing sequence surrogate metrics with nonlinear-band breakdown.
    """
    nlr = nonlinear_region_metrics(T0_eval, Istar_analytic, Istar_pred, nonlinear_band)
    result = {**seq_metrics, **nlr}

    if Tstar_analytic is not None and Tstar_pred is not None:
        valid = np.isfinite(Tstar_analytic) & np.isfinite(Tstar_pred)
        if np.any(valid):
            result["tstar_mae_K"] = float(np.mean(np.abs(
                Tstar_analytic[valid] - Tstar_pred[valid])))

    return result


# ── Report generation ────────────────────────────────────────────────────────

def make_benchmark_report(
    conductor_name: str,
    point_metrics: dict,
    seq_metrics: dict,
    *,
    out_dir: str | Path = "models",
) -> Path:
    """Write a JSON benchmark report and print a summary table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "conductor": conductor_name,
        "point_surrogates": point_metrics,
        "seq_surrogates": seq_metrics,
    }

    path = out_dir / f"{conductor_name}_eval_metrics.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)

    _print_summary(conductor_name, point_metrics, seq_metrics)
    return path


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def _print_summary(conductor: str, point_m: dict, seq_m: dict):
    print(f"\n{'='*60}")
    print(f"  Benchmark: {conductor}")
    print(f"{'='*60}")

    if point_m:
        print("\n  Point surrogates:")
        print(f"  {'Spec':<8} {'MAE':>10} {'R²':>8} {'Med RelErr':>12} {'Band MedErr':>12}")
        for name, m in point_m.items():
            band_err = m.get("band_median_rel_err", float("nan"))
            print(f"  {name:<8} {m['mae']:>10.4f} {m['r2']:>8.5f} "
                  f"{m['median_rel_err']:>12.4%} {band_err:>12.4%}")

    if seq_m:
        print("\n  Sequence surrogates:")
        for name, m in seq_m.items():
            if isinstance(m, dict):
                med = m.get("full_median_rel_err", m.get("istar_median_rel_err", float("nan")))
                band = m.get("band_median_rel_err", float("nan"))
                print(f"  {name}: I* med rel err = {med:.4%}, band = {band:.4%}")

    print(f"{'='*60}\n")


# ── Diagnostic figure ────────────────────────────────────────────────────────

def make_diagnostic_figure(
    conductor_name: str,
    T0_eval: np.ndarray,
    Istar_analytic: np.ndarray,
    Istar_pred: dict[str, np.ndarray],
    nonlinear_band: tuple[float, float],
    *,
    out_dir: str | Path = "models",
) -> Path:
    """
    3×3 diagnostic figure: I*(T0) for each model, error vs T0, residual
    distribution. Works for any conductor.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(Istar_pred.keys())
    n = min(len(model_names), 3)

    fig, axes = plt.subplots(3, max(n, 1), figsize=(5 * max(n, 1), 12))
    if n == 1:
        axes = axes.reshape(3, 1)

    T0 = np.asarray(T0_eval)
    yt = np.asarray(Istar_analytic)

    for col, name in enumerate(model_names[:n]):
        yp = np.asarray(Istar_pred[name])
        valid = np.isfinite(yt) & np.isfinite(yp) & (yt > 0)
        re = _rel_err(yt[valid], yp[valid])
        T0v = T0[valid] if len(T0) == len(yt) else T0[:np.sum(valid)]

        # Row 0: I*(T0) overlay
        ax = axes[0, col]
        ax.plot(T0v, yt[valid], "k-", lw=1.5, label="analytic")
        ax.plot(T0v, yp[valid], "r--", lw=1, label=name)
        ax.axvspan(*nonlinear_band, alpha=0.15, color="orange", label="NL band")
        ax.set_ylabel("I* [A]")
        ax.set_title(name)
        ax.legend(fontsize=7)

        # Row 1: relative error vs T0
        ax = axes[1, col]
        ax.semilogy(T0v, re, ".", ms=3, alpha=0.6)
        ax.axvspan(*nonlinear_band, alpha=0.15, color="orange")
        ax.set_ylabel("|rel err|")
        ax.set_xlabel("T0 [K]")

        # Row 2: residual histogram
        ax = axes[2, col]
        residual = yp[valid] - yt[valid]
        ax.hist(residual, bins=40, edgecolor="k", alpha=0.7)
        ax.set_xlabel("I*_pred - I*_analytic [A]")
        ax.set_ylabel("count")

    fig.suptitle(f"{conductor_name} — surrogate diagnostics", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / f"{conductor_name}_diagnostics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Error surface plots ──────────────────────────────────────────────────────

def plot_error_surfaces(
    conductor_name: str,
    T0_grid: np.ndarray,
    Istar_analytic: np.ndarray,
    Istar_pred: np.ndarray,
    nonlinear_band: tuple[float, float],
    *,
    I_grid: np.ndarray | None = None,
    curvature_vals: np.ndarray | None = None,
    sensitivity_vals: np.ndarray | None = None,
    time_margin_analytic: np.ndarray | None = None,
    time_margin_pred: np.ndarray | None = None,
    lam_grid: np.ndarray | None = None,
    out_dir: str | Path = "models",
) -> Path:
    """
    4-panel error surface figure inspired by the paper's equilibrium branch /
    stability boundary figures.

    For 1-parameter conductors (Bi-2223, no λ): T0-only line plots.
    For 2-parameter conductors (REBCO with λ): (T0, λ) heatmaps.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    T0 = np.asarray(T0_grid)
    yt = np.asarray(Istar_analytic)
    yp = np.asarray(Istar_pred)

    is_2d = lam_grid is not None and yt.ndim == 2

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    if is_2d:
        _plot_error_surfaces_2d(fig, axes, T0, lam_grid, yt, yp, nonlinear_band,
                                curvature_vals, time_margin_analytic, time_margin_pred)
    else:
        _plot_error_surfaces_1d(fig, axes, T0, yt, yp, nonlinear_band,
                                curvature_vals, sensitivity_vals,
                                time_margin_analytic, time_margin_pred, I_grid)

    fig.suptitle(f"{conductor_name} — error surfaces", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / f"{conductor_name}_error_surfaces.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_error_surfaces_1d(fig, axes, T0, yt, yp, band,
                             curvature, sensitivity, tm_true, tm_pred, I_grid):
    valid = np.isfinite(yt) & np.isfinite(yp) & (yt > 0)
    re = _rel_err(yt[valid], yp[valid])
    T0v = T0[valid]

    # Panel 1: I*(T0) analytic + prediction with error band
    ax = axes[0, 0]
    ax.plot(T0v, yt[valid], "k-", lw=2, label="analytic I*(T₀)")
    ax.plot(T0v, yp[valid], "r--", lw=1.5, label="surrogate")
    err_abs = np.abs(yp[valid] - yt[valid])
    ax.fill_between(T0v, yt[valid] - err_abs, yt[valid] + err_abs,
                     alpha=0.2, color="red", label="± |error|")
    ax.axvspan(*band, alpha=0.15, color="orange", label="NL band")
    ax.set_xlabel("T₀ [K]")
    ax.set_ylabel("I* [A]")
    ax.legend(fontsize=7)
    ax.set_title("I*(T₀) prediction vs analytic")

    # Panel 2: Relative error heatmap in (T0, I/I*) space — or error vs T0 for 1D
    ax = axes[0, 1]
    ax.semilogy(T0v, re, "o", ms=4, alpha=0.7, color="steelblue")
    ax.axvspan(*band, alpha=0.15, color="orange")
    ax.axhline(np.nanmedian(re), color="gray", ls="--", lw=0.8, label=f"median={np.nanmedian(re):.4%}")
    ax.set_xlabel("T₀ [K]")
    ax.set_ylabel("|relative error|")
    ax.set_title("Relative error vs T₀")
    ax.legend(fontsize=7)

    # Panel 3: Error vs curvature/sensitivity
    ax = axes[1, 0]
    if curvature is not None and len(curvature) == len(re):
        curv_valid = np.isfinite(curvature)
        ax.scatter(curvature[curv_valid], re[curv_valid], s=15, alpha=0.6, c="teal")
        ax.set_xlabel("runaway curvature (d²I*/dT₀²)")
        ax.set_ylabel("|relative error|")
        ax.set_yscale("log")
        ax.set_title("Error vs curvature")
    elif sensitivity is not None and len(sensitivity) == len(re):
        sens_valid = np.isfinite(sensitivity)
        ax.scatter(sensitivity[sens_valid], re[sens_valid], s=15, alpha=0.6, c="coral")
        ax.set_xlabel("runaway sensitivity (|dI*/dT₀|)")
        ax.set_ylabel("|relative error|")
        ax.set_yscale("log")
        ax.set_title("Error vs sensitivity")
    else:
        ax.text(0.5, 0.5, "No curvature/sensitivity\ndata available",
                transform=ax.transAxes, ha="center", va="center", fontsize=11, color="gray")
        ax.set_title("Error vs curvature (N/A)")

    # Panel 4: Time-margin error for overcurrent regime
    ax = axes[1, 1]
    if tm_true is not None and tm_pred is not None:
        tm_t = np.asarray(tm_true).ravel()
        tm_p = np.asarray(tm_pred).ravel()
        tm_valid = np.isfinite(tm_t) & np.isfinite(tm_p) & (tm_t > 0)
        if np.any(tm_valid):
            tm_re = _rel_err(tm_t[tm_valid], tm_p[tm_valid])
            ax.semilogy(tm_t[tm_valid] * 1000, tm_re, "s", ms=4, alpha=0.6, color="darkred")
            ax.set_xlabel("time margin (analytic) [ms]")
            ax.set_ylabel("|relative error|")
            ax.set_title("Time-margin prediction error")
        else:
            ax.text(0.5, 0.5, "No valid time-margin\ndata", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="gray")
            ax.set_title("Time-margin error (N/A)")
    else:
        ax.text(0.5, 0.5, "Time-margin data\nnot provided", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color="gray")
        ax.set_title("Time-margin error (N/A)")


def _plot_error_surfaces_2d(fig, axes, T0, lam, yt, yp, band, curvature,
                             tm_true, tm_pred):
    re = _rel_err(yt, yp)

    TT, LL = np.meshgrid(T0, lam)

    # Panel 1: I*(T0, λ) analytic surface
    ax = axes[0, 0]
    valid_mask = np.isfinite(yt)
    vmin, vmax = np.nanmin(yt[valid_mask]), np.nanmax(yt[valid_mask])
    c = ax.pcolormesh(TT, LL, yt, cmap="viridis", shading="auto")
    fig.colorbar(c, ax=ax, label="I* [A]")
    ax.axvspan(*band, alpha=0.2, color="orange")
    ax.set_xlabel("T₀ [K]")
    ax.set_ylabel("λ [W/m/K]")
    ax.set_title("Analytic I*(T₀, λ)")

    # Panel 2: Relative error heatmap
    ax = axes[0, 1]
    re_plot = np.where(np.isfinite(re), re, np.nan)
    c = ax.pcolormesh(TT, LL, re_plot, cmap="Reds", shading="auto",
                      norm=matplotlib.colors.LogNorm(vmin=1e-5, vmax=max(np.nanmax(re_plot), 1e-2)))
    fig.colorbar(c, ax=ax, label="|rel err|")
    ax.axvspan(*band, alpha=0.2, color="orange")
    ax.set_xlabel("T₀ [K]")
    ax.set_ylabel("λ [W/m/K]")
    ax.set_title("Surrogate relative error")

    # Panel 3: Error vs curvature
    ax = axes[1, 0]
    if curvature is not None:
        curv = np.asarray(curvature).ravel()
        re_flat = re.ravel()
        valid = np.isfinite(curv) & np.isfinite(re_flat)
        ax.scatter(curv[valid], re_flat[valid], s=10, alpha=0.5, c="teal")
        ax.set_xlabel("curvature")
        ax.set_ylabel("|rel err|")
        ax.set_yscale("log")
    ax.set_title("Error vs curvature")

    # Panel 4: Time-margin error
    ax = axes[1, 1]
    if tm_true is not None and tm_pred is not None:
        tm_re = _rel_err(np.asarray(tm_true), np.asarray(tm_pred))
        tm_re_plot = np.where(np.isfinite(tm_re), tm_re, np.nan)
        c = ax.pcolormesh(TT, LL, tm_re_plot, cmap="Oranges", shading="auto",
                          norm=matplotlib.colors.LogNorm(vmin=1e-5, vmax=max(np.nanmax(tm_re_plot), 1e-2)))
        fig.colorbar(c, ax=ax, label="|rel err|")
        ax.set_xlabel("T₀ [K]")
        ax.set_ylabel("λ [W/m/K]")
    ax.set_title("Time-margin error")
