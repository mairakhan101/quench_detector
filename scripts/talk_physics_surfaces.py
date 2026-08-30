#!/usr/bin/env python
"""
talk_physics_surfaces.py

Generate physics-grounded 3D error surface plots for the talk.
Shows WHERE in (T₀, λ) parameter space surrogates fail, with:
  - Analytic I*(T₀, λ) runaway surface
  - Surrogate prediction overlay
  - Signed error surface (over/underprediction matters for protection)
  - I/I* margin error in the critical near-boundary region

For REBCO (2-param): full 3D surfaces with λ dependence
For Bi-2223 (1-param): 1D curves with error bands
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path

# ── Load checkpoint and compute surfaces ──────────────────────────────────

def load_and_predict(checkpoint_path, T0_grid, lam_grid, env_kwargs=None,
                     conductor_type="rebco"):
    """Load a seq surrogate checkpoint and compute Istar surfaces."""
    from hts_tape_env.seq_surrogates import load_seq_surrogate, predict_runaway_surface, TRUE_FEATURES
    result = load_seq_surrogate(checkpoint_path)
    print(f"  Loaded {result.arch} from {checkpoint_path}")

    if conductor_type == "bi2223":
        surfaces = _predict_bi2223_surface(result, T0_grid, env_kwargs or {})
    else:
        surfaces = predict_runaway_surface(
            result, T0_grid, lam_grid,
            env_kwargs=env_kwargs or {},
        )
    return result.arch, surfaces


def _predict_bi2223_surface(result, T0_grid, env_kwargs):
    """Bi-2223 surface prediction using the correct env class."""
    from hts_tape_env.env_bi2223 import Bi2223QuenchDetectorEnv
    from hts_tape_env.seq_surrogates import TRUE_FEATURES

    device = "cpu"
    model = result.model.to(device)
    model.eval()

    env = Bi2223QuenchDetectorEnv(**env_kwargs)
    n_t0 = len(T0_grid)
    surfaces = {k: np.full((1, n_t0), np.nan) for k in
                ("Istar_analytic", "Istar_pred", "Tstar_analytic", "Tstar_pred",
                 "time_margin_analytic_ms", "time_margin_pred_ms")}
    feature_cols = result.meta.get("features", TRUE_FEATURES)

    feature_cols = result.meta.get("features", TRUE_FEATURES)

    for i, T0 in enumerate(T0_grid):
        obs, info = env.reset_at(float(T0), seed=i)
        if not np.isfinite(info.get("I_star", np.nan)) or info["I_star"] <= 0:
            continue

        istar_true = info["I_star"]
        tstar_true = info["T_star"]

        # Build feature rows using rollout_bi2223_env's _row convention:
        # model expects true values from info, not noisy obs
        rows = []
        done = False
        while True:
            row = {
                "T0_K": float(obs[0]),
                "I_A": float(obs[1]),
                "voltage_V": float(obs[2]),
                "T_surface_K": float(obs[3]),
                "I_true_A": info["I_true_A"],
                "voltage_true_V": info["voltage_true_V"],
                "T_surface_true_K": info["T_surface_true_K"],
            }
            rows.append(row)
            if done:
                break
            obs, _, terminated, truncated, info = env.step(0)
            done = terminated or truncated

        X = np.array([[r[c] for c in feature_cols] for r in rows], dtype=np.float32)
        X = result.normalizer.transform(X)
        Xt = torch.from_numpy(X).unsqueeze(0).to(device)
        lengths = torch.tensor([X.shape[0]], device=device)
        with torch.no_grad():
            out = model(Xt, lengths)

        surfaces["Istar_analytic"][0, i] = istar_true
        surfaces["Istar_pred"][0, i] = float(np.exp(out["istar_log"].cpu().item()))
        surfaces["Tstar_analytic"][0, i] = tstar_true
        surfaces["Tstar_pred"][0, i] = float(out["tstar"].cpu().item())

    return surfaces


def signed_rel_err(yt, yp):
    """Signed relative error: positive = overprediction (unsafe), negative = under."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(yt) & (yt > 0), (yp - yt) / yt, np.nan)


# ── REBCO 3D surfaces ────────────────────────────────────────────────────

def plot_rebco_3d_comparison(conductor_name, T0_grid, lam_grid,
                              arch_surfaces, out_dir, nonlinear_band=(70, 80)):
    """
    6-panel figure for a 2-parameter REBCO conductor:
    Row 1: Analytic I*(T₀,λ), GRU prediction, GRU signed error
    Row 2: Transformer prediction, Transformer signed error, error heatmap comparison
    """
    out_dir = Path(out_dir)
    TT, LL = np.meshgrid(T0_grid, lam_grid)

    archs = list(arch_surfaces.keys())
    yt = arch_surfaces[archs[0]]["Istar_analytic"]

    # Get valid range for I*
    valid = np.isfinite(yt)
    vmin_I, vmax_I = np.nanmin(yt[valid]), np.nanmax(yt[valid])

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("white")

    cmap_istar = "cividis"
    cmap_err = "RdBu_r"

    # Panel 1: Analytic I*(T₀, λ) — the ground truth
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    yt_plot = np.where(np.isfinite(yt), yt, np.nan)
    surf = ax1.plot_surface(TT, LL, yt_plot, cmap=cmap_istar,
                            vmin=vmin_I, vmax=vmax_I, alpha=0.9,
                            linewidth=0, antialiased=True)
    fig.colorbar(surf, ax=ax1, shrink=0.55, label="I* [A]", pad=0.1)
    ax1.set_xlabel("T₀ [K]", fontsize=8, labelpad=5)
    ax1.set_ylabel("λ [W/m/K]", fontsize=8, labelpad=5)
    ax1.set_zlabel("I* [A]", fontsize=8, labelpad=3)
    ax1.set_title("Analytic I*(T₀, λ)", fontsize=10, fontweight="bold", pad=10)
    ax1.view_init(elev=25, azim=-55)
    ax1.tick_params(labelsize=7)

    # Panels for each architecture
    for idx, arch in enumerate(archs[:2]):
        s = arch_surfaces[arch]
        yp = s["Istar_pred"]
        serr = signed_rel_err(yt, yp) * 100  # percent

        # Prediction surface
        col = idx + 2
        ax = fig.add_subplot(2, 3, col, projection="3d")
        yp_plot = np.where(np.isfinite(yp), yp, np.nan)
        surf = ax.plot_surface(TT, LL, yp_plot, cmap=cmap_istar,
                               vmin=vmin_I, vmax=vmax_I, alpha=0.9,
                               linewidth=0, antialiased=True)
        fig.colorbar(surf, ax=ax, shrink=0.55, label="I* [A]", pad=0.1)
        ax.set_xlabel("T₀ [K]", fontsize=8, labelpad=5)
        ax.set_ylabel("λ [W/m/K]", fontsize=8, labelpad=5)
        ax.set_zlabel("I* [A]", fontsize=8, labelpad=3)
        ax.set_title(f"{arch.upper()} I* prediction", fontsize=10, fontweight="bold", pad=10)
        ax.view_init(elev=25, azim=-55)
        ax.tick_params(labelsize=7)

        # Signed error surface
        row2_col = idx + 1
        ax_err = fig.add_subplot(2, 3, 3 + row2_col, projection="3d")
        serr_plot = np.where(np.isfinite(serr), serr, np.nan)
        emax = np.nanpercentile(np.abs(serr_plot), 98)
        emax = max(emax, 1.0)
        norm = TwoSlopeNorm(vmin=-emax, vcenter=0, vmax=emax)
        colors = plt.cm.RdBu_r(norm(np.nan_to_num(serr_plot)))
        ax_err.plot_surface(TT, LL, serr_plot, facecolors=colors, alpha=0.9,
                            linewidth=0, antialiased=True)
        sm = plt.cm.ScalarMappable(cmap=cmap_err, norm=norm)
        fig.colorbar(sm, ax=ax_err, shrink=0.55, label="error [%]", pad=0.1)
        ax_err.set_xlabel("T₀ [K]", fontsize=8, labelpad=5)
        ax_err.set_ylabel("λ [W/m/K]", fontsize=8, labelpad=5)
        ax_err.set_zlabel("(I*_pred − I*) / I* [%]", fontsize=8, labelpad=3)
        ax_err.set_title(f"{arch.upper()} signed error", fontsize=10, fontweight="bold", pad=10)
        ax_err.view_init(elev=25, azim=-55)
        ax_err.tick_params(labelsize=7)

    # Panel 6: Error heatmap comparison (2D top-down, all archs)
    ax_heat = fig.add_subplot(2, 3, 6)
    # Show worst-case |error| across architectures as heatmap
    all_abs_err = []
    arch_labels = []
    for arch in archs:
        s = arch_surfaces[arch]
        ae = np.abs(signed_rel_err(yt, s["Istar_pred"])) * 100
        all_abs_err.append(ae)
        arch_labels.append(arch.upper())

    # Use GRU error (the best) for the main heatmap
    gru_err = all_abs_err[0]
    gru_err_plot = np.where(np.isfinite(gru_err), gru_err, np.nan)
    c = ax_heat.pcolormesh(TT, LL, gru_err_plot, cmap="Reds",
                           norm=matplotlib.colors.LogNorm(
                               vmin=max(0.01, np.nanpercentile(gru_err_plot, 5)),
                               vmax=max(np.nanpercentile(gru_err_plot, 99), 1)),
                           shading="auto")
    fig.colorbar(c, ax=ax_heat, label="|GRU rel error| [%]")

    # Overlay NL band
    if nonlinear_band:
        ax_heat.axvspan(*nonlinear_band, alpha=0.2, color="orange",
                        label=f"NL band ({nonlinear_band[0]}–{nonlinear_band[1]} K)")

    # Add contour lines for error = 1%, 5%, 10%
    for level, ls in [(1, "-"), (5, "--"), (10, ":")]:
        try:
            ax_heat.contour(TT, LL, gru_err_plot, levels=[level],
                           colors=["black"], linewidths=0.8, linestyles=[ls])
        except ValueError:
            pass

    ax_heat.set_xlabel("T₀ [K]", fontsize=9)
    ax_heat.set_ylabel("λ [W/m/K]", fontsize=9)
    ax_heat.set_title("GRU |error| heatmap + contours", fontsize=10, fontweight="bold")
    ax_heat.legend(fontsize=7, loc="upper left")

    fig.suptitle(f"{conductor_name} — I*(T₀, λ) Surrogate Error Landscape",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = out_dir / f"talk_{conductor_name}_3d_surfaces.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ── Bi-2223 1D curves ────────────────────────────────────────────────────

def plot_bi2223_curves(conductor_name, T0_grid, arch_surfaces, out_dir,
                        nonlinear_band=None):
    """
    4-panel figure for 1-parameter Bi-2223:
    1. I*(T₀) analytic vs all architectures
    2. Signed error vs T₀
    3. Absolute error with NL band shading
    4. Relative error (%) with thresholds marked
    """
    out_dir = Path(out_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("white")

    colors = {"gru": "#2a78d6", "transformer": "#eb6834", "ssm": "#1baf7a"}
    archs = list(arch_surfaces.keys())
    yt = arch_surfaces[archs[0]]["Istar_analytic"][0]  # squeeze lam dim
    print(f"  [DEBUG plot] archs={archs}")
    print(f"  [DEBUG plot] T0_grid shape={T0_grid.shape}, range=[{T0_grid[0]:.2f}, {T0_grid[-1]:.2f}]")
    print(f"  [DEBUG plot] yt shape={yt.shape}, values={yt}")
    for arch in archs:
        yp = arch_surfaces[arch]["Istar_pred"][0]
        print(f"  [DEBUG plot] {arch} Istar_pred={yp}")

    # Panel 1: I*(T₀) curves
    ax = axes[0, 0]
    ax.plot(T0_grid, yt, "k-", lw=2.5, label="Analytic", zorder=10)
    for arch in archs:
        yp = arch_surfaces[arch]["Istar_pred"][0]
        ax.plot(T0_grid, yp, "--", color=colors.get(arch, "gray"),
                lw=1.5, label=f"{arch.upper()}", alpha=0.9)
    ax.set_xlabel("T₀ [K]")
    ax.set_ylabel("I* [A]")
    ax.set_title("Runaway current I*(T₀)", fontweight="bold")
    ax.legend(fontsize=8)
    if nonlinear_band:
        ax.axvspan(*nonlinear_band, alpha=0.1, color="orange")
    ax.grid(True, alpha=0.3)

    # Panel 2: Signed error
    ax = axes[0, 1]
    for arch in archs:
        yp = arch_surfaces[arch]["Istar_pred"][0]
        serr = signed_rel_err(yt, yp) * 100
        ax.plot(T0_grid, serr, "-", color=colors.get(arch, "gray"),
                lw=1.5, label=f"{arch.upper()}")
    ax.axhline(0, color="black", lw=0.5, ls="-")
    ax.axhline(5, color="gray", lw=0.5, ls=":")
    ax.axhline(-5, color="gray", lw=0.5, ls=":")
    ax.fill_between(T0_grid, -5, 5, alpha=0.05, color="green")
    ax.set_xlabel("T₀ [K]")
    ax.set_ylabel("(I*_pred − I*) / I* [%]")
    ax.set_title("Signed error (+ = overprediction = unsafe)", fontweight="bold")
    ax.legend(fontsize=8)
    if nonlinear_band:
        ax.axvspan(*nonlinear_band, alpha=0.1, color="orange")
    ax.grid(True, alpha=0.3)

    # Panel 3: Absolute error in Amps
    ax = axes[1, 0]
    for arch in archs:
        yp = arch_surfaces[arch]["Istar_pred"][0]
        ae = np.abs(yp - yt)
        ax.plot(T0_grid, ae, "-", color=colors.get(arch, "gray"),
                lw=1.5, label=f"{arch.upper()}")
    ax.set_xlabel("T₀ [K]")
    ax.set_ylabel("|I*_pred − I*| [A]")
    ax.set_title("Absolute error vs temperature", fontweight="bold")
    ax.legend(fontsize=8)
    if nonlinear_band:
        ax.axvspan(*nonlinear_band, alpha=0.1, color="orange", label="NL band")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Panel 4: Relative error (%)
    ax = axes[1, 1]
    for arch in archs:
        yp = arch_surfaces[arch]["Istar_pred"][0]
        re = np.abs(signed_rel_err(yt, yp)) * 100
        ax.plot(T0_grid, re, "-", color=colors.get(arch, "gray"),
                lw=1.5, label=f"{arch.upper()}")
    ax.axhline(1, color="green", lw=0.8, ls="--", alpha=0.6, label="1% threshold")
    ax.axhline(5, color="orange", lw=0.8, ls="--", alpha=0.6, label="5% threshold")
    ax.set_xlabel("T₀ [K]")
    ax.set_ylabel("|relative error| [%]")
    ax.set_title("Relative error with protection thresholds", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    if nonlinear_band:
        ax.axvspan(*nonlinear_band, alpha=0.1, color="orange")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    fig.suptitle(f"{conductor_name} — I*(T₀) Surrogate Error Landscape",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = out_dir / f"talk_{conductor_name}_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    out_dir = Path("models/talk_figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path("models/benchmark_prod_uniform")

    # REBCO no-stab (2-param: T₀, λ)
    print("=== REBCO no-stab ===")
    T0 = np.linspace(10, 80, 25)
    lam = np.linspace(20, 400, 20)
    rebco_ns = {}
    for arch in ["gru", "transformer"]:
        cp = base / f"rebco_nostab_seq_{arch}.pt"
        if cp.exists():
            a, s = load_and_predict(str(cp), T0, lam, env_kwargs={"mode": "no_stab"})
            rebco_ns[arch] = s
    if rebco_ns:
        plot_rebco_3d_comparison("REBCO no-stab", T0, lam, rebco_ns, out_dir)

    # REBCO Cu/SS stab (2-param)
    print("=== REBCO Cu/SS stab ===")
    rebco_st = {}
    for arch in ["gru", "transformer"]:
        cp = base / f"rebco_stab_seq_{arch}.pt"
        if cp.exists():
            a, s = load_and_predict(str(cp), T0, lam, env_kwargs={"mode": "stab"})
            rebco_st[arch] = s
    if rebco_st:
        plot_rebco_3d_comparison("REBCO Cu-SS stab", T0, lam, rebco_st, out_dir)

    # Bi-2223 Ag-stab (1-param: T₀ only, fixed cooling)
    print("=== Bi-2223 Ag-stab ===")
    T0_bi_ag = np.linspace(77.26, 79.0, 30)
    lam_dummy = np.array([100.0])
    bi_ag = {}
    for arch in ["gru", "transformer", "ssm"]:
        cp = base / f"bi2223_agstab_seq_{arch}.pt"
        if cp.exists():
            try:
                a, s = load_and_predict(str(cp), T0_bi_ag, lam_dummy,
                                        env_kwargs={"mode": "ag_stab"})
                bi_ag[arch] = s
            except RuntimeError as e:
                print(f"  Skipping {arch}: {e}")
                continue
    if bi_ag:
        plot_bi2223_curves("Bi-2223 Ag-stab", T0_bi_ag, bi_ag, out_dir,
                           nonlinear_band=(78.0, 79.0))

    # Bi-2223 no-stab (1-param)
    print("=== Bi-2223 no-stab ===")
    T0_bi_ns = np.linspace(77.5, 81.0, 30)
    bi_ns = {}
    for arch in ["gru", "transformer", "ssm"]:
        cp = base / f"bi2223_nostab_seq_{arch}.pt"
        if cp.exists():
            try:
                a, s = load_and_predict(str(cp), T0_bi_ns, lam_dummy,
                                        env_kwargs={"mode": "no_stab"})
                bi_ns[arch] = s
            except RuntimeError as e:
                print(f"  Skipping {arch}: {e}")
                continue
    if bi_ns:
        plot_bi2223_curves("Bi-2223 no-stab", T0_bi_ns, bi_ns, out_dir,
                           nonlinear_band=(79.5, 81.0))

    print("\n=== Done! All figures in", out_dir, "===")


if __name__ == "__main__":
    main()
