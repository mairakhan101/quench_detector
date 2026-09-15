#!/usr/bin/env python
"""
diffusion_map_analysis.py

Diffusion-map analysis of the thermal runaway parameter space across all
conductor families.  Computes the Coifman–Lafon diffusion map on the
(T0, I, [λ]) operating-point manifold and produces:

  1. Eigenvalue spectrum — reveals intrinsic dimensionality
  2. 2D embedding colored by I*, margin, runaway — shows manifold geometry
  3. Local sampling density vs distance to runaway boundary — guides where
     to add training points
  4. Cross-conductor comparison in shared diffusion coordinates
"""

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity


# ── Diffusion map core ───────────────────────────────────────────────────────

def diffusion_map(X, n_components=6, alpha=1.0, epsilon="auto"):
    """
    Compute the diffusion map embedding of X.

    Parameters
    ----------
    X : (n, d) array, already standardized
    n_components : number of diffusion coordinates to keep (excludes trivial)
    alpha : normalization exponent (1.0 = Laplace-Beltrami on the density)
    epsilon : kernel bandwidth; "auto" = median of squared distances

    Returns
    -------
    coords : (n, n_components) diffusion coordinates
    lambdas : (n_components,) eigenvalues (diffusion rates)
    epsilon_used : float, the bandwidth that was used
    """
    D2 = squareform(pdist(X, "sqeuclidean"))

    if epsilon == "auto":
        epsilon = float(np.median(D2[D2 > 0]))

    K = np.exp(-D2 / epsilon)

    # alpha-normalization (removes density bias)
    q = K.sum(axis=1)
    K_alpha = K / np.outer(q ** alpha, q ** alpha)

    # Row-stochastic normalization → Markov matrix
    d = K_alpha.sum(axis=1)
    D_inv = np.diag(1.0 / d)
    M = D_inv @ K_alpha

    # Symmetric similarity for stable eigendecomposition
    D_half = np.diag(np.sqrt(d))
    D_inv_half = np.diag(1.0 / np.sqrt(d))
    Ms = D_half @ M @ D_inv_half

    # Eigendecomposition (top n_components+1, skip trivial λ=1)
    n_eig = min(n_components + 1, len(X) - 1)
    vals, vecs = eigh(Ms)
    idx = np.argsort(-vals)
    vals = vals[idx]
    vecs = vecs[:, idx]

    # Convert back to right eigenvectors of M
    psi = D_inv_half @ vecs

    # Skip the trivial constant eigenvector
    coords = psi[:, 1:n_components + 1]
    lambdas = vals[1:n_components + 1]

    return coords, lambdas, epsilon


def local_density(X, bandwidth="scott"):
    """KDE-based local density estimate on standardized features."""
    kde = KernelDensity(bandwidth=0.3, kernel="gaussian")
    kde.fit(X)
    log_dens = kde.score_samples(X)
    return np.exp(log_dens)


# ── Data generators ──────────────────────────────────────────────────────────

def generate_rebco_data(mode, n_samples=3000, seed=42):
    from hts_tape_env.datagen import generate_dataset
    return generate_dataset(
        n_samples, mode=mode, lam_range=(20, 200),
        seed=seed, verbose=False,
    )


def generate_bi2223_data(mode, n_samples=3000, seed=42):
    from hts_tape_env.datagen_bi2223 import generate_dataset
    kwargs = dict(n_samples=n_samples, seed=seed, verbose=False)
    if mode == "no_stab":
        kwargs["T0_range"] = (77.16, 81.0)
    else:
        kwargs["T0_range"] = (77.16, 79.0)
        kwargs["mode"] = "ag_stab"
        from hts_tape_env.thermal_bi2223 import set_rho_multiplier
        set_rho_multiplier(10.0)
    return generate_dataset(**kwargs)


CONDUCTORS = {
    "rebco_nostab": dict(gen=generate_rebco_data, gen_kw=dict(mode="no_stab"),
                         features=["T0_K", "I_A", "lam"],
                         has_lambda=True),
    "rebco_stab":   dict(gen=generate_rebco_data, gen_kw=dict(mode="stab"),
                         features=["T0_K", "I_A", "lam"],
                         has_lambda=True),
    "bi2223_nostab": dict(gen=generate_bi2223_data, gen_kw=dict(mode="no_stab"),
                          features=["T0_K", "I_A"],
                          has_lambda=False),
    "bi2223_agstab": dict(gen=generate_bi2223_data, gen_kw=dict(mode="ag_stab"),
                          features=["T0_K", "I_A"],
                          has_lambda=False),
}


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_eigenvalue_spectrum(all_lambdas, out_dir):
    """Eigenvalue decay → intrinsic dimensionality."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, lam in all_lambdas.items():
        ax.plot(range(1, len(lam) + 1), lam, "o-", label=name, markersize=5)
    ax.set_xlabel("Diffusion coordinate index")
    ax.set_ylabel("Eigenvalue λ_k")
    ax.set_title("Diffusion Map Eigenvalue Spectrum — Intrinsic Dimensionality")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "diffmap_eigenvalue_spectrum.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / 'diffmap_eigenvalue_spectrum.png'}")


def plot_embeddings(name, coords, df, features, out_dir):
    """2D diffusion embedding colored by I*, margin, and runaway."""
    phi1, phi2 = coords[:, 0], coords[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Color by I*
    sc0 = axes[0].scatter(phi1, phi2, c=df["Istar_A"], s=4, alpha=0.6,
                          cmap="viridis")
    axes[0].set_title("Colored by I* (A)")
    plt.colorbar(sc0, ax=axes[0], shrink=0.8)

    # Color by margin
    vmin, vmax = df["margin_norm"].quantile(0.02), df["margin_norm"].quantile(0.98)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=min(vmin, -0.05), vmax=max(vmax, 0.05))
    sc1 = axes[1].scatter(phi1, phi2, c=df["margin_norm"], s=4, alpha=0.6,
                          cmap="RdYlGn", norm=norm)
    axes[1].set_title("Colored by margin (I*−I)/I*")
    plt.colorbar(sc1, ax=axes[1], shrink=0.8)

    # Color by runaway
    colors = np.where(df["runaway"] == 1, "red", "steelblue")
    axes[2].scatter(phi1, phi2, c=colors, s=4, alpha=0.5)
    axes[2].set_title("Runaway (red) vs safe (blue)")

    for ax in axes:
        ax.set_xlabel("ψ₁")
        ax.set_ylabel("ψ₂")
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"Diffusion Map Embedding — {name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f"diffmap_embedding_{name}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / f'diffmap_embedding_{name}.png'}")


def plot_density_vs_boundary(name, X_std, df, out_dir):
    """Sampling density vs distance to runaway boundary."""
    dens = local_density(X_std)
    margin = df["margin_norm"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Density vs margin
    axes[0].scatter(margin, dens, s=3, alpha=0.4, c="teal")
    axes[0].set_xlabel("Margin (I*−I)/I*")
    axes[0].set_ylabel("Local sampling density")
    axes[0].set_title(f"{name}: Density vs margin to boundary")
    axes[0].axvline(0, color="red", ls="--", lw=1, label="runaway boundary")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Density histogram split by runaway
    safe_dens = dens[df["runaway"].values == 0]
    unsafe_dens = dens[df["runaway"].values == 1]
    axes[1].hist(safe_dens, bins=40, alpha=0.6, label="safe", color="steelblue",
                 density=True)
    axes[1].hist(unsafe_dens, bins=40, alpha=0.6, label="runaway", color="red",
                 density=True)
    axes[1].set_xlabel("Local sampling density")
    axes[1].set_ylabel("Frequency (normalized)")
    axes[1].set_title(f"{name}: Density distribution by class")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"diffmap_density_{name}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / f'diffmap_density_{name}.png'}")


def plot_boundary_geometry(name, coords, df, out_dir):
    """Diffusion coordinates along the runaway boundary — reveals curvature."""
    margin = df["margin_norm"].values
    near_boundary = np.abs(margin) < 0.05
    if near_boundary.sum() < 20:
        near_boundary = np.abs(margin) < 0.15

    phi1 = coords[:, 0]
    phi2 = coords[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Full embedding with boundary highlighted
    axes[0].scatter(phi1, phi2, s=2, alpha=0.15, c="gray", label="all")
    axes[0].scatter(phi1[near_boundary], phi2[near_boundary], s=8, alpha=0.7,
                    c=df["T0_K"].values[near_boundary], cmap="plasma")
    axes[0].set_xlabel("ψ₁")
    axes[0].set_ylabel("ψ₂")
    axes[0].set_title(f"{name}: Runaway boundary in diffusion coords\n"
                      f"(colored by T₀)")
    axes[0].grid(True, alpha=0.2)

    # ψ₁ vs T0 near boundary — shows how the first diffusion coordinate
    # tracks temperature
    axes[1].scatter(df["T0_K"].values[near_boundary],
                    phi1[near_boundary], s=8, alpha=0.6, c="teal")
    axes[1].set_xlabel("T₀ (K)")
    axes[1].set_ylabel("ψ₁ (first diffusion coordinate)")
    axes[1].set_title(f"{name}: ψ₁ vs T₀ near boundary (|margin|<5%)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"diffmap_boundary_{name}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / f'diffmap_boundary_{name}.png'}")


def plot_cross_conductor_summary(all_results, out_dir):
    """Summary figure: eigenvalue gap + boundary curvature across conductors."""
    names = list(all_results.keys())
    n = len(names)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: eigenvalue ratios (spectral gap)
    ax = axes[0, 0]
    for name in names:
        lam = all_results[name]["lambdas"]
        ratios = lam[:-1] / lam[1:]
        ax.plot(range(1, len(ratios) + 1), ratios, "o-", label=name, markersize=5)
    ax.set_xlabel("Gap index k")
    ax.set_ylabel("λ_k / λ_{k+1}")
    ax.set_title("Spectral gap ratios — larger gap = dimension break")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: effective dimensionality (participation ratio)
    ax = axes[0, 1]
    dims = []
    for name in names:
        lam = all_results[name]["lambdas"]
        pr = (lam.sum()) ** 2 / (lam ** 2).sum()
        dims.append(pr)
    ax.bar(range(n), dims, color=["steelblue", "coral", "seagreen", "goldenrod"][:n])
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Participation ratio")
    ax.set_title("Effective dimensionality of runaway manifold")
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-left: fraction near boundary vs density
    ax = axes[1, 0]
    for i, name in enumerate(names):
        df = all_results[name]["df"]
        margin = df["margin_norm"].values
        near = np.abs(margin) < 0.05
        frac_near = near.mean()
        ax.bar(i, frac_near,
               color=["steelblue", "coral", "seagreen", "goldenrod"][i])
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of samples within 5% of boundary")
    ax.set_title("Sampling concentration near runaway boundary")
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-right: ψ₁ range near boundary (manifold extent)
    ax = axes[1, 1]
    for i, name in enumerate(names):
        coords = all_results[name]["coords"]
        df = all_results[name]["df"]
        margin = df["margin_norm"].values
        near = np.abs(margin) < 0.10
        if near.sum() > 10:
            psi1_range = coords[near, 0].max() - coords[near, 0].min()
        else:
            psi1_range = 0
        ax.bar(i, psi1_range,
               color=["steelblue", "coral", "seagreen", "goldenrod"][i])
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("ψ₁ range along boundary")
    ax.set_title("Manifold extent near runaway boundary")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Cross-Conductor Diffusion Map Summary", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "diffmap_cross_conductor_summary.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'diffmap_cross_conductor_summary.png'}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=3000,
                    help="Samples per conductor (datagen)")
    ap.add_argument("--n-components", type=int, default=6,
                    help="Diffusion coordinates to compute")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Run only these conductors")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    conductors = CONDUCTORS
    if args.only:
        conductors = {k: v for k, v in conductors.items() if k in args.only}

    all_results = {}
    all_lambdas = {}

    for name, cfg in conductors.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        # Generate data
        print(f"  Generating {args.n_samples} samples...")
        df = cfg["gen"](**cfg["gen_kw"], n_samples=args.n_samples, seed=args.seed)
        print(f"  Got {len(df)} valid rows, {df['runaway'].mean():.0%} unsafe")

        if len(df) < 100:
            print(f"  SKIP — too few rows")
            continue

        # Standardize features for diffusion kernel
        features = cfg["features"]
        X = df[features].values
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        # Subsample if large (diffusion map is O(n²))
        max_n = 2000
        if len(X_std) > max_n:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(X_std), max_n, replace=False)
            X_sub = X_std[idx]
            df_sub = df.iloc[idx].reset_index(drop=True)
        else:
            X_sub = X_std
            df_sub = df.reset_index(drop=True)

        # Compute diffusion map
        print(f"  Computing diffusion map ({len(X_sub)} points, "
              f"{len(features)}D)...")
        coords, lambdas, eps = diffusion_map(
            X_sub, n_components=args.n_components,
        )
        print(f"  ε = {eps:.3f}, eigenvalues: {lambdas[:4].round(4)}")

        all_results[name] = dict(
            coords=coords, lambdas=lambdas, df=df_sub,
            X_std=X_sub, features=features, epsilon=eps,
        )
        all_lambdas[name] = lambdas

        # Per-conductor plots
        print("  Generating plots...")
        plot_embeddings(name, coords, df_sub, features, out)
        plot_density_vs_boundary(name, X_sub, df_sub, out)
        plot_boundary_geometry(name, coords, df_sub, out)

    # Cross-conductor comparison
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  Cross-conductor summary")
        print(f"{'='*60}")
        plot_eigenvalue_spectrum(all_lambdas, out)
        plot_cross_conductor_summary(all_results, out)

    # Print summary table
    print("\n" + "="*70)
    print("DIFFUSION MAP SUMMARY")
    print("="*70)
    print(f"{'Conductor':<20} {'Rows':>6} {'Dim':>4} {'ε':>8} "
          f"{'λ₁':>8} {'λ₂':>8} {'λ₁/λ₂':>8} {'Eff.dim':>8}")
    print("-"*70)
    for name, r in all_results.items():
        lam = r["lambdas"]
        pr = lam.sum()**2 / (lam**2).sum()
        print(f"{name:<20} {len(r['df']):>6} {len(r['features']):>4} "
              f"{r['epsilon']:>8.3f} {lam[0]:>8.4f} {lam[1]:>8.4f} "
              f"{lam[0]/lam[1]:>8.2f} {pr:>8.2f}")
    print("="*70)
    print(f"\nPlots saved to {out}/diffmap_*.png")


if __name__ == "__main__":
    main()
