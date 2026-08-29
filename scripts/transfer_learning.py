#!/usr/bin/env python
"""
transfer_learning.py

Cross-conductor transfer learning: train a sequence surrogate on one
conductor family and evaluate on another. Tests whether surrogates learn
generalizable thermal-runaway physics or just memorize conductor-specific
patterns.

Transfer pairs:
  - Bi-2223 no-stab → REBCO no-stab  (and reverse)
  - Bi-2223 no-stab → REBCO stab     (and reverse)
  - REBCO no-stab ↔ REBCO stab       (within-family mode transfer)

Usage:
    # Quick smoke test
    python scripts/transfer_learning.py --scale quick

    # Production
    python scripts/transfer_learning.py --scale production

    # Specific pair
    python scripts/transfer_learning.py --pairs rebco_nostab:bi2223_nostab \
        --archs gru transformer --scale quick
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hts_tape_env.seq_surrogates import train_seq_surrogate, predict_episode
from hts_tape_env.seq_data import TRUE_FEATURES
from hts_tape_env.rollout import generate_rollout_dataset as rebco_rollout
from hts_tape_env.rollout_bi2223_env import generate_rollout_dataset as bi2223_rollout

CONDUCTOR_CONFIGS = {
    "rebco_nostab": {
        "rollout_fn": rebco_rollout,
        "rollout_kwargs": {"env_kwargs": {"mode": "no_stab"}},
        "label": "REBCO no-stab",
        "T0_range": (10, 75),
        "solve_fn": lambda T0: __import__("hts_tape_env.physics", fromlist=["solve_runaway"]).solve_runaway(T0, 20, mode="no_stab", lam_override=100.0),
    },
    "rebco_stab": {
        "rollout_fn": rebco_rollout,
        "rollout_kwargs": {"env_kwargs": {"mode": "stab"}},
        "label": "REBCO Cu/SS stab",
        "T0_range": (10, 75),
        "solve_fn": lambda T0: __import__("hts_tape_env.physics", fromlist=["solve_runaway"]).solve_runaway(T0, 20, mode="stab", lam_override=100.0),
    },
    "bi2223_nostab": {
        "rollout_fn": bi2223_rollout,
        "rollout_kwargs": {"env_kwargs": {"mode": "no_stab"}},
        "label": "Bi-2223 no-stab",
        "T0_range": (77.16, 81.0),
        "solve_fn": lambda T0: __import__("hts_tape_env.physics_bi2223", fromlist=["solve_runaway"]).solve_runaway(T0),
    },
    "bi2223_agstab": {
        "rollout_fn": bi2223_rollout,
        "rollout_kwargs": {"env_kwargs": {"mode": "ag_stab", "T0_range": (77.16, 79.0)}},
        "label": "Bi-2223 Ag-stab",
        "T0_range": (77.16, 79.0),
        "solve_fn": lambda T0: __import__("hts_tape_env.physics_bi2223", fromlist=["solve_runaway"]).solve_runaway(T0, mode="ag_stab"),
    },
}

DEFAULT_PAIRS = [
    ("rebco_nostab", "bi2223_nostab"),
    ("bi2223_nostab", "rebco_nostab"),
    ("rebco_nostab", "rebco_stab"),
    ("rebco_stab", "rebco_nostab"),
    ("bi2223_nostab", "rebco_stab"),
    ("rebco_stab", "bi2223_nostab"),
]

SCALE_PARAMS = {
    "quick": {"n_episodes": 50, "epochs": 10},
    "medium": {"n_episodes": 300, "epochs": 60},
    "production": {"n_episodes": 1000, "epochs": 200},
}


def generate_data(conductor: str, n_episodes: int, seed: int) -> pd.DataFrame:
    cfg = CONDUCTOR_CONFIGS[conductor]
    print(f"  Generating {n_episodes} episodes for {cfg['label']}...")
    return cfg["rollout_fn"](n_episodes, seed=seed, **cfg["rollout_kwargs"])


def evaluate_transfer(
    result,
    df_target: pd.DataFrame,
    conductor_target: str,
) -> dict:
    """Evaluate a trained surrogate on target conductor's trajectory data."""
    cfg = CONDUCTOR_CONFIGS[conductor_target]
    solve_fn = cfg["solve_fn"]

    Istar_preds = []
    Istar_true = []
    T0_vals = []

    for ep_id in sorted(df_target["episode_id"].unique()):
        ep = df_target[df_target["episode_id"] == ep_id]
        feats = result.meta.get("features", list(TRUE_FEATURES))
        X = ep[feats].values
        preds = predict_episode(result, X)

        T0 = ep["T0_K"].iloc[0]
        Is_analytic, _ = solve_fn(T0)
        if not np.isfinite(Is_analytic) or Is_analytic <= 0:
            continue

        Istar_preds.append(preds["istar_pred_A"])
        Istar_true.append(Is_analytic)
        T0_vals.append(T0)

    if not Istar_preds:
        return {"n_episodes": 0, "istar_mae_A": np.nan, "istar_median_rel_err": np.nan}

    pred = np.array(Istar_preds)
    true = np.array(Istar_true)
    T0s = np.array(T0_vals)

    abs_err = np.abs(pred - true)
    rel_err = abs_err / np.maximum(true, 1e-6)

    return {
        "n_episodes": len(pred),
        "istar_mae_A": float(np.mean(abs_err)),
        "istar_median_rel_err": float(np.median(rel_err)),
        "istar_95th_rel_err": float(np.percentile(rel_err, 95)),
        "istar_max_rel_err": float(np.max(rel_err)),
        "T0_range": (float(T0s.min()), float(T0s.max())),
        "istar_range_true": (float(true.min()), float(true.max())),
        "istar_range_pred": (float(pred.min()), float(pred.max())),
    }


def plot_transfer_matrix(results: list[dict], out_dir: Path):
    """Heatmap of I* median relative error for each (source, target, arch) combo."""
    df = pd.DataFrame(results)
    if df.empty:
        return

    archs = df["arch"].unique()

    for arch in archs:
        arch_df = df[df["arch"] == arch]
        sources = sorted(arch_df["source"].unique())
        targets = sorted(arch_df["target"].unique())

        matrix = np.full((len(sources), len(targets)), np.nan)
        for _, row in arch_df.iterrows():
            i = sources.index(row["source"])
            j = targets.index(row["target"])
            matrix[i, j] = row.get("istar_median_rel_err", np.nan)

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=1.0)

        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(sources)))
        ax.set_yticklabels(sources, fontsize=9)
        ax.set_xlabel("Target (evaluate on)", fontsize=10)
        ax.set_ylabel("Source (trained on)", fontsize=10)

        for i in range(len(sources)):
            for j in range(len(targets)):
                val = matrix[i, j]
                if np.isfinite(val):
                    color = "white" if val > 0.5 else "black"
                    txt = f"{val:.1%}"
                    if sources[i] == targets[j]:
                        txt = f"{val:.1%}\n(baseline)"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=8, color=color, fontweight="bold" if sources[i] == targets[j] else "normal")

        fig.colorbar(im, ax=ax, label="I* median rel err")
        ax.set_title(f"Cross-conductor transfer: {arch}", fontsize=11)
        fig.tight_layout()
        path = out_dir / f"transfer_matrix_{arch}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


def plot_transfer_bar(results: list[dict], out_dir: Path):
    """Grouped bar chart: baseline vs transfer for each pair."""
    df = pd.DataFrame(results)
    if df.empty:
        return

    transfer_only = df[df["source"] != df["target"]]
    if transfer_only.empty:
        return

    baselines = df[df["source"] == df["target"]].set_index(["target", "arch"])["istar_median_rel_err"]

    fig, ax = plt.subplots(figsize=(12, 5))
    pairs = []
    for _, row in transfer_only.iterrows():
        pair_label = f"{row['source']}→{row['target']}\n({row['arch']})"
        baseline_key = (row["target"], row["arch"])
        baseline_val = baselines.get(baseline_key, np.nan)
        pairs.append({
            "label": pair_label,
            "transfer": row["istar_median_rel_err"],
            "baseline": baseline_val,
        })

    pdf = pd.DataFrame(pairs)
    x = np.arange(len(pdf))
    width = 0.35

    bars1 = ax.bar(x - width / 2, pdf["baseline"], width, label="Within-conductor baseline",
                   color="tab:green", alpha=0.7)
    bars2 = ax.bar(x + width / 2, pdf["transfer"], width, label="Cross-conductor transfer",
                   color="tab:red", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(pdf["label"], fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("I* median rel err")
    ax.set_title("Transfer learning: baseline vs cross-conductor")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = out_dir / "transfer_comparison.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", default="quick", choices=list(SCALE_PARAMS.keys()))
    ap.add_argument("--archs", nargs="+", default=["gru", "transformer"],
                    help="Architectures to test")
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="Transfer pairs as source:target (e.g., rebco_nostab:bi2223_nostab)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path("models"))
    args = ap.parse_args()

    params = SCALE_PARAMS[args.scale]
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.pairs:
        pairs = [tuple(p.split(":")) for p in args.pairs]
    else:
        pairs = DEFAULT_PAIRS

    all_conductors = set()
    for src, tgt in pairs:
        all_conductors.add(src)
        all_conductors.add(tgt)

    # Add within-conductor baselines
    for c in list(all_conductors):
        if (c, c) not in pairs:
            pairs.append((c, c))

    print(f"=== Cross-conductor transfer learning ({args.scale}) ===")
    print(f"  Pairs: {len(pairs)}, archs: {args.archs}")
    print(f"  Episodes: {params['n_episodes']}, epochs: {params['epochs']}")

    # Generate data for all conductors involved
    print("\n=== Generating trajectory data ===")
    data_cache = {}
    for conductor in sorted(all_conductors):
        try:
            data_cache[conductor] = generate_data(
                conductor, params["n_episodes"], seed=args.seed,
            )
        except Exception as e:
            print(f"  {conductor}: FAILED — {e}")

    # Train and evaluate all pairs
    print("\n=== Training and evaluating ===")
    all_results = []
    model_cache = {}

    for source, target in pairs:
        if source not in data_cache or target not in data_cache:
            continue

        for arch in args.archs:
            cache_key = (source, arch)
            tag = f"{source}→{target}" if source != target else f"{source} (baseline)"
            print(f"\n  {tag} [{arch}]")

            # Train (or reuse cached model)
            if cache_key not in model_cache:
                try:
                    if arch == "diffusion":
                        from hts_tape_env.diffusion import train_diffusion_surrogate
                        result = train_diffusion_surrogate(
                            data_cache[source], epochs=params["epochs"],
                            seed=args.seed, feature_cols=TRUE_FEATURES,
                        )
                    else:
                        result = train_seq_surrogate(
                            data_cache[source], arch=arch, epochs=params["epochs"],
                            seed=args.seed, feature_cols=TRUE_FEATURES,
                        )
                    model_cache[cache_key] = result
                    fname = f"transfer_{source}_{arch}.pt"
                    result.save(out / fname)
                    print(f"    Trained on {source}, saved {fname}")
                except Exception as e:
                    print(f"    Training FAILED: {e}")
                    continue
            else:
                result = model_cache[cache_key]
                print(f"    Using cached model trained on {source}")

            # Evaluate on target
            metrics = evaluate_transfer(result, data_cache[target], target)
            metrics["source"] = source
            metrics["target"] = target
            metrics["arch"] = arch
            all_results.append(metrics)

            istar_err = metrics.get("istar_median_rel_err", np.nan)
            print(f"    → I* median rel err on {target}: {istar_err:.2%}"
                  f" ({metrics['n_episodes']} episodes)")

    # Summary and plots
    if not all_results:
        print("\nNo results to report.")
        return

    print("\n=== Results ===")
    df_results = pd.DataFrame(all_results)

    print(f"\n{'='*80}")
    print("  Cross-conductor transfer: I* median relative error")
    print(f"{'='*80}")
    print(f"  {'Source':<18} {'Target':<18} {'Arch':<13} {'I* med rel err':>14}")
    print("  " + "-" * 65)
    for _, row in df_results.sort_values(["source", "target", "arch"]).iterrows():
        marker = " *" if row["source"] == row["target"] else ""
        print(f"  {row['source']:<18} {row['target']:<18} {row['arch']:<13} "
              f"{row['istar_median_rel_err']:>14.2%}{marker}")
    print("  (* = within-conductor baseline)")
    print(f"{'='*80}\n")

    print("=== Generating plots ===")
    plot_transfer_matrix(all_results, out)
    plot_transfer_bar(all_results, out)

    json_path = out / "transfer_learning_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else None)
    print(f"  Saved {json_path}")

    csv_path = out / "transfer_learning_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")


if __name__ == "__main__":
    main()
