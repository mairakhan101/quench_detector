"""
train_fem_runaway.py

Train GRU / Transformer / S4D models on the FEA no-heater sweep to predict,
at every 10 ms step, the time to runaway onset and the probability that
runaway starts within --horizon seconds (see hts_tape_env/runaway_seq.py).
Train/val/test are contiguous I/Ic blocks (--block-width), so held-out runs
are never near-duplicates of training runs.

Usage
-----
    python scripts/train_fem_runaway.py
    python scripts/train_fem_runaway.py --archs gru --epochs 50
    python scripts/train_fem_runaway.py --features voltage_asinh T_surface_K --tag _no_current
"""

import _bootstrap  # noqa: F401  (first import: caps BLAS threads, fixes sys.path)

import argparse
import json
from pathlib import Path

from hts_tape_env.fem_runaway_data import DEFAULT_PATH, DT_S, T_RUNAWAY_K, load_runaway_sweep
from hts_tape_env.runaway_seq import (
    ARCHS,
    DEFAULT_FEATURES,
    add_model_features,
    evaluate_runaway_model,
    split_by_ratio_blocks,
    train_runaway_model,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=str(DEFAULT_PATH))
    ap.add_argument("--dt", type=float, default=DT_S, help="resampling grid step [s]")
    ap.add_argument("--threshold-k", type=float, default=T_RUNAWAY_K, help="runaway onset Tmax [K]")
    ap.add_argument("--horizon", type=float, default=0.5, help="runaway-probability horizon [s]")
    ap.add_argument("--archs", nargs="+", choices=ARCHS, default=list(ARCHS))
    ap.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    ap.add_argument("--block-width", type=float, default=0.02, help="I/Ic width of split blocks")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    args = ap.parse_args()

    df = add_model_features(load_runaway_sweep(args.path, dt_s=args.dt, threshold_K=args.threshold_k))
    splits = split_by_ratio_blocks(df, block_width=args.block_width)
    for name, part in splits.items():
        ep = part.groupby("episode_id").first()
        print(f"{name:<5} {len(ep):4d} runs ({int(ep['runaway'].sum())} runaway), "
              f"I/Ic {ep['ratio'].min():.3f}-{ep['ratio'].max():.3f}")
    print(f"features: {args.features}, horizon: {args.horizon} s\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    for arch in args.archs:
        print(f"=== {arch} ===")
        result = train_runaway_model(
            splits["train"], splits["val"], arch=arch, feature_cols=args.features,
            horizon_s=args.horizon, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, patience=args.patience, seed=args.seed)
        test = evaluate_runaway_model(result, splits["test"])
        result.meta.update(dt_s=args.dt, threshold_K=args.threshold_k, data_path=args.path,
                           block_width=args.block_width)
        ckpt = result.save(out_dir / f"fem_runaway_{arch}{args.tag}.pt")
        test["per_episode"].to_csv(out_dir / f"fem_runaway_{arch}{args.tag}_test_episodes.csv", index=False)
        all_metrics[arch] = {"val": result.metrics, "test": test["metrics"],
                             "stopped_epoch": result.meta["stopped_epoch"]}
        metrics_path = out_dir / f"fem_runaway_{arch}{args.tag}_metrics.json"
        metrics_path.write_text(json.dumps({"args": vars(args), **all_metrics[arch]}, indent=2))
        print(f"saved {ckpt}, {metrics_path}\n")

    metric_names = list(next(iter(all_metrics.values()))["test"])
    print(f"{'test metric':<32}" + "".join(f"{a:>13}" for a in all_metrics))
    for m in metric_names:
        print(f"{m:<32}" + "".join(f"{all_metrics[a]['test'][m]:>13.4g}" for a in all_metrics))


if __name__ == "__main__":
    main()
