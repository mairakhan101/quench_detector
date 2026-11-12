"""
runaway_seq.py

Sequence models that predict thermal runaway from FEA telemetry
(fem_runaway_data.load_runaway_sweep): at every step, the time remaining until
runaway onset and the probability that runaway starts within horizon_s.
Reuses the GRU / Transformer / S4D backbones from seq_surrogates.py; only
their per-step heads are trained (the per-episode I*/T* heads stay unused).

Targets and losses (assumptions explicit):
- ttr_log = log(time_to_runaway_s + TTR_OFFSET_S); the offset (one 10 ms grid
  step) keeps the log finite at onset. Runaway episodes: MSE. Censored
  episodes only bound the answer from below (true time > t_end_s - t_s), so
  only predictions under that bound are penalized (squared hinge).
- runaway_within_h = 1 if onset is within horizon_s. In censored episodes a 0
  label is known only while >= horizon_s of the observed window remains;
  later steps are masked out of the loss and metrics.
- Voltage enters as asinh(V / V_REF_V), V_REF_V = 1 uV: V spans 0-0.26 V and
  the precursor signal sits at ~0.3-6 mV in the last second before onset,
  which a linear z-score would flatten to ~0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from hts_tape_env.seq_data import Normalizer
from hts_tape_env.seq_surrogates import build_model

TTR_OFFSET_S = 0.01
V_REF_V = 1e-6
DEFAULT_FEATURES = ["I_A", "voltage_asinh", "T_surface_K"]
ARCHS = ("gru", "transformer", "ssm")


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(voltage_asinh=np.arcsinh(df["voltage_V"] / V_REF_V))


def split_by_ratio_blocks(df: pd.DataFrame, block_width: float = 0.02, n_folds: int = 5,
                          val_fold: int = 1, test_fold: int = 3) -> dict[str, pd.DataFrame]:
    """Assign contiguous I/Ic blocks round-robin to train/val/test, so held-out
    runs are never 0.001 Ic neighbours of training runs."""
    block = np.floor((df["ratio"] - df["ratio"].min()) / block_width + 1e-9).astype(int)
    fold = block % n_folds
    return {
        "train": df[(fold != val_fold) & (fold != test_fold)],
        "val": df[fold == val_fold],
        "test": df[fold == test_fold],
    }


class RunawayEpisodeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, normalizer: Normalizer | None = None,
                 feature_cols=DEFAULT_FEATURES, horizon_s: float = 0.5):
        self.feature_cols = list(feature_cols)
        self.normalizer = normalizer or Normalizer.fit(df, columns=self.feature_cols)
        self.horizon_s = horizon_s
        self.episodes: list[dict] = []
        for episode_id, g in df.sort_values("step").groupby("episode_id", sort=True):
            t = g["t_s"].to_numpy()
            censored = not bool(g["runaway"].iloc[0])
            if censored:
                ttr = g["t_end_s"].iloc[0] - t
                y = np.zeros_like(t)
                y_valid = ttr >= horizon_s
            else:
                ttr = g["time_to_runaway_s"].to_numpy()
                y = (ttr <= horizon_s).astype(float)
                y_valid = np.ones_like(t, dtype=bool)
            self.episodes.append(dict(
                episode_id=int(episode_id),
                ratio=float(g["ratio"].iloc[0]),
                censored=censored,
                X=self.normalizer.transform(g[self.feature_cols].to_numpy(dtype=np.float32)),
                t_s=t.astype(np.float32),
                ttr_s=ttr.astype(np.float32),
                ttr_log=np.log(ttr + TTR_OFFSET_S).astype(np.float32),
                y=y.astype(np.float32),
                y_valid=y_valid.astype(np.float32),
            ))

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict:
        return self.episodes[idx]


_PER_STEP = ("t_s", "ttr_s", "ttr_log", "y", "y_valid")


def collate_runaway(batch: list[dict]) -> dict:
    lengths = torch.tensor([len(e["X"]) for e in batch], dtype=torch.long)
    B, T = len(batch), int(lengths.max())
    X = torch.zeros(B, T, batch[0]["X"].shape[1])
    mask = torch.zeros(B, T, dtype=torch.bool)
    per_step = {k: torch.zeros(B, T) for k in _PER_STEP}
    for i, e in enumerate(batch):
        n = len(e["X"])
        X[i, :n] = torch.from_numpy(e["X"])
        mask[i, :n] = True
        for k in _PER_STEP:
            per_step[k][i, :n] = torch.from_numpy(e[k])
    censored = torch.tensor([e["censored"] for e in batch], dtype=torch.bool)
    return dict(X=X, mask=mask, lengths=lengths, censored=censored, **per_step)


class RunawayModel(nn.Module):
    """Backbone from seq_surrogates.build_model; its per-step margin/breach
    heads are trained as log time-to-runaway / runaway-within-horizon logit."""

    def __init__(self, arch: str, n_features: int, **backbone_kwargs):
        super().__init__()
        self.backbone = build_model(arch, n_features=n_features, **backbone_kwargs)

    def forward(self, X: torch.Tensor, lengths: torch.Tensor) -> dict:
        out = self.backbone(X, lengths)
        return {"ttr_log": out["margin"], "runaway_logit": out["breach_logit"]}


def runaway_loss(out: dict, batch: dict, ttr_weight: float = 1.0,
                 prob_weight: float = 1.0) -> torch.Tensor:
    mask = batch["mask"].float()
    cens = batch["censored"][:, None].float() * mask
    obs = mask - cens
    err = out["ttr_log"] - batch["ttr_log"]
    ttr_sq = obs * err ** 2 + cens * F.relu(-err) ** 2
    loss_ttr = ttr_sq.sum() / mask.sum().clamp(min=1.0)

    valid = batch["y_valid"] * mask
    bce = F.binary_cross_entropy_with_logits(out["runaway_logit"], batch["y"], reduction="none")
    loss_prob = (bce * valid).sum() / valid.sum().clamp(min=1.0)
    return ttr_weight * loss_ttr + prob_weight * loss_prob


@dataclass
class RunawayTrainResult:
    model: RunawayModel
    normalizer: Normalizer
    arch: str
    metrics: dict
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "arch": self.arch,
            "norm_mean": self.normalizer.mean,
            "norm_std": self.normalizer.std,
            "metrics": self.metrics,
            "meta": self.meta,
        }, path)
        return path


def load_runaway_model(path: str | Path, device: str = "cpu") -> RunawayTrainResult:
    bundle = torch.load(Path(path), map_location=device, weights_only=False)
    meta = bundle["meta"]
    model = RunawayModel(bundle["arch"], len(meta["features"]), **meta["model_kwargs"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return RunawayTrainResult(
        model=model, normalizer=Normalizer(mean=bundle["norm_mean"], std=bundle["norm_std"]),
        arch=bundle["arch"], metrics=bundle["metrics"], meta=meta)


def train_runaway_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    arch: str = "gru",
    feature_cols=DEFAULT_FEATURES,
    horizon_s: float = 0.5,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    ttr_weight: float = 1.0,
    prob_weight: float = 1.0,
    patience: int = 20,
    min_delta: float = 1e-4,
    reduce_lr_patience: int = 8,
    seed: int = 42,
    device: str | None = None,
    model_kwargs: dict | None = None,
    verbose: bool = True,
) -> RunawayTrainResult:
    """Early-stops on val loss; metrics on val_df are attached to the result."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model_kwargs = dict(model_kwargs or {})

    train_ds = RunawayEpisodeDataset(train_df, feature_cols=feature_cols, horizon_s=horizon_s)
    val_ds = RunawayEpisodeDataset(val_df, normalizer=train_ds.normalizer,
                                   feature_cols=feature_cols, horizon_s=horizon_s)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_runaway,
                          generator=torch.Generator().manual_seed(seed))
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_runaway)

    model = RunawayModel(arch, len(feature_cols), **model_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=reduce_lr_patience, factor=0.5, min_lr=1e-6)

    def run_epoch(dl, train: bool) -> float:
        model.train(train)
        total, n = 0.0, 0
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.set_grad_enabled(train):
                loss = runaway_loss(model(batch["X"], batch["lengths"]), batch,
                                    ttr_weight, prob_weight)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item()
            n += 1
        return total / max(n, 1)

    best_val, best_state, since_best = float("inf"), None, 0
    stopped_epoch = epochs - 1
    for epoch in range(epochs):
        train_loss = run_epoch(train_dl, train=True)
        val_loss = run_epoch(val_dl, train=False)
        scheduler.step(val_loss)
        if val_loss < best_val - min_delta:
            best_val, since_best = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if patience > 0 and since_best >= patience:
            stopped_epoch = epoch
            if verbose:
                print(f"  early stopped at epoch {epoch} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval().cpu()

    result = RunawayTrainResult(
        model=model, normalizer=train_ds.normalizer, arch=arch, metrics={},
        meta={"features": list(feature_cols), "horizon_s": horizon_s,
              "model_kwargs": model_kwargs, "ttr_offset_s": TTR_OFFSET_S,
              "v_ref_v": V_REF_V, "best_val_loss": best_val,
              "stopped_epoch": stopped_epoch},
    )
    result.metrics = evaluate_runaway_model(result, val_df)["metrics"]
    return result


def predict_episodes(result: RunawayTrainResult, df: pd.DataFrame) -> list[dict]:
    """Per-episode dataset fields plus prob (runaway within horizon) and
    ttr_pred_s, as numpy arrays over the episode's steps."""
    ds = RunawayEpisodeDataset(df, normalizer=result.normalizer,
                               feature_cols=result.meta["features"],
                               horizon_s=result.meta["horizon_s"])
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_runaway)
    preds = []
    with torch.no_grad():
        for batch in dl:
            out = result.model(batch["X"], batch["lengths"])
            prob = torch.sigmoid(out["runaway_logit"]).numpy()
            ttr_pred = (torch.exp(out["ttr_log"]) - TTR_OFFSET_S).numpy()
            for i, n in enumerate(batch["lengths"].tolist()):
                preds.append(dict(prob=prob[i, :n], ttr_pred_s=ttr_pred[i, :n]))
    return [{**e, **p} for e, p in zip(ds.episodes, preds)]


def evaluate_runaway_model(result: RunawayTrainResult, df: pd.DataFrame,
                           alarm_threshold: float = 0.5, near_s: float = 1.0) -> dict:
    """
    Per-step: AUROC/Brier of runaway-within-horizon on known labels; time-to-
    runaway MAE over the last near_s before onset and median relative error
    over all observed steps; fraction of censored steps predicted below their
    lower bound. Per-episode: an alarm is the first step with prob >=
    alarm_threshold. Runaway runs report detection rate and warning time
    (onset - alarm); censored runs count a false alarm only if it fires while
    >= horizon_s of observed safe window remains.
    """
    horizon_s = result.meta["horizon_s"]
    episodes = predict_episodes(result, df)

    y, p = [], []
    near_err, rel_err, bound_viol = [], [], []
    rows = []
    for e in episodes:
        valid = e["y_valid"] > 0
        y.append(e["y"][valid])
        p.append(e["prob"][valid])
        alarm_idx = np.flatnonzero(e["prob"] >= alarm_threshold)
        first = int(alarm_idx[0]) if len(alarm_idx) else None
        if e["censored"]:
            bound_viol.append(e["ttr_pred_s"] < e["ttr_s"])
            false_alarm = bool(np.any((e["prob"] >= alarm_threshold) & valid))
            rows.append(dict(episode_id=e["episode_id"], ratio=e["ratio"], runaway=0,
                             alarm_t_s=None if first is None else float(e["t_s"][first]),
                             warning_s=np.nan, false_alarm=false_alarm,
                             evaluable=bool(valid.any())))
        else:
            near = e["ttr_s"] <= near_s
            near_err.append(np.abs(e["ttr_pred_s"][near] - e["ttr_s"][near]))
            rel_err.append(np.abs(e["ttr_pred_s"] - e["ttr_s"]) / (e["ttr_s"] + TTR_OFFSET_S))
            rows.append(dict(episode_id=e["episode_id"], ratio=e["ratio"], runaway=1,
                             alarm_t_s=None if first is None else float(e["t_s"][first]),
                             warning_s=np.nan if first is None else float(e["ttr_s"][first]),
                             false_alarm=False, evaluable=True))

    y, p = np.concatenate(y), np.concatenate(p)
    per_episode = pd.DataFrame(rows)
    run = per_episode[per_episode["runaway"] == 1]
    cens = per_episode[per_episode["runaway"] == 0]
    # A censored run shorter than horizon_s has no known-safe step to false-alarm on.
    cens_eval = cens[cens["evaluable"]]
    warnings = run["warning_s"].dropna()

    def _cat(chunks):
        return np.concatenate(chunks) if chunks else np.array([])

    near_err, rel_err, bound_viol = _cat(near_err), _cat(rel_err), _cat(bound_viol)
    metrics = {
        "n_runaway_episodes": int(len(run)),
        "n_censored_episodes": int(len(cens)),
        "n_censored_evaluable": int(len(cens_eval)),
        "prob_auroc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "prob_brier": float(np.mean((p - y) ** 2)),
        f"ttr_mae_s_last_{near_s:g}s": float(near_err.mean()) if len(near_err) else float("nan"),
        "ttr_median_rel_err": float(np.median(rel_err)) if len(rel_err) else float("nan"),
        "censored_bound_violation_rate": float(bound_viol.mean()) if len(bound_viol) else float("nan"),
        "detection_rate": float(run["warning_s"].notna().mean()) if len(run) else float("nan"),
        "median_warning_s": float(warnings.median()) if len(warnings) else float("nan"),
        "p10_warning_s": float(warnings.quantile(0.1)) if len(warnings) else float("nan"),
        "early_alarm_rate": float((warnings > horizon_s).sum() / len(run)) if len(run) else float("nan"),
        "false_alarm_rate": float(cens_eval["false_alarm"].mean()) if len(cens_eval) else float("nan"),
    }
    return {"metrics": metrics, "per_episode": per_episode}
