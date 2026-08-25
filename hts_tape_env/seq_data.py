"""
seq_data.py

Turn a long-format rollout table (rollout.py) into padded tensors for
sequence-model training: one sequence per episode_id, padded to each batch's
own max length, with a boolean mask marking real (non-pad) timesteps.

Targets cover all three runaway surfaces from Marchevsky & Prestemon (2024),
not just I*: margin/I* (Fig. 2/3), T* (also Fig. 2/3, as deltaT = T*-T0), and
the adiabatic time margin (Fig. 4). See seq_surrogates.py for the model heads
that consume these.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

# Noisy instrument channels (env's obs vector) -- what a deployed agent
# actually sees. Kept for future noise-robustness work.
FEATURES = ["T0_K", "I_A", "voltage_V", "T_surface_K"]

# Noiseless ground-truth channels (env's info dict). This is the default for
# now: it isolates the sequence-to-I* mapping from instrument noise, so
# training/eval reflect what the model can learn from trajectory shape alone,
# not noise-floor limitations. Swap to FEATURES to train the noise-robust
# version once that's the thing being worked on.
TRUE_FEATURES = ["T0_K", "I_true_A", "voltage_true_V", "T_surface_true_K"]


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, df, columns=TRUE_FEATURES) -> "Normalizer":
        X = df[columns].to_numpy(dtype=np.float32)
        return cls(mean=X.mean(axis=0), std=X.std(axis=0) + 1e-6)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


class EpisodeSequenceDataset(Dataset):
    """
    One item per episode: telemetry [T, 4] plus per-step and per-episode
    targets covering all three runaway surfaces (I*, T*/deltaT, time margin).

    Sequences are kept ragged here; padding happens in collate_episodes so a
    batch only pads to its own max length rather than the dataset's global max.
    """

    def __init__(self, df, normalizer: Normalizer | None = None, log_istar: bool = True,
                 feature_cols=TRUE_FEATURES, dump_threshold: float = 0.05):
        self.feature_cols = list(feature_cols)
        self.normalizer = normalizer or Normalizer.fit(df, columns=self.feature_cols)
        self.log_istar = log_istar
        self.dump_threshold = dump_threshold
        self.episodes: list[tuple] = []
        n_no_time_margin = 0
        for _, g in df.sort_values("step").groupby("episode_id", sort=False):
            X = self.normalizer.transform(g[self.feature_cols].to_numpy(dtype=np.float32))
            margin = g["margin_norm"].to_numpy(dtype=np.float32)
            breach = (margin < dump_threshold).astype(np.float32)  # per-step "in the danger band"

            istar = float(g["Istar_A"].iloc[0])
            if self.log_istar:
                istar = float(np.log(max(istar, 1e-6)))
            tstar = float(g["Tstar_K"].iloc[0])

            tm_ms = float(g["time_margin_ms"].iloc[0])
            tm_valid = np.isfinite(tm_ms)
            if not tm_valid:
                n_no_time_margin += 1
            tm_log = float(np.log(max(tm_ms, 1e-6))) if tm_valid else 0.0

            self.episodes.append((X, margin, breach, istar, tstar, tm_log, tm_valid))
        if n_no_time_margin:
            print(f"  (EpisodeSequenceDataset: {n_no_time_margin}/{len(self.episodes)} episodes "
                  f"have no finite time_margin -- excluded from that loss term only)")

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int):
        X, margin, breach, istar, tstar, tm_log, tm_valid = self.episodes[idx]
        return (
            torch.from_numpy(X),
            torch.from_numpy(margin),
            torch.from_numpy(breach),
            torch.tensor(istar, dtype=torch.float32),
            torch.tensor(tstar, dtype=torch.float32),
            torch.tensor(tm_log, dtype=torch.float32),
            torch.tensor(float(tm_valid), dtype=torch.float32),
        )


def collate_episodes(batch) -> dict:
    """Pad a batch of ragged episodes to the batch's own max length; returns a dict of tensors."""
    lengths = torch.tensor([item[0].shape[0] for item in batch], dtype=torch.long)
    T = int(lengths.max())
    B = len(batch)
    Xp = torch.zeros(B, T, batch[0][0].shape[1])
    Mp = torch.zeros(B, T)
    Bp = torch.zeros(B, T)  # breach label per step
    mask = torch.zeros(B, T, dtype=torch.bool)
    istar = torch.zeros(B)
    tstar = torch.zeros(B)
    tm_log = torch.zeros(B)
    tm_valid = torch.zeros(B)
    for i, (X, margin, breach, ist, tst, tml, tmv) in enumerate(batch):
        t = X.shape[0]
        Xp[i, :t] = X
        Mp[i, :t] = margin
        Bp[i, :t] = breach
        mask[i, :t] = True
        istar[i], tstar[i], tm_log[i], tm_valid[i] = ist, tst, tml, tmv
    return dict(X=Xp, margin=Mp, breach=Bp, mask=mask, lengths=lengths,
                istar_log=istar, tstar=tstar, time_margin_log=tm_log, time_margin_valid=tm_valid)
