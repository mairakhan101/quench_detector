"""
seq_surrogates.py

Sequence-model surrogates for thermal-runaway prediction: given the telemetry
trajectory (T0, I, V, T_surface) produced while ramping current in
QuenchDetectorEnv, predict all three runaway *surfaces* from Marchevsky &
Prestemon (2024) from the trajectory shape alone -- not just I*(T0,lambda)
(Fig. 2/3), but also T*/deltaT (also Fig. 2/3) and the adiabatic time margin
(Fig. 4). The model never sees lambda: it must infer the effective cooling
from how fast V and T_surface rise as I ramps -- exactly what a real
protection loop has to do. For now this trains/evaluates on the *noiseless*
(ground-truth) channels (seq_data.TRUE_FEATURES) -- isolating the
sequence-to-surface mapping from instrument noise. Pass
feature_cols=seq_data.FEATURES to instead train on the noisy channels the
agent actually observes, once that's the thing being worked on.

Each model has five heads:
    margin          [B,T] running (I*-I)/I* at every step -- Fig. 3 in miniature.
    breach_logit    [B,T] logit of "margin_t < dump_threshold" -- trained
                    directly with BCE against the actual decision boundary,
                    rather than inferred indirectly from the margin regression
                    (margin regression alone is dominated by the easy, safely-
                    below-threshold majority of the trajectory; this head is
                    what a controller should actually condition on).
    istar_log       [B] log I* from the final valid hidden state -- Fig. 2/3.
    tstar           [B] T* from the final valid hidden state -- also Fig. 2/3
                    (report as deltaT = tstar - T0 if that's the more natural
                    paper quantity).
    time_margin_log [B] log(adiabatic time-to-300K) from the final valid
                    hidden state -- Fig. 4. Some (T0,lambda) cells have no
                    finite time margin (thermal.time_margin_to_safe's t_max
                    isn't reached); those episodes are masked out of this
                    loss term only (seq_data.py's time_margin_valid).

margin_loss_weighting="boundary" (the default) reweights the per-step margin
MSE by exp(-|margin|/tau): steps near the decision boundary count more than
the many easy steps at the start of a ramp, which is where a *uniform*-weight
loss spends most of its gradient even though the controller decision only
happens in a narrow band.

Two architectures, same five-head output:
    GRUSurrogate          nn.GRU encoder (recommended default: cheap, causal
                          by construction, a good fit for streaming telemetry)
    TransformerSurrogate  causal nn.TransformerEncoder, for comparison

Usage
-----
    from hts_tape_env.rollout import generate_rollout_dataset
    from hts_tape_env.seq_surrogates import train_seq_surrogate

    df = generate_rollout_dataset(1500, seed=7)
    result = train_seq_surrogate(df, arch="gru")
    print(result.metrics)
    result.save("models/seq_gru.pt")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from hts_tape_env.seq_data import (
    TRUE_FEATURES,
    EpisodeSequenceDataset,
    Normalizer,
    collate_episodes,
)


# ── Models ──────────────────────────────────────────────────────────────────

class GRUSurrogate(nn.Module):
    def __init__(self, n_features: int = 4, hidden: int = 64, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0)
        self.margin_head = nn.Linear(hidden, 1)
        self.breach_head = nn.Linear(hidden, 1)
        self.istar_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.tstar_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.time_margin_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, X: torch.Tensor, lengths: torch.Tensor) -> dict:
        packed = nn.utils.rnn.pack_padded_sequence(
            X, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, h_n = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True, total_length=X.shape[1])
        h_last = h_n[-1]  # [B, hidden] final layer's final hidden state
        return {
            "margin": self.margin_head(out).squeeze(-1),               # [B, T]
            "breach_logit": self.breach_head(out).squeeze(-1),         # [B, T]
            "istar_log": self.istar_head(h_last).squeeze(-1),          # [B]
            "tstar": self.tstar_head(h_last).squeeze(-1),              # [B]
            "time_margin_log": self.time_margin_head(h_last).squeeze(-1),  # [B]
        }


class TransformerSurrogate(nn.Module):
    """Causal Transformer encoder over the same telemetry sequence."""

    def __init__(self, n_features: int = 4, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1,
                 max_len: int = 1024):
        super().__init__()
        self.in_proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.margin_head = nn.Linear(d_model, 1)
        self.breach_head = nn.Linear(d_model, 1)
        self.istar_head = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1))
        self.tstar_head = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1))
        self.time_margin_head = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1))

    def forward(self, X: torch.Tensor, lengths: torch.Tensor) -> dict:
        B, T, _ = X.shape
        h = self.in_proj(X) + self.pos[:, :T]
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=X.device), diagonal=1)
        key_padding_mask = torch.arange(T, device=X.device)[None, :] >= lengths[:, None].to(X.device)
        out = self.encoder(h, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        idx = (lengths - 1).clamp(min=0).to(X.device)
        h_last = out[torch.arange(B, device=X.device), idx]     # last *valid* timestep per sequence
        return {
            "margin": self.margin_head(out).squeeze(-1),
            "breach_logit": self.breach_head(out).squeeze(-1),
            "istar_log": self.istar_head(h_last).squeeze(-1),
            "tstar": self.tstar_head(h_last).squeeze(-1),
            "time_margin_log": self.time_margin_head(h_last).squeeze(-1),
        }


class S4DBlock(nn.Module):
    """Single diagonal state-space layer (S4D-Lin variant).

    Recurrence: h_t = A * h_{t-1} + B * x_t,  y_t = Re(C * h_t) + D * x_t
    where A is a complex diagonal initialized on the negative real half-plane
    (HiPPO-inspired). Pure PyTorch, no custom CUDA kernels.
    """

    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        log_dt = torch.rand(d_model) * (np.log(0.1) - np.log(0.001)) + np.log(0.001)
        self.log_dt = nn.Parameter(log_dt)
        A_real = -0.5 * torch.ones(d_model, d_state)
        A_imag = torch.arange(d_state).float().unsqueeze(0).expand(d_model, -1) * np.pi
        self.A_real = nn.Parameter(A_real)
        self.A_imag = nn.Parameter(A_imag)
        self.B_re = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.B_im = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.C_re = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.C_im = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.D = nn.Parameter(torch.ones(d_model))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, T, d_model] -> [B, T, d_model]"""
        B, T, _ = x.shape
        dt = self.log_dt.exp()
        A = torch.complex(self.A_real, self.A_imag)
        dA = torch.exp(A * dt.unsqueeze(-1))           # [d_model, d_state]
        B_c = torch.complex(self.B_re, self.B_im)
        dB = B_c * dt.unsqueeze(-1)                    # [d_model, d_state]
        C_c = torch.complex(self.C_re, self.C_im)

        h = torch.zeros(B, self.d_model, self.d_state, dtype=torch.cfloat, device=x.device)
        ys = []
        for t in range(T):
            h = dA.unsqueeze(0) * h + dB.unsqueeze(0) * x[:, t, :].unsqueeze(-1).to(torch.cfloat)
            y_t = (C_c.unsqueeze(0) * h).sum(-1).real + self.D.unsqueeze(0) * x[:, t]
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        return self.norm(self.dropout(y) + x)


class SSMSurrogate(nn.Module):
    """State-space model surrogate using stacked diagonal S4D blocks."""

    def __init__(self, n_features: int = 4, hidden: int = 64, n_layers: int = 2,
                 d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(n_features, hidden)
        self.blocks = nn.ModuleList([S4DBlock(hidden, d_state, dropout) for _ in range(n_layers)])
        self.margin_head = nn.Linear(hidden, 1)
        self.breach_head = nn.Linear(hidden, 1)
        self.istar_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.tstar_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.time_margin_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, X: torch.Tensor, lengths: torch.Tensor) -> dict:
        B, T, _ = X.shape
        h = self.in_proj(X)
        for block in self.blocks:
            h = block(h, lengths)
        idx = (lengths - 1).clamp(min=0).to(X.device)
        h_last = h[torch.arange(B, device=X.device), idx]
        return {
            "margin": self.margin_head(h).squeeze(-1),
            "breach_logit": self.breach_head(h).squeeze(-1),
            "istar_log": self.istar_head(h_last).squeeze(-1),
            "tstar": self.tstar_head(h_last).squeeze(-1),
            "time_margin_log": self.time_margin_head(h_last).squeeze(-1),
        }


def build_model(arch: str, **kwargs) -> nn.Module:
    if arch == "gru":
        return GRUSurrogate(**kwargs)
    if arch == "transformer":
        return TransformerSurrogate(**kwargs)
    if arch == "ssm":
        return SSMSurrogate(**kwargs)
    raise KeyError(f"unknown arch '{arch}', expected 'gru', 'transformer', or 'ssm'")


# ── Train / evaluate / persist ────────────────────────────────────────────────

@dataclass
class SeqTrainResult:
    model: nn.Module
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


def load_seq_surrogate(path: str | Path, device: str = "cpu") -> SeqTrainResult:
    bundle = torch.load(Path(path), map_location=device, weights_only=False)
    model = build_model(bundle["arch"], **bundle["meta"].get("model_kwargs", {}))
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    normalizer = Normalizer(mean=bundle["norm_mean"], std=bundle["norm_std"])
    return SeqTrainResult(model=model, normalizer=normalizer, arch=bundle["arch"],
                           metrics=bundle["metrics"], meta=bundle["meta"])


def _masked_mse(pred, target, weight):
    """weight is 0 on padded timesteps and (optionally) boundary-scaled elsewhere."""
    err2 = (pred - target) ** 2 * weight
    return err2.sum() / weight.sum().clamp(min=1e-8)


def train_seq_surrogate(
    df,
    *,
    arch: str = "gru",
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    margin_weight: float = 1.0,
    breach_weight: float = 1.0,
    istar_weight: float = 1.0,
    tstar_weight: float = 0.0,
    time_margin_weight: float = 0.3,
    margin_loss_weighting: str = "boundary",
    boundary_tau: float = 0.05,
    dump_threshold: float = 0.05,
    val_frac: float = 0.2,
    seed: int = 42,
    device: str | None = None,
    model_kwargs: dict | None = None,
    feature_cols=TRUE_FEATURES,
    verbose: bool = True,
) -> SeqTrainResult:
    """
    Train a sequence surrogate on a rollout dataframe (rollout.generate_rollout_dataset).

    feature_cols defaults to the noiseless ground-truth channels
    (seq_data.TRUE_FEATURES) -- for now the model is trained/evaluated on the
    true trajectory, isolating the sequence-to-I* mapping from instrument
    noise. Pass seq_data.FEATURES to train on the noisy instrument channels
    instead.

    margin_loss_weighting="boundary" (default) weights each step's margin MSE
    by exp(-|margin|/boundary_tau), so the loss actually reflects the narrow
    band a controller decides in, instead of being dominated by the easy
    early-ramp majority. Pass "uniform" for the original unweighted version.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    full = EpisodeSequenceDataset(df, feature_cols=feature_cols, dump_threshold=dump_threshold)
    n_val = max(1, int(val_frac * len(full)))
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_episodes)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_episodes)

    model_kwargs = model_kwargs or {}
    model = build_model(arch, **model_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def compute_loss(batch: dict) -> torch.Tensor:
        X, mask = batch["X"].to(device), batch["mask"].to(device)
        margin, breach = batch["margin"].to(device), batch["breach"].to(device)
        istar, tstar = batch["istar_log"].to(device), batch["tstar"].to(device)
        tm_log, tm_valid = batch["time_margin_log"].to(device), batch["time_margin_valid"].to(device)
        lengths = batch["lengths"]

        out = model(X, lengths)
        mask_f = mask.float()

        if margin_loss_weighting == "boundary":
            step_weight = mask_f * torch.exp(-margin.abs() / boundary_tau)
        elif margin_loss_weighting == "uniform":
            step_weight = mask_f
        else:
            raise KeyError(f"unknown margin_loss_weighting '{margin_loss_weighting}'")

        loss_margin = _masked_mse(out["margin"], margin, step_weight)
        loss_breach = (bce(out["breach_logit"], breach) * mask_f).sum() / mask_f.sum().clamp(min=1e-8)
        loss_istar = nn.functional.mse_loss(out["istar_log"], istar)
        loss_tstar = nn.functional.mse_loss(out["tstar"], tstar)
        loss_tm = ((out["time_margin_log"] - tm_log) ** 2 * tm_valid).sum() / tm_valid.sum().clamp(min=1e-8)

        return (margin_weight * loss_margin + breach_weight * loss_breach
                + istar_weight * loss_istar + tstar_weight * loss_tstar
                + time_margin_weight * loss_tm)

    def run_epoch(dl, train: bool) -> float:
        model.train(train)
        total, n_batches = 0.0, 0
        for batch in dl:
            with torch.set_grad_enabled(train):
                loss = compute_loss(batch)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item()
            n_batches += 1
        return total / max(n_batches, 1)

    best_val, best_state = float("inf"), None
    for epoch in range(epochs):
        train_loss = run_epoch(train_dl, train=True)
        val_loss = run_epoch(val_dl, train=False)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Held-out metrics in physical units, across all three surfaces + the
    # breach classifier (the thing controllers actually condition on).
    istar_true_all, istar_pred_all = [], []
    tstar_true_all, tstar_pred_all = [], []
    tm_true_all, tm_pred_all, tm_valid_all = [], [], []
    breach_true_all, breach_prob_all, breach_mask_all = [], [], []
    margin_near_true, margin_near_pred = [], []  # steps with |true margin| < 2*boundary_tau
    with torch.no_grad():
        for batch in val_dl:
            X, lengths = batch["X"].to(device), batch["lengths"]
            out = model(X, lengths)
            istar_true_all.append(batch["istar_log"].numpy())
            istar_pred_all.append(out["istar_log"].cpu().numpy())
            tstar_true_all.append(batch["tstar"].numpy())
            tstar_pred_all.append(out["tstar"].cpu().numpy())
            tm_true_all.append(batch["time_margin_log"].numpy())
            tm_pred_all.append(out["time_margin_log"].cpu().numpy())
            tm_valid_all.append(batch["time_margin_valid"].numpy())

            mask_np = batch["mask"].numpy()
            breach_true_all.append(batch["breach"].numpy()[mask_np])
            breach_prob_all.append(torch.sigmoid(out["breach_logit"]).cpu().numpy()[mask_np])

            margin_np = batch["margin"].numpy()
            near = (np.abs(margin_np) < 2 * boundary_tau) & mask_np
            margin_near_true.append(margin_np[near])
            margin_near_pred.append(out["margin"].cpu().numpy()[near])

    istar_true = np.exp(np.concatenate(istar_true_all))
    istar_pred = np.exp(np.concatenate(istar_pred_all))
    istar_mae = float(np.mean(np.abs(istar_true - istar_pred)))
    istar_rel = float(np.median(np.abs(istar_true - istar_pred) / istar_true))

    tstar_mae = float(np.mean(np.abs(np.concatenate(tstar_true_all) - np.concatenate(tstar_pred_all))))

    tm_valid_all = np.concatenate(tm_valid_all).astype(bool)
    tm_true = np.exp(np.concatenate(tm_true_all))[tm_valid_all]
    tm_pred = np.exp(np.concatenate(tm_pred_all))[tm_valid_all]
    tm_rel = float(np.median(np.abs(tm_pred - tm_true) / tm_true)) if tm_valid_all.any() else float("nan")

    breach_true = np.concatenate(breach_true_all)
    breach_prob = np.concatenate(breach_prob_all)
    breach_acc = float(((breach_prob > 0.5).astype(np.float32) == breach_true).mean())

    margin_near_true = np.concatenate(margin_near_true)
    margin_near_pred = np.concatenate(margin_near_pred)
    margin_near_mae = (float(np.mean(np.abs(margin_near_true - margin_near_pred)))
                        if len(margin_near_true) else float("nan"))

    metrics = {
        "best_val_loss": best_val,
        "istar_mae_A": istar_mae,
        "istar_median_rel_err": istar_rel,
        "tstar_mae_K": tstar_mae,
        "time_margin_median_rel_err": tm_rel,
        "breach_accuracy": breach_acc,
        "margin_near_boundary_mae": margin_near_mae,
    }
    if verbose:
        print(f"I* held-out:        MAE={istar_mae:.2f} A, median rel err={istar_rel:.2%}")
        print(f"T* held-out:        MAE={tstar_mae:.2f} K")
        print(f"time-margin:        median rel err={tm_rel:.2%} ({tm_valid_all.sum()} valid episodes)")
        print(f"breach classifier:  accuracy={breach_acc:.2%}")
        print(f"margin MAE within 2*tau of the boundary: {margin_near_mae:.4f}")

    return SeqTrainResult(
        model=model, normalizer=full.normalizer, arch=arch, metrics=metrics,
        meta={"model_kwargs": model_kwargs, "features": full.feature_cols,
              "dump_threshold": dump_threshold},
    )


# ── Deployment adapter (wires into QuenchDetectorEnv.set_surrogate) ────────────

class SequenceSurrogateAdapter:
    """
    Stateful wrapper around a trained SeqTrainResult for streaming, one-step-
    at-a-time inference inside QuenchDetectorEnv -- the object you pass to
    env.set_surrogate().

    The env still generates ground-truth (T_surface, voltage) from the
    analytic solver every step (it has no real conductor to measure); what
    this adapter replaces is the *decision* step a real controller would
    otherwise need brentq for: turning the live (T0, I, V, T_surface) reading
    into an I*/margin/breach estimate. That's the actual deployment role of
    the surrogate in Marchevsky & Prestemon's proposed protection loop.

    For arch='gru' this is true streaming: the GRU hidden state is carried
    across calls, so each predict_step() is a single-timestep forward pass
    (O(1), not O(episode length)). For arch='transformer' there is no
    incremental hidden state, so the adapter replays the buffered history
    each call (O(t) at step t); pass `max_history` to cap that if long
    episodes make it too slow.

    Usage
    -----
        result = load_seq_surrogate("models/seq_gru.pt")
        env.set_surrogate(SequenceSurrogateAdapter(result))
    """

    def __init__(self, result: SeqTrainResult, *, risk_scale: float = 0.05,
                 device: str = "cpu", max_history: int | None = None):
        self.model = result.model.to(device).eval()
        self.normalizer = result.normalizer
        self.arch = result.arch
        self.risk_scale = risk_scale
        self.device = device
        self.max_history = max_history
        self._hidden = None    # GRU path: carried recurrent state
        self._buffer: list[np.ndarray] = []  # Transformer path: replayed history

    def reset(self) -> None:
        """Clear state at the start of a new episode. Call from env.reset()."""
        self._hidden = None
        self._buffer = []

    @torch.no_grad()
    def predict_step(self, obs: np.ndarray) -> dict:
        """
        obs : array-like [T0, I, V, T_surface] -- the env's current observation.
        Returns margin_pred, I_star_pred, T_star_pred, time_margin_pred_ms,
        breach_prob_pred (direct P(margin < dump_threshold), the classifier
        head -- prefer this over thresholding margin_pred/risk_pred), risk_pred.
        """
        x = self.normalizer.transform(np.asarray(obs, dtype=np.float32).reshape(1, 4))[0]

        if self.arch == "gru":
            xt = torch.from_numpy(x).view(1, 1, 4).to(self.device)
            out, self._hidden = self.model.gru(xt, self._hidden)
            margin_pred = float(self.model.margin_head(out).view(-1)[0])
            breach_logit = float(self.model.breach_head(out).view(-1)[0])
            h_last = self._hidden[-1]
            istar_log = float(self.model.istar_head(h_last).view(-1)[0])
            tstar_pred = float(self.model.tstar_head(h_last).view(-1)[0])
            tm_log = float(self.model.time_margin_head(h_last).view(-1)[0])
        else:
            self._buffer.append(x)
            if self.max_history:
                self._buffer = self._buffer[-self.max_history:]
            Xseq = torch.from_numpy(np.stack(self._buffer)).unsqueeze(0).to(self.device)
            lengths = torch.tensor([Xseq.shape[1]])
            out = self.model(Xseq, lengths)
            margin_pred = float(out["margin"][0, -1])
            breach_logit = float(out["breach_logit"][0, -1])
            istar_log = float(out["istar_log"][0])
            tstar_pred = float(out["tstar"][0])
            tm_log = float(out["time_margin_log"][0])

        istar_pred = float(np.exp(istar_log))
        breach_prob = float(1.0 / (1.0 + np.exp(-breach_logit)))
        risk_pred = float(1.0 / (1.0 + np.exp(margin_pred / self.risk_scale)))
        return {
            "margin_pred": margin_pred,
            "I_star_pred": istar_pred,
            "T_star_pred": tstar_pred,
            "time_margin_pred_ms": float(np.exp(tm_log)),
            "breach_prob_pred": breach_prob,
            "risk_pred": risk_pred,
        }


# ── Runaway-surface evaluation ─────────────────────────────────────────────

def predict_episode(result: SeqTrainResult, X: np.ndarray) -> dict:
    """
    Offline (non-streaming) inference over one already-complete trajectory --
    the shared building block behind predict_runaway_surface() below and
    scripts/fem_surrogate_inference.py / scripts/plot_surrogate_bi2223.py,
    which run a trained surrogate against trajectories it never trained on
    (FEM sweeps, physics_bi2223 rollouts) rather than QuenchDetectorEnv's own.

    X : [T, F] raw (unnormalized) feature array, columns matching
        result.meta["features"] (defaults to seq_data.TRUE_FEATURES).

    Returns per-step margin_pred/breach_prob [T] plus final istar_pred_A/
    tstar_pred_K/time_margin_pred_ms from the last valid hidden state --
    same quantities SequenceSurrogateAdapter.predict_step returns, but from
    one batched forward pass over the whole sequence instead of streaming.
    """
    Xn = result.normalizer.transform(np.asarray(X, dtype=np.float32))
    Xt = torch.from_numpy(Xn).unsqueeze(0)
    lengths = torch.tensor([Xn.shape[0]])
    with torch.no_grad():
        out = result.model(Xt, lengths)

    breach_prob = torch.sigmoid(out["breach_logit"][0]).numpy()
    margin_pred = out["margin"][0].numpy()
    alarm_idx = np.where(breach_prob > 0.5)[0]

    return dict(
        margin_pred=margin_pred,
        breach_prob=breach_prob,
        first_alarm_step=int(alarm_idx[0]) if len(alarm_idx) else None,
        istar_pred_A=float(np.exp(out["istar_log"].item())),
        tstar_pred_K=float(out["tstar"].item()),
        time_margin_pred_ms=float(np.exp(out["time_margin_log"].item())),
    )


def predict_runaway_surface(
    result: SeqTrainResult,
    T0_grid: np.ndarray,
    lam_grid: np.ndarray,
    *,
    env_kwargs: dict | None = None,
    device: str | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """
    For every (T0, lambda) grid point, roll out one full episode under that
    exact condition (env.reset_at) and ask the trained model for all three
    surfaces from the *entire* trajectory. Mirrors the notebook's Fig. 2/3/4
    solver-vs-surrogate comparisons, except the model never sees lambda --
    only the telemetry shape it produces.

    Returns a dict of arrays, each shaped [len(lam_grid), len(T0_grid)]
    (meshgrid convention: rows=lam, cols=T0), NaN where no interior runaway
    exists at that grid point:
        Istar_analytic, Istar_pred
        Tstar_analytic, Tstar_pred
        time_margin_analytic_ms, time_margin_pred_ms
    """
    from hts_tape_env.env import QuenchDetectorEnv
    from hts_tape_env.rollout import rollout_episode

    device = device or "cpu"
    model = result.model.to(device)
    model.eval()

    env = QuenchDetectorEnv(**(env_kwargs or {}))
    n_lam, n_t0 = len(lam_grid), len(T0_grid)
    surfaces = {k: np.full((n_lam, n_t0), np.nan) for k in
                ("Istar_analytic", "Istar_pred", "Tstar_analytic", "Tstar_pred",
                 "time_margin_analytic_ms", "time_margin_pred_ms")}
    feature_cols = result.meta.get("features", TRUE_FEATURES)

    ep = 0
    for i, T0 in enumerate(T0_grid):
        for j, lam in enumerate(lam_grid):
            obs, info = env.reset_at(float(T0), float(lam), seed=seed + ep)
            ep += 1
            if not np.isfinite(info["I_star"]) or info["I_star"] <= 0:
                continue
            rows = rollout_episode(env, ep, obs=obs, info=info)
            X = np.array([[r[c] for c in feature_cols] for r in rows], dtype=np.float32)
            X = result.normalizer.transform(X)
            Xt = torch.from_numpy(X).unsqueeze(0).to(device)          # [1, T, 4]
            lengths = torch.tensor([X.shape[0]])
            with torch.no_grad():
                out = model(Xt, lengths)

            surfaces["Istar_analytic"][j, i] = info["I_star"]
            surfaces["Istar_pred"][j, i] = float(np.exp(out["istar_log"].cpu().item()))
            surfaces["Tstar_analytic"][j, i] = info["T_star"]
            surfaces["Tstar_pred"][j, i] = float(out["tstar"].cpu().item())
            surfaces["time_margin_analytic_ms"][j, i] = rows[0]["time_margin_ms"]
            surfaces["time_margin_pred_ms"][j, i] = float(np.exp(out["time_margin_log"].cpu().item()))

    return surfaces
