"""
diffusion.py

Conditional DDPM (Denoising Diffusion Probabilistic Model) surrogate for
thermal-runaway prediction. Generates the margin trajectory conditioned on
the operating point, then extracts the 5 standard prediction heads for
benchmark-compatible evaluation.

Architecture: 1D temporal U-Net (ConditionalUNet1D) conditioned on (T0, lambda)
or telemetry summary statistics. The DDPM forward process corrupts the
target margin trajectory with Gaussian noise; the reverse process learns
to denoise. At inference, a margin trajectory is sampled from pure noise
via iterative denoising, and the 5 heads are extracted from pooled
features and the generated trajectory.

Reference: Ho et al., "Denoising Diffusion Probabilistic Models" (2020);
RapDI (diffusion-based PDE surrogates) for the conditional generation approach.

Usage:
    from hts_tape_env.diffusion import train_diffusion_surrogate

    result = train_diffusion_surrogate(df, epochs=100, seed=42)
    result.save("models/seq_diffusion.pt")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from hts_tape_env.seq_data import (
    TRUE_FEATURES,
    EpisodeSequenceDataset,
    Normalizer,
    collate_episodes,
)


# ── Noise schedule ───────────────────────────────────────────────────────────

def cosine_beta_schedule(n_timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.linspace(0, n_timesteps, n_timesteps + 1)
    alpha_bar = torch.cos(((steps / n_timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(max=0.999)


def linear_beta_schedule(n_timesteps: int, beta_start: float = 1e-4,
                          beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, n_timesteps)


# ── U-Net building blocks ───────────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConditionalResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.cond_proj = nn.Linear(cond_dim, out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(F.silu(self.conv1(x)))
        h = h + self.cond_proj(cond).unsqueeze(-1) + self.time_proj(t_emb).unsqueeze(-1)
        h = self.norm2(F.silu(self.conv2(h)))
        return h + self.skip(x)


class ConditionalUNet1D(nn.Module):
    def __init__(self, in_ch: int = 1, cond_dim: int = 2,
                 channels: tuple[int, ...] = (32, 64, 128),
                 time_dim: int = 64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        self.down_blocks = nn.ModuleList()
        self.down_pools = nn.ModuleList()
        ch_prev = in_ch
        for ch in channels:
            self.down_blocks.append(ConditionalResBlock1D(ch_prev, ch, cond_dim, time_dim))
            self.down_pools.append(nn.Conv1d(ch, ch, 3, stride=2, padding=1))
            ch_prev = ch

        self.mid = ConditionalResBlock1D(channels[-1], channels[-1], cond_dim, time_dim)

        self.up_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i, ch in enumerate(reversed(channels)):
            ch_in = channels[-1] if i == 0 else channels[len(channels) - i]
            self.up_convs.append(nn.ConvTranspose1d(ch_in, ch, 4, stride=2, padding=1))
            self.up_blocks.append(ConditionalResBlock1D(ch * 2, ch, cond_dim, time_dim))

        self.out_conv = nn.Conv1d(channels[0], in_ch, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        skips = []
        h = x
        for block, pool in zip(self.down_blocks, self.down_pools):
            h = block(h, cond, t_emb)
            skips.append(h)
            h = pool(h)

        h = self.mid(h, cond, t_emb)

        for up_conv, block, skip in zip(self.up_convs, self.up_blocks, reversed(skips)):
            h = up_conv(h)
            if h.shape[-1] != skip.shape[-1]:
                h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))
            h = torch.cat([h, skip], dim=1)
            h = block(h, cond, t_emb)

        return self.out_conv(h)


# ── Diffusion Surrogate ─────────────────────────────────────────────────────

class DiffusionSurrogate(nn.Module):
    """Conditional DDPM wrapping a 1D U-Net with cosine noise schedule.

    The model generates a margin trajectory conditioned on operating-point
    features extracted from the input telemetry. At inference, it runs the
    full reverse diffusion loop and extracts the 5 standard prediction heads.
    """

    def __init__(self, n_features: int = 4, d_cond: int = 8, hidden: int = 64,
                 n_timesteps: int = 200, beta_schedule: str = "cosine",
                 n_ddim_steps: int = 50, max_seq_len: int = 256,
                 channels: tuple[int, ...] = (32, 64, 128)):
        super().__init__()
        self.n_features = n_features
        self.n_timesteps = n_timesteps
        self.n_ddim_steps = n_ddim_steps
        self.max_seq_len = max_seq_len

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(n_timesteps)
        else:
            betas = linear_beta_schedule(n_timesteps)

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1 - alpha_bar))

        self.cond_encoder = nn.Sequential(
            nn.Linear(n_features, d_cond * 2),
            nn.SiLU(),
            nn.Linear(d_cond * 2, d_cond),
        )

        self.unet = ConditionalUNet1D(in_ch=1, cond_dim=d_cond, channels=channels)

        head_in_dim = d_cond + 1  # cond features + pooled generated margin
        self.istar_head = nn.Sequential(
            nn.Linear(head_in_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.tstar_head = nn.Sequential(
            nn.Linear(head_in_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.time_margin_head = nn.Sequential(
            nn.Linear(head_in_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def _extract_condition(self, X: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B = X.shape[0]
        stats = []
        for b in range(B):
            L = lengths[b].item()
            x_valid = X[b, :L]
            stats.append(x_valid.mean(dim=0))
        cond_input = torch.stack(stats)
        return self.cond_encoder(cond_input)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                  noise: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        return sqrt_ab * x_0 + sqrt_omab * noise

    def training_step(self, x_0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B = x_0.shape[0]
        t = torch.randint(0, self.n_timesteps, (B,), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        eps_pred = self.unet(x_t, t, cond)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def _ddim_sample(self, shape: tuple, cond: torch.Tensor) -> torch.Tensor:
        B = shape[0]
        device = cond.device
        x = torch.randn(shape, device=device)

        step_size = max(1, self.n_timesteps // self.n_ddim_steps)
        timesteps = list(range(self.n_timesteps - 1, -1, -step_size))

        for t_val in timesteps:
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            eps_pred = self.unet(x, t, cond)

            alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)
            if t_val > 0:
                t_prev = max(t_val - step_size, 0)
                alpha_bar_prev = self.alpha_bar[t_prev].view(-1, 1, 1)
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)

            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
            x0_pred = x0_pred.clamp(-5, 5)
            x = torch.sqrt(alpha_bar_prev) * x0_pred + \
                torch.sqrt(1 - alpha_bar_prev) * eps_pred

        return x

    def forward(self, X: torch.Tensor, lengths: torch.Tensor) -> dict:
        B, T, _ = X.shape
        cond = self._extract_condition(X, lengths)

        T_padded = min(T, self.max_seq_len)
        # Round up to nearest power of 2 for clean U-Net downsampling
        T_unet = 1
        while T_unet < T_padded:
            T_unet *= 2

        x_gen = self._ddim_sample((B, 1, T_unet), cond)
        margin = x_gen[:, 0, :T]

        breach_logit = -(margin - 0.05) / 0.02

        gen_pool = x_gen.mean(dim=-1)
        head_input = torch.cat([cond, gen_pool], dim=-1)

        return {
            "margin": margin,
            "breach_logit": breach_logit,
            "istar_log": self.istar_head(head_input).squeeze(-1),
            "tstar": self.tstar_head(head_input).squeeze(-1),
            "time_margin_log": self.time_margin_head(head_input).squeeze(-1),
        }


# ── Training function ──────────────────────────────────────────────────────

def train_diffusion_surrogate(
    df,
    *,
    epochs: int = 100,
    denoising_frac: float = 0.6,
    batch_size: int = 64,
    lr: float = 1e-4,
    istar_weight: float = 1.0,
    tstar_weight: float = 0.0,
    time_margin_weight: float = 0.3,
    n_timesteps: int = 200,
    beta_schedule: str = "cosine",
    n_ddim_steps: int = 50,
    dump_threshold: float = 0.05,
    val_frac: float = 0.2,
    patience: int = 20,
    min_delta: float = 1e-4,
    reduce_lr_patience: int = 10,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    seed: int = 42,
    device: str | None = None,
    model_kwargs: dict | None = None,
    feature_cols=TRUE_FEATURES,
    verbose: bool = True,
):
    """Train a conditional DDPM surrogate on rollout data.

    Two-phase training:
      Phase 1 (first denoising_frac of epochs): denoising loss only
      Phase 2 (remaining epochs): denoising + auxiliary head losses for I*/T*/time_margin

    Returns the same SeqTrainResult used by GRU/Transformer/SSM for checkpoint
    and evaluation compatibility.
    """
    from hts_tape_env.seq_surrogates import SeqTrainResult

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
    model_kwargs.setdefault("n_timesteps", n_timesteps)
    model_kwargs.setdefault("beta_schedule", beta_schedule)
    model_kwargs.setdefault("n_ddim_steps", n_ddim_steps)
    model = DiffusionSurrogate(**model_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=reduce_lr_patience, factor=reduce_lr_factor,
        min_lr=min_lr,
    )

    denoising_epochs = int(epochs * denoising_frac)

    def compute_loss(batch: dict, epoch: int) -> torch.Tensor:
        X, mask = batch["X"].to(device), batch["mask"].to(device)
        margin_target = batch["margin"].to(device)
        lengths = batch["lengths"]

        B, T, _ = X.shape
        cond = model._extract_condition(X, lengths)

        T_unet = 1
        while T_unet < T:
            T_unet *= 2

        x_0 = margin_target.unsqueeze(1)
        if T_unet > T:
            x_0 = F.pad(x_0, (0, T_unet - T))
        x_0 = x_0.to(device)

        loss_denoise = model.training_step(x_0, cond)

        if epoch < denoising_epochs:
            return loss_denoise

        istar_target = batch["istar_log"].to(device)
        tstar_target = batch["tstar"].to(device)
        tm_target = batch["time_margin_log"].to(device)
        tm_valid = batch["time_margin_valid"].to(device)

        x_gen = model._ddim_sample((B, 1, T_unet), cond)
        gen_pool = x_gen.mean(dim=-1)
        head_input = torch.cat([cond, gen_pool], dim=-1)

        loss_istar = F.mse_loss(model.istar_head(head_input).squeeze(-1), istar_target)
        loss_tstar = F.mse_loss(model.tstar_head(head_input).squeeze(-1), tstar_target)
        loss_tm = ((model.time_margin_head(head_input).squeeze(-1) - tm_target) ** 2 * tm_valid).sum() / tm_valid.sum().clamp(min=1e-8)

        return loss_denoise + istar_weight * loss_istar + tstar_weight * loss_tstar + time_margin_weight * loss_tm

    def run_epoch(dl, train: bool, epoch: int) -> float:
        model.train(train)
        total, n_batches = 0.0, 0
        for batch in dl:
            if train:
                loss = compute_loss(batch, epoch)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            else:
                with torch.no_grad():
                    loss = compute_loss(batch, epoch)
            total += loss.item()
            n_batches += 1
        return total / max(n_batches, 1)

    best_val, best_state = float("inf"), None
    epochs_no_improve = 0
    stopped_epoch = epochs - 1
    for epoch in range(epochs):
        # Reset tracking at phase boundary — loss landscape changes
        if epoch == denoising_epochs:
            best_val = float("inf")
            epochs_no_improve = 0
            if verbose:
                print("  --- Phase 2: denoise + heads ---")

        train_loss = run_epoch(train_dl, train=True, epoch=epoch)
        val_loss = run_epoch(val_dl, train=False, epoch=epoch)
        scheduler.step(val_loss)
        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            phase = "denoise" if epoch < denoising_epochs else "denoise+heads"
            print(f"  epoch {epoch:3d} [{phase}]  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if patience > 0 and epochs_no_improve >= patience:
            stopped_epoch = epoch
            if verbose:
                phase = "denoise" if epoch < denoising_epochs else "denoise+heads"
                print(f"  Early stopped at epoch {epoch}/{epochs} [{phase}] (patience={patience})")
            break
    else:
        stopped_epoch = epochs - 1

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Held-out metrics matching seq_surrogates.train_seq_surrogate output
    istar_true_all, istar_pred_all = [], []
    tstar_true_all, tstar_pred_all = [], []
    tm_true_all, tm_pred_all, tm_valid_all = [], [], []
    breach_true_all, breach_prob_all = [], []
    margin_near_true, margin_near_pred = [], []
    boundary_tau = 0.05
    with torch.no_grad():
        for batch in val_dl:
            X, lengths = batch["X"].to(device), batch["lengths"]
            out = model(X, lengths)
            mask_np = batch["mask"].numpy()

            istar_true_all.append(batch["istar_log"].numpy())
            istar_pred_all.append(out["istar_log"].cpu().numpy())
            tstar_true_all.append(batch["tstar"].numpy())
            tstar_pred_all.append(out["tstar"].cpu().numpy())
            tm_true_all.append(batch["time_margin_log"].numpy())
            tm_pred_all.append(out["time_margin_log"].cpu().numpy())
            tm_valid_all.append(batch["time_margin_valid"].numpy())

            breach_true_all.append(batch["breach"].numpy()[mask_np])
            breach_prob_all.append(torch.sigmoid(out["breach_logit"]).cpu().numpy()[:, :mask_np.shape[1]][mask_np])

            margin_np = batch["margin"].numpy()
            margin_pred_np = out["margin"].cpu().numpy()[:, :margin_np.shape[1]]
            near = (np.abs(margin_np) < 2 * boundary_tau) & mask_np
            margin_near_true.append(margin_np[near])
            margin_near_pred.append(margin_pred_np[near])

    istar_true = np.exp(np.concatenate(istar_true_all))
    istar_pred = np.exp(np.concatenate(istar_pred_all))
    istar_mae = float(np.mean(np.abs(istar_true - istar_pred)))
    istar_rel = float(np.median(np.abs(istar_true - istar_pred) / istar_true))

    tstar_mae = float(np.mean(np.abs(np.concatenate(tstar_true_all) - np.concatenate(tstar_pred_all))))

    tm_valid_all_arr = np.concatenate(tm_valid_all).astype(bool)
    tm_true = np.exp(np.concatenate(tm_true_all))[tm_valid_all_arr]
    tm_pred = np.exp(np.concatenate(tm_pred_all))[tm_valid_all_arr]
    tm_rel = float(np.median(np.abs(tm_pred - tm_true) / tm_true)) if tm_valid_all_arr.any() else float("nan")

    breach_true = np.concatenate(breach_true_all)
    breach_prob = np.concatenate(breach_prob_all)
    breach_acc = float(((breach_prob > 0.5).astype(np.float32) == breach_true).mean()) if len(breach_true) else float("nan")

    margin_near_true_arr = np.concatenate(margin_near_true)
    margin_near_pred_arr = np.concatenate(margin_near_pred)
    margin_near_mae = (float(np.mean(np.abs(margin_near_true_arr - margin_near_pred_arr)))
                        if len(margin_near_true_arr) else float("nan"))

    converged = patience > 0 and epochs_no_improve >= patience
    metrics = {
        "best_val_loss": best_val,
        "istar_mae_A": istar_mae,
        "istar_median_rel_err": istar_rel,
        "tstar_mae_K": tstar_mae,
        "time_margin_median_rel_err": tm_rel,
        "breach_accuracy": breach_acc,
        "margin_near_boundary_mae": margin_near_mae,
        "converged": converged,
        "stopped_epoch": stopped_epoch,
        "total_epochs": epochs,
    }
    if verbose:
        print(f"I* held-out:        MAE={istar_mae:.2f} A, median rel err={istar_rel:.2%}")
        print(f"T* held-out:        MAE={tstar_mae:.2f} K")
        print(f"time-margin:        median rel err={tm_rel:.2%} ({tm_valid_all_arr.sum()} valid episodes)")
        print(f"breach classifier:  accuracy={breach_acc:.2%}")
        print(f"margin MAE within 2*tau of the boundary: {margin_near_mae:.4f}")

    return SeqTrainResult(
        model=model, normalizer=full.normalizer, arch="diffusion", metrics=metrics,
        meta={"model_kwargs": model_kwargs, "features": list(feature_cols),
              "dump_threshold": dump_threshold, "training_mode": "ddpm"},
    )
