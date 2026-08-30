"""Tests for the knowledge distillation pipeline (teacher→student)."""

import numpy as np
import pandas as pd
import torch
import pytest

from hts_tape_env.seq_data import (
    TRUE_FEATURES,
    TEACHER_FEATURES,
    DistillationDataset,
    collate_distillation,
    EpisodeSequenceDataset,
    collate_episodes,
)
from hts_tape_env.seq_surrogates import (
    train_seq_surrogate,
    train_distilled_surrogate,
    build_model,
    SeqTrainResult,
)
from hts_tape_env.seq_data import Normalizer


def _make_tiny_rollout(n_episodes=30, steps_per_ep=20, seed=42):
    """Synthetic rollout DataFrame with all required columns including lam."""
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(n_episodes):
        T0 = rng.uniform(20, 70)
        lam = rng.uniform(30, 180)
        Istar = 200 + T0 * 2 + lam * 0.5
        Tstar = T0 + 10
        for s in range(steps_per_ep):
            frac = 0.3 + s * 0.005
            I = frac * Istar
            margin = 1 - frac
            V = max(0, I * 1e-6 * (T0 / 80) ** 20)
            T_surf = T0 + s * 0.1
            rows.append({
                "episode_id": ep, "step": s,
                "T0_K": T0, "I_A": I + rng.normal(0, 0.5),
                "voltage_V": V + rng.normal(0, 1e-6),
                "T_surface_K": T_surf + rng.normal(0, 0.05),
                "I_true_A": I, "voltage_true_V": V,
                "T_surface_true_K": T_surf,
                "lam": lam,
                "Istar_A": Istar, "Tstar_K": Tstar,
                "margin_norm": margin,
                "runaway": margin <= 0,
                "time_margin_ms": max(1.0, 100 - s * 5),
            })
    return pd.DataFrame(rows)


def _train_teacher(df, epochs=3):
    """Train a tiny teacher model on 5 features."""
    return train_seq_surrogate(
        df, arch="gru", epochs=epochs,
        feature_cols=TEACHER_FEATURES,
        model_kwargs={"n_features": 5, "hidden": 16, "n_layers": 1},
        verbose=False, seed=42,
    )


class TestDistillationDataset:
    def test_shapes(self):
        df = _make_tiny_rollout(n_episodes=20)
        teacher = _train_teacher(df)
        ds = DistillationDataset(df, teacher, student_feature_cols=TRUE_FEATURES)
        assert len(ds) == 20
        item = ds[0]
        # 7 student items + 5 soft targets = 12
        assert len(item) == 12

    def test_student_has_4_features(self):
        df = _make_tiny_rollout(n_episodes=10)
        teacher = _train_teacher(df)
        ds = DistillationDataset(df, teacher, student_feature_cols=TRUE_FEATURES)
        X = ds[0][0]  # student feature tensor
        assert X.shape[1] == 4

    def test_soft_targets_match_teacher_output(self):
        df = _make_tiny_rollout(n_episodes=10)
        teacher = _train_teacher(df)
        ds = DistillationDataset(df, teacher, student_feature_cols=TRUE_FEATURES)
        margin_soft = ds[0][7]
        assert margin_soft.ndim == 1
        assert margin_soft.shape[0] == ds[0][0].shape[0]  # same seq length


class TestCollateDistillation:
    def test_batch_dict_keys(self):
        df = _make_tiny_rollout(n_episodes=10)
        teacher = _train_teacher(df)
        ds = DistillationDataset(df, teacher)
        batch_items = [ds[i] for i in range(min(4, len(ds)))]
        batch = collate_distillation(batch_items)
        for key in ("X", "margin", "breach", "mask", "lengths",
                    "istar_log", "tstar", "time_margin_log", "time_margin_valid",
                    "margin_soft", "breach_logit_soft", "istar_log_soft",
                    "tstar_soft", "tm_log_soft"):
            assert key in batch, f"Missing key: {key}"

    def test_soft_targets_padded(self):
        df = _make_tiny_rollout(n_episodes=10)
        teacher = _train_teacher(df)
        ds = DistillationDataset(df, teacher)
        batch_items = [ds[i] for i in range(min(4, len(ds)))]
        batch = collate_distillation(batch_items)
        B, T = batch["X"].shape[:2]
        assert batch["margin_soft"].shape == (B, T)
        assert batch["breach_logit_soft"].shape == (B, T)
        assert batch["istar_log_soft"].shape == (B,)


class TestTrainDistilledSurrogate:
    def test_smoke(self):
        df = _make_tiny_rollout(n_episodes=30)
        teacher = _train_teacher(df, epochs=3)
        student = train_distilled_surrogate(
            df, teacher, arch="gru", epochs=3,
            alpha=0.5, temperature=2.0,
            model_kwargs={"hidden": 16, "n_layers": 1},
            verbose=False, seed=42,
        )
        assert isinstance(student, SeqTrainResult)
        assert student.arch == "gru"
        assert "istar_mae_A" in student.metrics

    def test_checkpoint_metadata(self):
        df = _make_tiny_rollout(n_episodes=20)
        teacher = _train_teacher(df, epochs=2)
        student = train_distilled_surrogate(
            df, teacher, arch="gru", epochs=2,
            alpha=0.7, temperature=3.0,
            teacher_path="/tmp/teacher.pt",
            model_kwargs={"hidden": 16, "n_layers": 1},
            verbose=False,
        )
        assert student.meta["distill_alpha"] == 0.7
        assert student.meta["distill_temperature"] == 3.0
        assert student.meta["teacher_path"] == "/tmp/teacher.pt"
        assert student.meta["features"] == list(TRUE_FEATURES)

    def test_student_normalizer_is_4d(self):
        df = _make_tiny_rollout(n_episodes=20)
        teacher = _train_teacher(df, epochs=2)
        student = train_distilled_surrogate(
            df, teacher, arch="gru", epochs=2,
            model_kwargs={"hidden": 16, "n_layers": 1},
            verbose=False,
        )
        assert student.normalizer.mean.shape[0] == 4
        assert student.normalizer.std.shape[0] == 4

    def test_save_and_load(self, tmp_path):
        df = _make_tiny_rollout(n_episodes=20)
        teacher = _train_teacher(df, epochs=2)
        student = train_distilled_surrogate(
            df, teacher, arch="gru", epochs=2,
            model_kwargs={"hidden": 16, "n_layers": 1},
            verbose=False,
        )
        path = tmp_path / "student.pt"
        student.save(path)

        from hts_tape_env.seq_surrogates import load_seq_surrogate
        loaded = load_seq_surrogate(path)
        assert loaded.arch == "gru"
        assert loaded.meta["distill_alpha"] == 0.5
        # Verify forward pass works
        X = torch.randn(1, 10, 4)
        lengths = torch.tensor([10])
        out = loaded.model(X, lengths)
        assert out["istar_log"].shape == (1,)
