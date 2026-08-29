"""Tests for sinusoidal positional embeddings in Transformer and SSM surrogates."""

import math

import torch
import pytest

from hts_tape_env.seq_surrogates import (
    _sinusoidal_pos_encoding,
    TransformerSurrogate,
    SSMSurrogate,
    build_model,
)


class TestSinusoidalEncoding:
    def test_shape(self):
        pe = _sinusoidal_pos_encoding(128, 64)
        assert pe.shape == (128, 64)

    def test_sin_cos_interleaving(self):
        pe = _sinusoidal_pos_encoding(10, 4)
        # Position 1: sin columns at even indices, cos at odd
        pos1_sin = math.sin(1.0 * math.exp(0 * -(math.log(10000.0) / 4)))
        assert abs(pe[1, 0].item() - pos1_sin) < 1e-5

    def test_position_zero_sin_is_zero(self):
        pe = _sinusoidal_pos_encoding(10, 64)
        # sin(0 * anything) = 0 for all even columns
        assert torch.allclose(pe[0, 0::2], torch.zeros(32), atol=1e-6)

    def test_position_zero_cos_is_one(self):
        pe = _sinusoidal_pos_encoding(10, 64)
        # cos(0 * anything) = 1 for all odd columns
        assert torch.allclose(pe[0, 1::2], torch.ones(32), atol=1e-6)


class TestTransformerSinusoidal:
    def test_sinusoidal_forward(self):
        model = TransformerSurrogate(n_features=4, d_model=32, n_heads=2,
                                      n_layers=1, max_len=64, pos_emb="sinusoidal")
        X = torch.randn(2, 16, 4)
        lengths = torch.tensor([16, 10])
        out = model(X, lengths)
        assert out["istar_log"].shape == (2,)
        assert out["margin"].shape == (2, 16)
        assert out["breach_logit"].shape == (2, 16)
        assert out["tstar"].shape == (2,)
        assert out["time_margin_log"].shape == (2,)

    def test_sinusoidal_uses_buffer(self):
        model = TransformerSurrogate(n_features=4, d_model=32, max_len=64,
                                      pos_emb="sinusoidal")
        assert "pos" in dict(model.named_buffers())
        assert "pos" not in dict(model.named_parameters())

    def test_learned_backward_compat(self):
        model = TransformerSurrogate(n_features=4, d_model=32, max_len=64,
                                      pos_emb="learned")
        assert isinstance(model.pos, torch.nn.Parameter)
        # Should still work in forward
        X = torch.randn(2, 16, 4)
        lengths = torch.tensor([16, 10])
        out = model(X, lengths)
        assert out["istar_log"].shape == (2,)


class TestSSMSinusoidal:
    def test_sinusoidal_forward(self):
        model = SSMSurrogate(n_features=4, hidden=32, n_layers=1,
                              d_state=16, pos_emb="sinusoidal", max_len=64)
        X = torch.randn(2, 16, 4)
        lengths = torch.tensor([16, 10])
        out = model(X, lengths)
        assert out["istar_log"].shape == (2,)
        assert out["margin"].shape == (2, 16)

    def test_sinusoidal_uses_buffer(self):
        model = SSMSurrogate(n_features=4, hidden=32, pos_emb="sinusoidal",
                              max_len=64)
        assert "pos" in dict(model.named_buffers())

    def test_none_embedding(self):
        model = SSMSurrogate(n_features=4, hidden=32, n_layers=1,
                              d_state=16, pos_emb="none")
        assert model.pos is None
        X = torch.randn(2, 16, 4)
        lengths = torch.tensor([16, 10])
        out = model(X, lengths)
        assert out["istar_log"].shape == (2,)


class TestBuildModelPassthrough:
    def test_transformer_pos_emb(self):
        model = build_model("transformer", n_features=4, d_model=32, n_heads=2,
                             n_layers=1, pos_emb="sinusoidal")
        assert "pos" in dict(model.named_buffers())

    def test_ssm_pos_emb(self):
        model = build_model("ssm", n_features=4, hidden=32, n_layers=1,
                             d_state=16, pos_emb="sinusoidal")
        assert "pos" in dict(model.named_buffers())
