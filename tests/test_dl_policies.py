"""Tests for src/aiam/dl/policies.py — output shapes, activations, parameter counts."""
from __future__ import annotations

import torch
import pytest

from aiam.dl.policies import LSTMPolicy, MLPPolicy, TransformerPolicy

N_FEATURES = 5
N_ASSETS = 4
BATCH = 8
LOOKBACK = 16


def _tabular_x(batch: int = BATCH, n: int = N_FEATURES) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, n)


def _seq_x(batch: int = BATCH, lb: int = LOOKBACK, n: int = N_FEATURES) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, lb, n)


# ── MLPPolicy ─────────────────────────────────────────────────────────────────

def test_mlp_output_shape():
    model = MLPPolicy(n_features=N_FEATURES, n_assets=N_ASSETS, hidden_dims=(8,))
    out = model(_tabular_x())
    assert out.shape == (BATCH, N_ASSETS), f"Unexpected shape: {out.shape}"


def test_mlp_relu_nonnegative():
    model = MLPPolicy(n_features=N_FEATURES, n_assets=N_ASSETS, activation="relu")
    model.eval()
    with torch.no_grad():
        out = model(_tabular_x())
    assert (out >= 0).all(), "ReLU policy output contains negative values"


def test_mlp_sigmoid_bounded():
    model = MLPPolicy(n_features=N_FEATURES, n_assets=N_ASSETS, activation="sigmoid")
    model.eval()
    with torch.no_grad():
        out = model(_tabular_x())
    assert (out >= 0).all() and (out <= 1).all(), "Sigmoid output outside [0,1]"


def test_mlp_invalid_activation():
    with pytest.raises(ValueError):
        MLPPolicy(n_features=N_FEATURES, n_assets=N_ASSETS, activation="tanh")


def test_mlp_dropout_train_vs_eval():
    """Dropout should only affect train mode, not eval mode (with fixed seed)."""
    model = MLPPolicy(n_features=N_FEATURES, n_assets=N_ASSETS, dropout=0.5)
    x = _tabular_x()
    model.eval()
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2), "Eval mode outputs differ (dropout not disabled)"


# ── LSTMPolicy ────────────────────────────────────────────────────────────────

def test_lstm_output_shape():
    model = LSTMPolicy(n_features=N_FEATURES, n_assets=N_ASSETS, hidden_dim=8)
    out = model(_seq_x())
    assert out.shape == (BATCH, N_ASSETS)


def test_lstm_relu_nonnegative():
    model = LSTMPolicy(n_features=N_FEATURES, n_assets=N_ASSETS, activation="relu")
    model.eval()
    with torch.no_grad():
        out = model(_seq_x())
    assert (out >= 0).all()


def test_lstm_sigmoid_bounded():
    model = LSTMPolicy(n_features=N_FEATURES, n_assets=N_ASSETS, activation="sigmoid")
    model.eval()
    with torch.no_grad():
        out = model(_seq_x())
    assert (out >= 0).all() and (out <= 1).all()


# ── TransformerPolicy ─────────────────────────────────────────────────────────

def test_transformer_output_shape():
    model = TransformerPolicy(n_features=N_FEATURES, n_assets=N_ASSETS, d_model=8, nhead=2, num_layers=1)
    out = model(_seq_x())
    assert out.shape == (BATCH, N_ASSETS)


def test_transformer_relu_nonnegative():
    model = TransformerPolicy(n_features=N_FEATURES, n_assets=N_ASSETS, d_model=8, nhead=2, num_layers=1, activation="relu")
    model.eval()
    with torch.no_grad():
        out = model(_seq_x())
    assert (out >= 0).all()


def test_transformer_invalid_dmodel_nhead():
    """d_model not divisible by nhead should raise ValueError."""
    with pytest.raises(ValueError):
        TransformerPolicy(n_features=N_FEATURES, n_assets=N_ASSETS, d_model=9, nhead=4)


def test_transformer_pos_embed_shape():
    max_lb = 32
    model = TransformerPolicy(n_features=N_FEATURES, n_assets=N_ASSETS, d_model=8, nhead=2, max_lookback=max_lb)
    assert model.pos_embed.shape == (1, max_lb, 8), f"Unexpected pos_embed shape: {model.pos_embed.shape}"


def test_parameter_counts_are_sensible():
    """Each policy should have at least n_assets parameters (output head)."""
    mlp = MLPPolicy(n_features=N_FEATURES, n_assets=N_ASSETS, hidden_dims=(8,))
    lstm = LSTMPolicy(n_features=N_FEATURES, n_assets=N_ASSETS, hidden_dim=8)
    tf = TransformerPolicy(n_features=N_FEATURES, n_assets=N_ASSETS, d_model=8, nhead=2, num_layers=1)
    for name, model in [("MLP", mlp), ("LSTM", lstm), ("Transformer", tf)]:
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params >= N_ASSETS, f"{name} param count {n_params} < {N_ASSETS}"
