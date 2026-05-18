"""Tests for src/aiam/dl/models.py — forward-pass shape contracts."""
from __future__ import annotations

import pytest
import torch

from aiam.dl.models import LSTMRegressor, MLPRegressor, TransformerRegressor

BATCH = 8
N_FEAT = 17
LOOKBACK = 10


def test_mlp_output_shape():
    model = MLPRegressor(n_features=N_FEAT, hidden_dims=(4,), dropout=0.0)
    x = torch.randn(BATCH, N_FEAT)
    out = model(x)
    assert out.shape == (BATCH,)


def test_mlp_single_sample():
    model = MLPRegressor(n_features=N_FEAT, hidden_dims=(4,), dropout=0.0)
    x = torch.randn(1, N_FEAT)
    assert model(x).shape == (1,)


def test_lstm_output_shape():
    model = LSTMRegressor(n_features=N_FEAT, hidden_dim=4, dropout=0.0)
    x = torch.randn(BATCH, LOOKBACK, N_FEAT)
    out = model(x)
    assert out.shape == (BATCH,)


def test_transformer_output_shape():
    model = TransformerRegressor(n_features=N_FEAT, d_model=8, nhead=2, num_layers=1, dropout=0.0)
    x = torch.randn(BATCH, LOOKBACK, N_FEAT)
    out = model(x)
    assert out.shape == (BATCH,)


def test_transformer_nhead_mismatch_raises():
    with pytest.raises(ValueError, match="divisible"):
        TransformerRegressor(n_features=N_FEAT, d_model=7, nhead=4)


def test_mlp_param_count():
    model = MLPRegressor(n_features=4, hidden_dims=(4,), dropout=0.0)
    n_params = sum(p.numel() for p in model.parameters())
    # Layer 1: 4*4 + 4 = 20; Layer 2: 4*1 + 1 = 5 → total 25
    assert n_params == 25


def test_lstm_output_is_float32():
    model = LSTMRegressor(n_features=N_FEAT, hidden_dim=4, dropout=0.0)
    x = torch.randn(BATCH, LOOKBACK, N_FEAT)
    assert model(x).dtype == torch.float32


def test_transformer_no_cross_asset_leakage():
    """Each sample in the batch is independent: shuffling batch dim gives same predictions."""
    model = TransformerRegressor(n_features=N_FEAT, d_model=8, nhead=2, num_layers=1, dropout=0.0)
    model.eval()
    x = torch.randn(BATCH, LOOKBACK, N_FEAT)
    with torch.no_grad():
        out_orig = model(x)
        idx = torch.randperm(BATCH)
        out_shuffled = model(x[idx])
    torch.testing.assert_close(out_orig[idx], out_shuffled)
