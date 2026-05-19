"""Tests for src/aiam/dl/losses.py — gradient flow, numerics, edge cases."""
from __future__ import annotations

import torch
import pytest

from aiam.dl.losses import crra_loss, crra_shrinkage_loss, sharpe_loss


def _rand(batch: int = 16, n_assets: int = 4, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    weights = torch.rand(batch, n_assets, requires_grad=True)
    returns = torch.randn(batch, n_assets) * 0.01
    return weights, returns


# ── sharpe_loss ──────────────────────────────────────────────────────────────

def test_sharpe_loss_returns_scalar():
    w, r = _rand()
    loss = sharpe_loss(w, r)
    assert loss.shape == (), f"Expected scalar, got {loss.shape}"


def test_sharpe_loss_gradient_flows():
    w, r = _rand()
    loss = sharpe_loss(w, r)
    loss.backward()
    assert w.grad is not None
    assert not torch.all(w.grad == 0), "Gradient is all-zero"


def test_sharpe_loss_negative_for_positive_mean():
    """Portfolio with strong positive mean returns should yield a negative Sharpe loss."""
    torch.manual_seed(7)
    weights = torch.ones(32, 4) / 4
    returns = torch.full((32, 4), 0.01)
    loss = sharpe_loss(weights, returns)
    assert loss.item() < 0, f"Expected negative loss, got {loss.item():.4f}"


# ── crra_loss ─────────────────────────────────────────────────────────────────

def test_crra_loss_returns_scalar():
    w, r = _rand()
    loss = crra_loss(w, r)
    assert loss.shape == ()


def test_crra_loss_gradient_flows():
    w, r = _rand()
    loss = crra_loss(w, r)
    loss.backward()
    assert w.grad is not None and not torch.all(w.grad == 0)


def test_crra_loss_no_nan_at_clip_boundary():
    """Portfolio return exactly at clip_min=-0.99 should not produce NaN."""
    torch.manual_seed(3)
    weights = torch.ones(8, 2) / 2
    returns = torch.full((8, 2), -0.99)
    loss = crra_loss(weights, returns, gamma=5.0, clip_min=-0.99)
    assert not torch.isnan(loss), f"NaN at clip boundary: {loss}"
    assert torch.isfinite(loss), f"Non-finite loss: {loss}"


def test_crra_loss_no_nan_below_clip_boundary():
    """Returns below clip_min should be clamped; loss must remain finite."""
    torch.manual_seed(5)
    weights = torch.ones(8, 2) / 2
    returns = torch.full((8, 2), -5.0)
    loss = crra_loss(weights, returns, gamma=5.0, clip_min=-0.99)
    assert torch.isfinite(loss)


# ── crra_shrinkage_loss ───────────────────────────────────────────────────────

def test_crra_shrinkage_returns_scalar():
    w, r = _rand()
    bw = torch.ones(4) / 4
    loss = crra_shrinkage_loss(w, r, bw)
    assert loss.shape == ()


def test_crra_shrinkage_gradient_flows():
    w, r = _rand()
    bw = torch.ones(4) / 4
    loss = crra_shrinkage_loss(w, r, bw)
    loss.backward()
    assert w.grad is not None and not torch.all(w.grad == 0)


def test_crra_shrinkage_equals_plain_crra_when_benchmark_is_ones():
    """When benchmark_w is all-ones, effective_weights = weights -> same as plain crra_loss."""
    torch.manual_seed(9)
    weights = torch.rand(16, 4, requires_grad=False)
    returns = torch.randn(16, 4) * 0.01
    bw_ones = torch.ones(4)
    loss_shrink = crra_shrinkage_loss(weights, returns, bw_ones, gamma=5.0)
    loss_plain = crra_loss(weights, returns, gamma=5.0)
    assert abs(loss_shrink.item() - loss_plain.item()) < 1e-5, (
        f"Expected equal losses with benchmark_w=ones, got {loss_shrink:.6f} vs {loss_plain:.6f}"
    )
