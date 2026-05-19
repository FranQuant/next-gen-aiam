"""Tests for src/aiam/rl/policy.py — SimplexPolicy contract."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from aiam.rl.policy import SimplexPolicy

F = 10  # n_features used throughout
H = 16  # hidden_dim


# ── Simplex output ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_assets", [1, 2, 3, 29])
def test_forward_output_on_simplex(n_assets: int):
    policy = SimplexPolicy(n_features=F, hidden_dim=H)
    batch = 4
    features = torch.randn(batch, n_assets, F)
    weights = torch.rand(batch, n_assets, 1)
    weights = weights / weights.sum(dim=1, keepdim=True)

    with torch.no_grad():
        out = policy(features, weights)

    assert out.shape == (batch, n_assets), f"expected ({batch}, {n_assets}), got {out.shape}"
    assert torch.allclose(out.sum(dim=-1), torch.ones(batch), atol=1e-5), "rows must sum to 1"
    assert (out >= 0).all(), "weights must be non-negative"


# ── Parameter count is independent of N ─────────────────────────────────────

def test_parameter_count_independent_of_n():
    policy = SimplexPolicy(n_features=F, hidden_dim=H)
    param_count = sum(p.numel() for p in policy.parameters())

    # Running forward passes for different N must not change parameter count.
    for n_assets in (1, 5, 29):
        features = torch.randn(1, n_assets, F)
        weights = torch.ones(1, n_assets, 1) / n_assets
        with torch.no_grad():
            out = policy(features, weights)
        assert out.shape == (1, n_assets)

    assert sum(p.numel() for p in policy.parameters()) == param_count


def test_expected_parameter_count():
    """Verify exact param count: encoder (F+1→H, H→H) + head (H→1) with biases."""
    in_dim = F + 1  # use_weights=True by default
    expected = (
        in_dim * H + H   # Linear(in_dim, H): weight + bias
        + H * H + H      # Linear(H, H)
        + H * 1 + 1      # Linear(H, 1)
    )
    policy = SimplexPolicy(n_features=F, hidden_dim=H)
    actual = sum(p.numel() for p in policy.parameters())
    assert actual == expected


# ── Deterministic .act ──────────────────────────────────────────────────────

def test_act_deterministic_under_fixed_seed():
    """Same state → same output on two calls (no stochasticity in .act)."""
    torch.manual_seed(99)
    policy = SimplexPolicy(n_features=F, hidden_dim=H)
    state = {
        "features": np.random.default_rng(0).standard_normal((5, F)).astype(np.float32),
        "weights": np.ones(5, dtype=np.float32) / 5,
    }
    w1 = policy.act(state)
    w2 = policy.act(state)
    np.testing.assert_array_equal(w1, w2)
    assert np.isclose(w1.sum(), 1.0, atol=1e-6)


def test_act_output_shape_matches_n():
    for n in (1, 4, 29):
        policy = SimplexPolicy(n_features=F, hidden_dim=H)
        state = {
            "features": np.zeros((n, F), dtype=np.float32),
            "weights": np.ones(n, dtype=np.float32) / n,
        }
        out = policy.act(state)
        assert out.shape == (n,)
        assert np.isclose(out.sum(), 1.0, atol=1e-6)
