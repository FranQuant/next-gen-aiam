"""Portfolio-level loss functions for direct-weight DL policies.

All three operate on torch Tensors and support gradient flow.
Ported from the proof-of-concept in notebooks/_local_16b_reference.html.
"""
from __future__ import annotations

import torch


def sharpe_loss(weights: torch.Tensor, returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Negative Sharpe of portfolio returns. weights, returns shape: (batch, n_assets)."""
    portfolio_returns = (weights * returns).sum(dim=-1)
    return -portfolio_returns.mean() / (portfolio_returns.std() + eps)


def crra_loss(
    weights: torch.Tensor,
    returns: torch.Tensor,
    gamma: float = 5.0,
    clip_min: float = -0.99,
) -> torch.Tensor:
    """Negative CRRA utility (gamma=5 is the JPM 2024 default).

    Clips portfolio_returns at min=-0.99 to prevent (1+R)^(1-gamma) blowup
    on negative tail moves.
    """
    portfolio_returns = (weights * returns).sum(dim=-1)
    portfolio_returns = torch.clamp(portfolio_returns, min=clip_min)
    utility = ((1 + portfolio_returns) ** (1 - gamma)) / (1 - gamma)
    return -utility.mean()


def crra_shrinkage_loss(
    weights: torch.Tensor,
    returns: torch.Tensor,
    benchmark_w: torch.Tensor,
    gamma: float = 5.0,
    clip_min: float = -0.99,
) -> torch.Tensor:
    """CRRA where effective_weights = weights * benchmark_w (element-wise).

    benchmark_w: tensor of shape (n_assets,) — prior allocation (e.g., 1/N EW).
    Network output (Sigmoid activation) acts as a per-asset shrinkage multiplier.
    """
    benchmark_w = benchmark_w.to(weights.device)
    effective_weights = weights * benchmark_w
    portfolio_returns = (effective_weights * returns).sum(dim=-1)
    portfolio_returns = torch.clamp(portfolio_returns, min=clip_min)
    utility = ((1 + portfolio_returns) ** (1 - gamma)) / (1 - gamma)
    return -utility.mean()
