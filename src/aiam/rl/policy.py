"""SimplexPolicy: shared per-asset encoder → softmax portfolio weights."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SimplexPolicy(nn.Module):
    """Per-asset MLP encoder with weights tied across assets, softmax output on the simplex.

    Architecture:
        input  : (B, N, F) features, optionally concat (B, N, 1) current weights
        encoder: Linear(F[+1] → H) → ReLU → Linear(H → H) → ReLU  (shared across all N)
        head   : Linear(H → 1) per asset, squeeze → (B, N) logits
        output : softmax(logits, dim=-1) → (B, N) weights summing to 1

    The encoder is applied to (B*N, F[+1]) so parameter count is independent of N.

    Stochastic sampling (for training, Session 4b): Dirichlet(alpha = softmax_logits * temperature).
    Dirichlet is the natural distribution on the simplex; temperature > 1 → more uniform exploration.
    Deterministic inference (.act): return softmax weights directly.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 32,
        use_weights: bool = True,
    ) -> None:
        super().__init__()
        self.use_weights = use_weights
        in_dim = n_features + 1 if use_weights else n_features
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            features : (B, N, F)
            weights  : (B, N, 1) optional current portfolio weights

        Returns:
            (B, N) simplex weights
        """
        B, N, _ = features.shape
        x = torch.cat([features, weights], dim=-1) if (self.use_weights and weights is not None) else features
        h = self.encoder(x.reshape(B * N, -1))     # (B*N, hidden_dim)
        logits = self.head(h).reshape(B, N)         # (B, N)
        return torch.softmax(logits, dim=-1)

    def act(self, state: dict) -> np.ndarray:
        """Deterministic inference: softmax weights from the current state.

        Args:
            state: dict with 'features' (N, F) and 'weights' (N,)

        Returns:
            (N,) numpy weight vector on the simplex
        """
        feat = torch.tensor(state["features"][None], dtype=torch.float32)   # (1, N, F)
        w = torch.tensor(state["weights"][None, :, None], dtype=torch.float32)  # (1, N, 1)
        self.eval()
        with torch.no_grad():
            out = self.forward(feat, w if self.use_weights else None)  # (1, N)
        return out.squeeze(0).cpu().numpy()

    def sample(self, state: dict, temperature: float = 1.0) -> np.ndarray:
        """Stochastic sampling for training: Dirichlet(alpha = softmax_logits * temperature).

        Higher temperature → more uniform (more exploration).
        Reserved for Session 4b training loop.
        """
        alpha = self.act(state) * temperature
        alpha = np.clip(alpha, 1e-3, None)
        return np.random.dirichlet(alpha)
