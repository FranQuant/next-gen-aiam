"""Direct-weight portfolio policy networks: MLPPolicy, LSTMPolicy, TransformerPolicy.

Each class outputs (batch, n_assets) — raw weights before normalization.
No sum-to-1 constraint inside the network; the strategy wrapper normalizes at inference.
Mirrors Session 3 model structure in models.py but outputs n_assets instead of scalar.
"""
from __future__ import annotations

import torch
from torch import nn


def _output_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unknown activation '{name}'. Use 'relu' or 'sigmoid'.")


class MLPPolicy(nn.Module):
    """Tabular MLP: (batch, n_features) → (batch, n_assets).

    activation='relu' for long-only Sharpe/CRRA losses;
    activation='sigmoid' for shrinkage multiplier variant.
    """

    def __init__(
        self,
        n_features: int,
        n_assets: int,
        hidden_dims: tuple[int, ...] = (32, 16),
        dropout: float = 0.10,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_assets))
        layers.append(_output_activation(activation))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LSTMPolicy(nn.Module):
    """Single-layer LSTM: (batch, lookback, n_features) → (batch, n_assets).

    Takes last hidden state and maps through a linear head + output activation.
    """

    def __init__(
        self,
        n_features: int,
        n_assets: int,
        hidden_dim: int = 24,
        dropout: float = 0.10,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, n_assets)
        self.activation = _output_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.activation(self.head(self.drop(out[:, -1, :])))


class _TransformerPolicyBlock(nn.Module):
    """Manual MHA+FFN encoder block — avoids nn.TransformerEncoder segfault on Apple Silicon."""

    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class TransformerPolicy(nn.Module):
    """Per-asset Transformer encoder: (batch, lookback, n_features) → (batch, n_assets).

    Uses learnable positional embeddings (max_lookback=64, matching Session 3c-lite/3c-full).
    Manual MHA+FFN blocks avoid nn.TransformerEncoder segfault on Apple Silicon.
    """

    def __init__(
        self,
        n_features: int,
        n_assets: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.10,
        activation: str = "relu",
        max_lookback: int = 64,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed = nn.Parameter(torch.empty(1, max_lookback, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([_TransformerPolicyBlock(d_model, nhead, dropout) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_assets)
        self.activation = _output_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x) + self.pos_embed[:, : x.size(1), :]
        for block in self.blocks:
            z = block(z)
        return self.activation(self.head(self.drop(z[:, -1, :])))
