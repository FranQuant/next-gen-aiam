"""PyTorch nn.Module subclasses: MLPRegressor, LSTMRegressor, TransformerRegressor."""
from __future__ import annotations

import math

import torch
from torch import nn


class MLPRegressor(nn.Module):
    """Tabular MLP: (batch, n_features) → (batch,)."""

    def __init__(self, n_features: int, hidden_dims: tuple[int, ...] = (32, 16), dropout: float = 0.10) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class LSTMRegressor(nn.Module):
    """Single-layer LSTM: (batch, lookback, n_features) → (batch,). Takes last hidden state."""

    def __init__(self, n_features: int, hidden_dim: int = 24, dropout: float = 0.10) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1, :])).squeeze(-1)


class _TransformerBlock(nn.Module):
    """Single MHA + FFN encoder block (manual implementation avoids nn.TransformerEncoder segfault on Apple Silicon)."""

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


class TransformerRegressor(nn.Module):
    """Per-asset Transformer encoder: (batch, lookback, n_features) → (batch,).

    Projects n_features → d_model, adds learnable positional embeddings, runs num_layers of
    self-attention, takes last position. NOT cross-asset: each asset's sequence is processed
    independently. Uses manual MHA+FFN blocks instead of nn.TransformerEncoder (avoids Apple
    Silicon segfault).
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.10,
        max_lookback: int = 64,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed = nn.Parameter(torch.empty(1, max_lookback, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([_TransformerBlock(d_model, nhead, dropout) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x) + self.pos_embed[:, : x.size(1), :]
        for block in self.blocks:
            z = block(z)
        return self.head(self.drop(z[:, -1, :])).squeeze(-1)
