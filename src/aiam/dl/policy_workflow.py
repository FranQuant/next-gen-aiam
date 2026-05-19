"""Training loop and multi-seed orchestration for direct-weight portfolio policies.

Mirrors the structure of workflow.py but accepts portfolio-level loss functions
(from losses.py) and produces y of shape (N, n_assets) rather than scalar targets.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aiam.dl.workflow import set_global_seed
from aiam.ml.workflow import apply_standardizer, chronological_splits, fit_standardizer


@dataclass
class DirectWeightFitResult:
    model: nn.Module
    history: pd.DataFrame   # cols: epoch, train_loss, val_loss
    summary: dict           # best_epoch, best_val_loss, n_epochs_trained, seed


@dataclass
class DirectWeightSeedEnsemble:
    fits: list[DirectWeightFitResult]
    seeds: tuple[int, ...]

    def predict_weights(self, X: np.ndarray) -> np.ndarray:
        """Mean weight matrix across seeds. X shape: (batch, ...) matching policy input."""
        preds = [_predict_policy(fr.model, X) for fr in self.fits]
        return np.mean(preds, axis=0)


def _predict_policy(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        t = torch.tensor(np.asarray(X, dtype="float32"))
        return model(t).cpu().numpy()


def fit_direct_weight_policy(
    policy_class: type,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    seed: int = 42,
    device: str = "cpu",
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    max_epochs: int = 80,
    patience: int = 12,
    **policy_kwargs,
) -> DirectWeightFitResult:
    """Train a single-seed direct-weight policy with the given portfolio-level loss.

    Validation loss is computed on the full validation set (not batched) so that
    Sharpe and CRRA are evaluated on the full return distribution — not batch estimates.
    """
    set_global_seed(seed)
    X_train = np.asarray(X_train, dtype="float32")
    y_train = np.asarray(y_train, dtype="float32")
    X_val = np.asarray(X_val, dtype="float32")
    y_val = np.asarray(y_val, dtype="float32")

    model = policy_class(**policy_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val, dtype=torch.float32).to(device)

    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(Xt, yt),
        batch_size=max(1, min(batch_size, len(Xt))),
        shuffle=True,
        generator=gen,
    )

    best_state = deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = patience
    rows: list[dict] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        run_loss, n_obs = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * len(xb)
            n_obs += len(xb)

        train_loss = run_loss / max(n_obs, 1)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()

        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-10:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    model = model.cpu()
    history = pd.DataFrame(rows)
    summary = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "n_epochs_trained": len(rows),
        "seed": seed,
    }
    return DirectWeightFitResult(model=model, history=history, summary=summary)


def fit_direct_weight_seed_ensemble(
    policy_class: type,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    device: str = "cpu",
    **kwargs,
) -> DirectWeightSeedEnsemble:
    """Train N direct-weight policies with different seeds; return as ensemble.

    kwargs are forwarded to both fit_direct_weight_policy and policy_class.__init__,
    keyed so that the policy kwargs are passed via **policy_kwargs.
    """
    fits = []
    for s in seeds:
        fits.append(
            fit_direct_weight_policy(
                policy_class, X_train, y_train, X_val, y_val,
                loss_fn, seed=s, device=device, **kwargs,
            )
        )
    return DirectWeightSeedEnsemble(fits=fits, seeds=tuple(seeds))


def build_policy_sequence_windows(
    feature_panel: pd.DataFrame,
    target_panel: pd.Series,
    feature_cols: list[str],
    assets: list[str],
    lookback: int,
    allowed_dates: Optional[set] = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build LSTM/Transformer input windows with cross-asset y targets.

    Unlike build_sequence_windows in workflow.py (scalar y per asset-window),
    y here has shape (N, n_assets): each row contains realized returns for
    ALL assets at the window's terminal date. This enables direct portfolio-
    level loss computation without a gather step.

    allowed_dates: if provided, only include windows whose terminal date is in this set.
    The full panel is still needed for lookback history even when filtering terminal dates.

    Returns (X, y, meta_df) where:
        X: (N, lookback, n_features) float32
        y: (N, n_assets) float32
        meta_df: columns [Date, asset]
    """
    index_names = list(feature_panel.index.names)
    date_name = index_names[0]
    asset_name = index_names[1] if len(index_names) > 1 else "asset"

    fp_reset = feature_panel[feature_cols].reset_index()
    fp_reset = fp_reset.rename(columns={date_name: "Date", asset_name: "asset"})

    tp_reset = target_panel.reset_index()
    tp_reset = tp_reset.rename(columns={date_name: "Date", asset_name: "asset", tp_reset.columns[-1]: "target"})

    merged = fp_reset.merge(tp_reset[["Date", "asset", "target"]], on=["Date", "asset"], how="left")
    target_wide = merged.pivot_table(index="Date", columns="asset", values="target")[assets]

    seq_x: list[np.ndarray] = []
    seq_y: list[np.ndarray] = []
    meta_rows: list[dict] = []

    for asset, grp in merged.groupby("asset", sort=False):
        grp = grp.sort_values("Date").reset_index(drop=True)
        feat_arr = grp[feature_cols].to_numpy(dtype="float32")
        dates_arr = pd.to_datetime(grp["Date"])

        for end in range(lookback - 1, len(grp)):
            date_t = dates_arr.iloc[end]
            if allowed_dates is not None and date_t not in allowed_dates:
                continue
            window = feat_arr[end - lookback + 1 : end + 1]
            if np.isnan(window).any():
                continue
            if date_t not in target_wide.index:
                continue
            y_row = target_wide.loc[date_t].to_numpy(dtype="float32")
            if np.isnan(y_row).any():
                continue
            seq_x.append(window)
            seq_y.append(y_row)
            meta_rows.append({"Date": date_t, "asset": str(asset)})

    if not seq_x:
        return (
            np.empty((0, lookback, len(feature_cols)), dtype="float32"),
            np.empty((0, len(assets)), dtype="float32"),
            pd.DataFrame(columns=["Date", "asset"]),
        )
    return (
        np.stack(seq_x, axis=0),
        np.stack(seq_y, axis=0),
        pd.DataFrame(meta_rows),
    )
