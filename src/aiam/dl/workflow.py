"""DL workflow helpers: seed control, training loops, sequence windowing, multi-seed ensemble."""
from __future__ import annotations

import random
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aiam.ml.workflow import apply_standardizer, chronological_splits, fit_standardizer


def set_global_seed(seed: int = 42) -> None:
    """Seed python, numpy, torch (including cudnn determinism flags)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _rank_ic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Spearman rank correlation between predictions and targets."""
    from scipy.stats import spearmanr
    if len(y_pred) < 2:
        return float("nan")
    rho, _ = spearmanr(y_pred, y_true)
    return float(rho) if not np.isnan(rho) else 0.0


@dataclass
class FitResult:
    model: nn.Module
    history: pd.DataFrame  # cols: epoch, train_loss, val_loss, val_rank_ic
    summary: dict          # best_epoch, best_val_loss, val_ic_at_best, n_epochs_trained, seed


@dataclass
class SeedEnsembleResult:
    fits: list[FitResult]
    seeds: tuple[int, ...]

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        """Mean of per-seed predictions. X shape depends on model type."""
        preds = [_predict(fr.model, X) for fr in self.fits]
        return np.mean(preds, axis=0)

    def stability_summary(self) -> dict:
        ics = [fr.summary["val_ic_at_best"] for fr in self.fits]
        ics_arr = np.array([v for v in ics if not np.isnan(v)], dtype=float)
        if len(ics_arr) == 0:
            return {"mean": float("nan"), "stdev": float("nan"), "min": float("nan"), "max": float("nan")}
        return {
            "mean": float(ics_arr.mean()),
            "stdev": float(ics_arr.std()),
            "min": float(ics_arr.min()),
            "max": float(ics_arr.max()),
        }


def _predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        t = torch.tensor(np.asarray(X, dtype="float32"))
        return model(t).cpu().numpy()


def _training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    max_epochs: int,
    patience: int,
    seed: int,
    device: str,
) -> FitResult:
    model = model.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)

    best_state = deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = patience
    rows: list[dict] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        run_loss, n_obs = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * len(xb)
            n_obs += len(xb)

        train_loss = run_loss / max(n_obs, 1)
        model.eval()
        with torch.no_grad():
            val_out = model(val_x)
            val_loss = criterion(val_out, val_y).item()
            val_ic = _rank_ic(val_out.cpu().numpy(), val_y.cpu().numpy())

        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_rank_ic": val_ic})

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
    history = pd.DataFrame(rows)
    best_row = history[history["epoch"] == best_epoch].iloc[0]
    summary = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "val_ic_at_best": float(best_row["val_rank_ic"]),
        "n_epochs_trained": len(rows),
        "seed": seed,
    }
    return FitResult(model=model.cpu(), history=history, summary=summary)


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=max(1, min(batch_size, len(dataset))), shuffle=True, generator=gen)


def fit_mlp_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden_dims: tuple[int, ...] = (32, 16),
    dropout: float = 0.10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    max_epochs: int = 120,
    patience: int = 15,
    seed: int = 42,
    device: str = "cpu",
) -> FitResult:
    from aiam.dl.models import MLPRegressor

    set_global_seed(seed)
    X_train = np.asarray(X_train, dtype="float32")
    y_train = np.asarray(y_train, dtype="float32")
    X_val = np.asarray(X_val, dtype="float32")
    y_val = np.asarray(y_val, dtype="float32")

    model = MLPRegressor(n_features=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = _make_loader(X_train, y_train, batch_size, seed)
    return _training_loop(
        model, loader,
        torch.tensor(X_val), torch.tensor(y_val),
        optimizer, nn.MSELoss(), max_epochs, patience, seed, device,
    )


def fit_lstm_regressor(
    X_train_seq: np.ndarray,
    y_train: np.ndarray,
    X_val_seq: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden_dim: int = 24,
    dropout: float = 0.10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    max_epochs: int = 80,
    patience: int = 12,
    seed: int = 42,
    device: str = "cpu",
) -> FitResult:
    from aiam.dl.models import LSTMRegressor

    set_global_seed(seed)
    X_train_seq = np.asarray(X_train_seq, dtype="float32")
    y_train = np.asarray(y_train, dtype="float32")
    X_val_seq = np.asarray(X_val_seq, dtype="float32")
    y_val = np.asarray(y_val, dtype="float32")

    n_features = X_train_seq.shape[2]
    model = LSTMRegressor(n_features=n_features, hidden_dim=hidden_dim, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = _make_loader(X_train_seq, y_train, batch_size, seed)
    return _training_loop(
        model, loader,
        torch.tensor(X_val_seq), torch.tensor(y_val),
        optimizer, nn.MSELoss(), max_epochs, patience, seed, device,
    )


def fit_transformer_regressor(
    X_train_seq: np.ndarray,
    y_train: np.ndarray,
    X_val_seq: np.ndarray,
    y_val: np.ndarray,
    *,
    d_model: int = 32,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    max_epochs: int = 80,
    patience: int = 12,
    seed: int = 42,
    device: str = "cpu",
) -> FitResult:
    from aiam.dl.models import TransformerRegressor

    set_global_seed(seed)
    X_train_seq = np.asarray(X_train_seq, dtype="float32")
    y_train = np.asarray(y_train, dtype="float32")
    X_val_seq = np.asarray(X_val_seq, dtype="float32")
    y_val = np.asarray(y_val, dtype="float32")

    n_features = X_train_seq.shape[2]
    model = TransformerRegressor(n_features=n_features, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = _make_loader(X_train_seq, y_train, batch_size, seed)
    return _training_loop(
        model, loader,
        torch.tensor(X_val_seq), torch.tensor(y_val),
        optimizer, nn.MSELoss(), max_epochs, patience, seed, device,
    )


def fit_with_seed_ensemble(
    fit_fn: Callable[..., FitResult],
    fit_kwargs: dict,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> SeedEnsembleResult:
    """Run fit_fn once per seed and return a SeedEnsembleResult."""
    fits = []
    for s in seeds:
        kwargs = dict(fit_kwargs)
        kwargs["seed"] = s
        fits.append(fit_fn(**kwargs))
    return SeedEnsembleResult(fits=fits, seeds=seeds)


def build_sequence_windows(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lookback: int = 63,
    allowed_splits: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Slice long-format panel into per-asset sequences of shape (lookback, n_features).

    Returns (X_seq, y_seq, meta) where meta has columns (Date, asset, split).
    Sequences never cross asset boundaries. When allowed_splits is given, only rows
    where the target date's split is in allowed_splits are included.
    """
    allowed = None if allowed_splits is None else set(allowed_splits)
    asset_col = "asset" if "asset" in frame.columns else "Asset"
    split_col = "split" if "split" in frame.columns else None

    seq_x: list[np.ndarray] = []
    seq_y: list[float] = []
    meta_rows: list[dict] = []

    for asset, af in frame.sort_values([asset_col, "Date"]).groupby(asset_col, sort=False):
        af = af.sort_values("Date").reset_index(drop=True)
        feats = af[feature_cols].to_numpy(dtype="float32")
        targets = af[target_col].to_numpy(dtype="float32")
        dates = pd.to_datetime(af["Date"]).to_numpy()
        splits = af[split_col].astype(str).to_numpy() if split_col and split_col in af.columns else np.full(len(af), "", dtype=object)

        for end in range(lookback - 1, len(af)):
            if allowed is not None and splits[end] not in allowed:
                continue
            if np.isnan(targets[end]):
                continue
            window = feats[end - lookback + 1 : end + 1]
            if np.isnan(window).any():
                continue
            seq_x.append(window)
            seq_y.append(float(targets[end]))
            meta_rows.append({"Date": pd.Timestamp(dates[end]), "asset": str(asset), "split": str(splits[end])})

    if not seq_x:
        return (
            np.empty((0, lookback, len(feature_cols)), dtype="float32"),
            np.empty(0, dtype="float32"),
            pd.DataFrame(columns=["Date", "asset", "split"]),
        )
    return np.stack(seq_x, axis=0), np.array(seq_y, dtype="float32"), pd.DataFrame(meta_rows)
