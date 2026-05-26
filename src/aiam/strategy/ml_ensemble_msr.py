"""Deterministic MSR ensemble runner for Notebook 03's ML winner."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aiam.estimators.covariance import ledoit_wolf_cov
from aiam.features.asset_class import asset_class_one_hot
from aiam.features.technical import atr, bollinger, gap, momentum, rsi, volatility, volume_signal
from aiam.strategy.ml_strategies import LassoSignalStrategy, RFSignalStrategy, XGBSignalStrategy


FEATURE_COLUMNS = [
    "mom_21",
    "mom_63",
    "mom_252",
    "vol_60",
    "vol_252",
    "rsi_14",
    "atr_14_ratio",
    "bb_pct",
    "gap",
    "vol_signal_21",
    "us_single_stock",
    "us_sector_etf",
    "broad_equity_etf",
    "intl_equity_etf",
    "fixed_income_etf",
    "commodity_etf",
    "fx_spot",
]


@dataclass(frozen=True)
class MLEnsembleMSRConfig:
    train_end: str = "2022-12-31"
    test_start: str = "2023-01-01"
    horizon: int = 21
    validation_share: float = 0.15
    cov_lookback: int = 504
    min_assets: int = 5
    ridge: float = 1e-8
    random_state: int = 42


def _as_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.index = pd.to_datetime(result.index)
    result = result.sort_index()
    result.index.name = "Date"
    result.columns.name = "Asset"
    return result


def _wide_ohlcv(ohlcv: pd.DataFrame | dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if isinstance(ohlcv, dict):
        return {k: _as_datetime_index(v) for k, v in ohlcv.items()}

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in ohlcv.columns]
    if missing:
        raise ValueError(f"OHLCV input missing columns: {missing}")

    if not isinstance(ohlcv.index, pd.MultiIndex):
        raise ValueError("OHLCV input must use a MultiIndex with date and asset levels")

    names = list(ohlcv.index.names)
    asset_level = "Asset" if "Asset" in names else "ticker" if "ticker" in names else names[1]

    result = {}
    for col in required:
        wide = ohlcv[col].unstack(asset_level)
        wide.index = pd.to_datetime(wide.index)
        wide = wide.sort_index()
        wide.index.name = "Date"
        wide.columns.name = "Asset"
        result[col] = wide
    return result


def build_ml_feature_panel_from_ohlcv(
    returns: pd.DataFrame,
    ohlcv: pd.DataFrame | dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build Notebook 03's 17-column feature panel from cached OHLCV and returns."""
    rets = _as_datetime_index(returns)
    ohlc = _wide_ohlcv(ohlcv)

    numeric_frames = {
        "mom_21": momentum(rets, 21),
        "mom_63": momentum(rets, 63),
        "mom_252": momentum(rets, 252),
        "vol_60": volatility(rets, 60),
        "vol_252": volatility(rets, 252),
        "rsi_14": rsi(ohlc["close"], 14),
        "atr_14_ratio": atr(ohlc, 14) / ohlc["close"],
        "bb_pct": bollinger(ohlc["close"], window=20)["pct"],
        "gap": gap(ohlc),
        "vol_signal_21": volume_signal(ohlc["volume"], lookback=21),
    }
    panel_numeric = pd.concat({k: v.stack() for k, v in numeric_frames.items()}, axis=1)
    panel_numeric.index.names = ["Date", "Asset"]

    one_hot = asset_class_one_hot(rets.columns)
    feature_panel = panel_numeric.join(one_hot, on="Asset")
    feature_panel = feature_panel.dropna(subset=["mom_252", "vol_252"])
    return feature_panel[FEATURE_COLUMNS]


def build_target_21d(returns: pd.DataFrame, horizon: int = 21) -> pd.Series:
    """Forward cumulative returns: features at t predict t+1 through t+horizon."""
    rets = _as_datetime_index(returns)
    target = rets.shift(-horizon).rolling(horizon).sum().stack()
    target.index.names = ["Date", "Asset"]
    target.name = f"target_{horizon}d"
    return target


def fit_base_ml_strategies(
    features: pd.DataFrame,
    target: pd.Series,
    config: MLEnsembleMSRConfig,
) -> dict[str, Any]:
    """Fit the three Notebook 03 base forecasters and return their strategy objects."""
    common_idx = features.index.intersection(target.dropna().index)
    X = features.loc[common_idx, FEATURE_COLUMNS].dropna()
    y = target.loc[X.index]
    if X.empty:
        raise ValueError("No aligned non-null feature rows are available for fitting")

    try:
        import xgboost  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("XGBoost is required for MSR(Ensemble_mu_hat)") from exc

    return {
        "lasso": LassoSignalStrategy(
            X,
            y,
            FEATURE_COLUMNS,
            train_end=config.train_end,
            validation_share=config.validation_share,
            alpha=1e-4,
        ),
        "rf": RFSignalStrategy(
            X,
            y,
            FEATURE_COLUMNS,
            train_end=config.train_end,
            validation_share=config.validation_share,
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=50,
        ),
        "xgb": XGBSignalStrategy(
            X,
            y,
            FEATURE_COLUMNS,
            train_end=config.train_end,
            validation_share=config.validation_share,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
        ),
    }


def build_ensemble_predictions(predictions: dict[str, pd.Series]) -> pd.Series:
    """Equal-weighted average of lasso, RF, and XGB expected-return predictions."""
    required = ["lasso", "rf", "xgb"]
    missing = [name for name in required if name not in predictions]
    if missing:
        raise ValueError(f"Missing prediction series: {missing}")
    ensemble = (predictions["lasso"] + predictions["rf"] + predictions["xgb"]) / 3.0
    ensemble = ensemble.dropna().sort_index()
    ensemble.index.names = ["Date", "Asset"]
    ensemble.name = "ensemble_pred"
    return ensemble


def _equal_weights(assets: pd.Index) -> pd.Series:
    return pd.Series(1.0 / len(assets), index=assets, dtype=float)


def compute_msr_weights_from_mu(
    predictions: pd.Series,
    returns: pd.DataFrame,
    config: MLEnsembleMSRConfig,
) -> pd.DataFrame:
    """Compute long-only MSR approximation weights from ML expected returns."""
    rets = _as_datetime_index(returns)
    dates = predictions.index.get_level_values(0).unique().sort_values()
    rows: dict[pd.Timestamp, pd.Series] = {}

    for date in dates:
        mu_hat = predictions.xs(date, level=0).replace([np.inf, -np.inf], np.nan).dropna()
        hist_window = rets.loc[rets.index <= date].tail(config.cov_lookback).dropna(axis=1, how="all")
        common = hist_window.columns.intersection(mu_hat.index)
        if len(common) < config.min_assets:
            continue

        clean_returns = hist_window.loc[:, common].dropna(how="any")
        if len(clean_returns) < max(2, len(common)):
            continue

        mu = mu_hat.loc[common].to_numpy(dtype=float)
        if not np.isfinite(mu).all():
            continue

        try:
            cov = ledoit_wolf_cov(clean_returns)
            cov_inv = np.linalg.pinv(cov + config.ridge * np.eye(len(common)))
            raw_weights = cov_inv @ mu
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            raw_weights = np.full(len(common), np.nan)

        clipped = pd.Series(raw_weights, index=common, dtype=float).clip(lower=0.0)
        if not np.isfinite(clipped.to_numpy()).all() or clipped.sum() <= 1e-12:
            weight = _equal_weights(common)
        else:
            weight = clipped / clipped.sum()
        rows[pd.Timestamp(date)] = weight.reindex(rets.columns).fillna(0.0)

    weights = pd.DataFrame.from_dict(rows, orient="index", columns=rets.columns)
    if not weights.empty:
        weights.index.name = "Date"
        weights.columns.name = "Asset"
        weights = weights.sort_index()
    return weights


def backtest_lagged_weights(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Apply weights from each signal date to the next available return row."""
    rets = _as_datetime_index(returns)
    weights = weights.sort_index()
    realized: dict[pd.Timestamp, float] = {}
    for date, weight in weights.iterrows():
        next_dates = rets.index[rets.index > date]
        if len(next_dates) == 0:
            continue
        next_date = next_dates[0]
        next_returns = rets.loc[next_date]
        realized[next_date] = float((weight.reindex(rets.columns).fillna(0.0) * next_returns).sum())
    result = pd.Series(realized, name="MSR(Ensemble_mu_hat)")
    result.index.name = "Date"
    return result.sort_index()


def compute_performance_metrics(returns: pd.Series) -> dict[str, float]:
    """Performance metrics using 252 sessions per year.

    ``annual_return`` is retained as a backward-compatible alias for
    ``annual_return_arithmetic``. It is not CAGR.
    """
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {
            "annual_return": np.nan,
            "annual_return_arithmetic": np.nan,
            "cagr": np.nan,
            "annual_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "total_return": np.nan,
            "observations": 0.0,
        }

    annual_return_arithmetic = float(clean.mean() * 252.0)
    annual_volatility = float(clean.std() * np.sqrt(252.0))
    sharpe = annual_return_arithmetic / annual_volatility if annual_volatility > 0 else np.nan
    cumulative = (1.0 + clean).cumprod()
    final_wealth = float(cumulative.iloc[-1])
    total_return = final_wealth - 1.0
    cagr = final_wealth ** (252.0 / len(clean)) - 1.0 if final_wealth > 0 else np.nan
    drawdown = cumulative / cumulative.cummax() - 1.0
    return {
        "annual_return": annual_return_arithmetic,
        "annual_return_arithmetic": annual_return_arithmetic,
        "cagr": float(cagr),
        "annual_volatility": annual_volatility,
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "total_return": float(total_return),
        "observations": float(len(clean)),
    }


def compute_turnover_diagnostics(weights: pd.DataFrame) -> dict[str, float]:
    """Summarize one-way turnover from consecutive weight vectors."""
    nan_result = {
        "average_turnover": np.nan,
        "median_turnover": np.nan,
        "max_turnover": np.nan,
        "observations": 0.0,
    }
    if len(weights) < 2:
        return nan_result

    aligned = weights.sort_index().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    turnover = 0.5 * aligned.diff().abs().sum(axis=1).iloc[1:]
    if turnover.empty:
        return nan_result
    return {
        "average_turnover": float(turnover.mean()),
        "median_turnover": float(turnover.median()),
        "max_turnover": float(turnover.max()),
        "observations": float(len(turnover)),
    }


def compute_concentration_diagnostics(weights: pd.DataFrame) -> dict[str, float]:
    """Summarize descriptive concentration statistics for portfolio weights."""
    nan_result = {
        "average_herfindahl": np.nan,
        "average_effective_positions": np.nan,
        "average_max_weight": np.nan,
        "max_single_asset_weight": np.nan,
        "average_top_5_weight_share": np.nan,
        "max_top_5_weight_share": np.nan,
        "observations": 0.0,
    }
    if weights.empty:
        return nan_result

    clean = weights.sort_index().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    herfindahl = (clean**2).sum(axis=1)
    effective_positions = 1.0 / herfindahl.where(herfindahl > 0.0)
    max_weight = clean.max(axis=1)
    top_5_weight_share = clean.apply(lambda row: row.nlargest(5).sum(), axis=1)
    return {
        "average_herfindahl": float(herfindahl.mean()),
        "average_effective_positions": float(effective_positions.mean()),
        "average_max_weight": float(max_weight.mean()),
        "max_single_asset_weight": float(max_weight.max()),
        "average_top_5_weight_share": float(top_5_weight_share.mean()),
        "max_top_5_weight_share": float(top_5_weight_share.max()),
        "observations": float(len(clean)),
    }


def load_cached_inputs(
    ohlcv_path: str | Path,
    returns_path: str | Path,
    prices_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    data = {
        "ohlcv": pd.read_parquet(ohlcv_path),
        "returns": pd.read_parquet(returns_path),
    }
    if prices_path is not None:
        data["prices"] = pd.read_parquet(prices_path)
    return data


def run_ml_ensemble_msr_research(
    *,
    ohlcv: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
    returns: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    ohlcv_path: str | Path = "data/cache/prices_29_ohlcv_2003_2026.parquet",
    returns_path: str | Path = "data/cache/returns_29_2003_2026.parquet",
    prices_path: str | Path | None = "data/cache/prices_29.parquet",
    config: MLEnsembleMSRConfig | None = None,
    output_dir: str | Path | None = None,
    write_artifacts: bool = False,
) -> dict[str, Any]:
    """Run the deterministic cache-driven MSR(Ensemble_mu_hat) workflow."""
    cfg = config or MLEnsembleMSRConfig()
    if returns is None or ohlcv is None:
        loaded = load_cached_inputs(ohlcv_path, returns_path, prices_path)
        ohlcv = loaded["ohlcv"] if ohlcv is None else ohlcv
        returns = loaded["returns"] if returns is None else returns
        prices = loaded.get("prices", prices)

    returns = _as_datetime_index(returns)
    features = build_ml_feature_panel_from_ohlcv(returns, ohlcv)
    target = build_target_21d(returns, cfg.horizon)
    strategies = fit_base_ml_strategies(features, target, cfg)
    base_predictions = {name: strat.predictions for name, strat in strategies.items()}
    predictions = build_ensemble_predictions(base_predictions)
    predictions = predictions.loc[
        predictions.index.get_level_values("Date") >= pd.Timestamp(cfg.test_start)
    ]
    weights = compute_msr_weights_from_mu(predictions, returns, cfg)
    strategy_returns = backtest_lagged_weights(weights, returns)
    strategy_returns = strategy_returns.loc[strategy_returns.index >= pd.Timestamp(cfg.test_start)]
    metrics = compute_performance_metrics(strategy_returns)
    turnover_diagnostics = compute_turnover_diagnostics(weights)
    concentration_diagnostics = compute_concentration_diagnostics(weights)

    caveats = [
        "historical backtest only",
        "no investment advice",
        "no transaction costs in first version",
        "single-fit ML setup",
        "possible optimizer concentration",
        "local cache / EODHD cache dependence",
        "must verify against Notebook 03 published metrics",
        "CAGR and arithmetic annualized return are both reported",
        "Sharpe uses arithmetic annualized return over annualized volatility",
        "concentration diagnostics are descriptive; no constraint is imposed in this baseline",
    ]
    result = {
        "features": features,
        "target": target,
        "base_predictions": base_predictions,
        "predictions": predictions,
        "weights": weights,
        "strategy_returns": strategy_returns,
        "metrics": metrics,
        "turnover_diagnostics": turnover_diagnostics,
        "concentration_diagnostics": concentration_diagnostics,
        "caveats": caveats,
        "config": cfg,
        "universe_size": int(returns.shape[1]),
        "date_range": (str(returns.index.min().date()), str(returns.index.max().date())),
        "prices_shape": None if prices is None else tuple(prices.shape),
    }

    if write_artifacts:
        if output_dir is None:
            raise ValueError("output_dir is required when write_artifacts is True")
        write_research_artifacts(result, output_dir)

    return result


def write_research_artifacts(result: dict[str, Any], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result["predictions"].to_frame("ensemble_pred").to_parquet(out / "predictions.parquet")
    result["weights"].to_parquet(out / "weights.parquet")
    result["strategy_returns"].to_frame("return").to_parquet(out / "strategy_returns.parquet")
    metrics_payload = {
        "metrics": result["metrics"],
        "turnover_diagnostics": result["turnover_diagnostics"],
        "concentration_diagnostics": result["concentration_diagnostics"],
    }
    (out / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True))
    (out / "report.md").write_text(build_report(result))


def build_report(result: dict[str, Any]) -> str:
    cfg: MLEnsembleMSRConfig = result["config"]
    metrics = result["metrics"]
    turnover = result["turnover_diagnostics"]
    concentration = result["concentration_diagnostics"]
    performance_keys = [
        "annual_return_arithmetic",
        "cagr",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "total_return",
        "observations",
    ]
    turnover_keys = ["average_turnover", "median_turnover", "max_turnover", "observations"]
    concentration_keys = [
        "average_herfindahl",
        "average_effective_positions",
        "average_max_weight",
        "max_single_asset_weight",
        "average_top_5_weight_share",
        "max_top_5_weight_share",
        "observations",
    ]
    performance_rows = "\n".join(f"| {k} | {metrics[k]:.6g} |" for k in performance_keys)
    turnover_rows = "\n".join(f"| {k} | {turnover[k]:.6g} |" for k in turnover_keys)
    concentration_rows = "\n".join(f"| {k} | {concentration[k]:.6g} |" for k in concentration_keys)
    caveats = "\n".join(f"- {item}" for item in result["caveats"])
    return (
        "# ML Ensemble MSR Research Run\n\n"
        "- Strategy: MSR(Ensemble_mu_hat)\n"
        f"- Universe size: {result['universe_size']}\n"
        f"- Date range: {result['date_range'][0]} to {result['date_range'][1]}\n"
        f"- Train end: {cfg.train_end}\n"
        f"- Test start: {cfg.test_start}\n"
        f"- Feature count: {len(FEATURE_COLUMNS)}\n"
        "- Model components: Lasso, Random Forest, XGBoost\n"
        f"- Covariance lookback: {cfg.cov_lookback}\n\n"
        "## Performance Metrics\n\n"
        "Annual return is arithmetic annualized return. CAGR is geometric annualized return.\n\n"
        "| Metric | Value |\n"
        "| --- | ---: |\n"
        f"{performance_rows}\n\n"
        "## Turnover Diagnostics\n\n"
        "| Metric | Value |\n"
        "| --- | ---: |\n"
        f"{turnover_rows}\n\n"
        "## Concentration Diagnostics\n\n"
        "| Metric | Value |\n"
        "| --- | ---: |\n"
        f"{concentration_rows}\n\n"
        "## Caveats\n\n"
        f"{caveats}\n"
    )


def summary_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy": "MSR(Ensemble_mu_hat)",
        "universe_size": result["universe_size"],
        "date_range": result["date_range"],
        "train_end": result["config"].train_end,
        "test_start": result["config"].test_start,
        "feature_count": len(FEATURE_COLUMNS),
        "cov_lookback": result["config"].cov_lookback,
        "metrics": result["metrics"],
        "turnover_diagnostics": result["turnover_diagnostics"],
        "concentration_diagnostics": result["concentration_diagnostics"],
        "observations": int(result["metrics"]["observations"]),
    }
