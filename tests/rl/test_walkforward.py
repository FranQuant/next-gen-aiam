"""Interface tests for src/aiam/rl/walkforward.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiam.dl.walkforward import generate_refit_dates
from aiam.rl.env import PortfolioEnv
from aiam.rl.policy import SimplexPolicy
from aiam.rl.agent import RLAgent
from aiam.rl.trainer import TrainConfig, TrainHistory
from aiam.rl.walkforward import (
    RLRefitResult,
    WalkForwardRLEnsemble,
    fit_walkforward_rl,
)

N = 5
T = 700   # ~2.8 years of business days from 2019-01-01 → ends ~2021-09
LOOKBACK = 20
SEEDS = [0, 1]


@pytest.fixture
def returns_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        rng.standard_normal((T, N)) * 0.01,
        index=pd.bdate_range("2019-01-01", periods=T),
        columns=[f"A{i}" for i in range(N)],
    )


@pytest.fixture
def minimal_ensemble(returns_df) -> WalkForwardRLEnsemble:
    """A WalkForwardRLEnsemble with two refits, two seeds each."""
    policy0 = SimplexPolicy(n_features=LOOKBACK, hidden_dim=16)
    policy1 = SimplexPolicy(n_features=LOOKBACK, hidden_dim=16)
    agent0 = RLAgent(policy=policy0, lookback=LOOKBACK, seed=0)
    agent1 = RLAgent(policy=policy1, lookback=LOOKBACK, seed=1)

    dummy_hist = TrainHistory(episode_rewards=[0.0], mean_turnovers=[0.0], mean_weights=[])
    refit0 = RLRefitResult(
        refit_date=pd.Timestamp("2020-01-01"),
        agents=[agent0],
        histories=[dummy_hist],
    )
    refit1 = RLRefitResult(
        refit_date=pd.Timestamp("2020-07-01"),
        agents=[agent1],
        histories=[dummy_hist],
    )
    return WalkForwardRLEnsemble(refit_results=[refit0, refit1])


# ── generate_refit_dates ─────────────────────────────────────────────────────

def test_generate_refit_dates_monthly_count():
    """Monthly refit over 1 year should produce approximately 12-13 dates."""
    start = pd.Timestamp("2023-01-01")
    end = pd.Timestamp("2023-12-31")
    dates = generate_refit_dates(start, end, cadence="monthly")
    assert 12 <= len(dates) <= 14


def test_generate_refit_dates_start_before_or_at_test_start():
    start = pd.Timestamp("2023-01-01")
    end = pd.Timestamp("2023-06-30")
    dates = generate_refit_dates(start, end, cadence="monthly")
    assert dates[0] <= start
    assert all(d <= end for d in dates)


# ── WalkForwardRLEnsemble routing ────────────────────────────────────────────

def test_agents_for_date_routes_correctly(minimal_ensemble):
    """Dates before second refit should route to first refit's agents."""
    early = pd.Timestamp("2020-03-01")
    late = pd.Timestamp("2020-09-01")
    early_agents = minimal_ensemble._agents_for_date(early)
    late_agents = minimal_ensemble._agents_for_date(late)
    assert early_agents is not late_agents


def test_agents_for_date_raises_before_first_refit(minimal_ensemble):
    with pytest.raises(ValueError, match="precedes earliest refit"):
        minimal_ensemble._agents_for_date(pd.Timestamp("2019-01-01"))


# ── predict_weights_for_state ────────────────────────────────────────────────

def test_predict_weights_returns_simplex_vector(minimal_ensemble):
    feat = np.zeros((N, LOOKBACK), dtype=np.float32)
    prev_w = np.ones(N, dtype=np.float32) / N
    state = {"features": feat, "weights": prev_w}
    date = pd.Timestamp("2020-03-01")
    w = minimal_ensemble.predict_weights_for_state(state, date)
    assert w.shape == (N,)
    assert np.isclose(w.sum(), 1.0, atol=1e-5)
    assert np.all(w >= 0)


# ── evaluate_oos ─────────────────────────────────────────────────────────────

def test_evaluate_oos_shape(minimal_ensemble, returns_df):
    oos_start = pd.Timestamp("2020-08-01")
    oos_end = pd.Timestamp("2020-12-31")
    ret_series, weights_df, diags = minimal_ensemble.evaluate_oos(
        returns_df, oos_start, oos_end, lookback=LOOKBACK
    )
    n_oos = len(returns_df.loc[oos_start:oos_end])
    assert len(ret_series) == n_oos
    assert weights_df.shape == (n_oos, N)
    assert np.isclose(weights_df.sum(axis=1).mean(), 1.0, atol=1e-4)


def test_evaluate_oos_diagnostics_keys(minimal_ensemble, returns_df):
    oos_start = pd.Timestamp("2020-08-01")
    oos_end = pd.Timestamp("2020-12-31")
    _, _, diags = minimal_ensemble.evaluate_oos(
        returns_df, oos_start, oos_end, lookback=LOOKBACK
    )
    assert "mean_turnover" in diags
    assert "weight_std_across_time" in diags
    assert "n_oos_days" in diags
    assert diags["mean_turnover"] >= 0
    assert diags["weight_std_across_time"] >= 0


# ── lambda_risk env parameter ────────────────────────────────────────────────

def test_env_lambda_risk_zero_removes_penalty(returns_df):
    """With lambda_risk=0.0, risk_penalty should be 0 in every step."""
    env = PortfolioEnv(returns_df.iloc[:50], lambda_risk=0.0)
    env.reset()
    w = np.ones(N, dtype=np.float32) / N
    _, _, _, info = env.step(w)
    assert info["risk_penalty"] == 0.0


def test_env_lambda_risk_default_matches_constant(returns_df):
    """Default lambda_risk should match the module-level LAMBDA_RISK constant."""
    from aiam.rl.env import LAMBDA_RISK
    env = PortfolioEnv(returns_df.iloc[:50])
    assert env._lambda_risk == LAMBDA_RISK


# ── fit_walkforward_rl smoke test ─────────────────────────────────────────────

def test_fit_walkforward_rl_smoke(returns_df):
    """Minimal fit: 1 refit, 2 seeds, 3 episodes, 10 steps. Should not raise."""
    refit_dates = [pd.Timestamp("2020-06-01")]
    config = TrainConfig(episodes=3, max_steps_per_episode=10, seed=0)
    ensemble = fit_walkforward_rl(
        returns_df,
        refit_dates=refit_dates,
        config=config,
        seeds=[0, 1],
        hidden_dim=16,
        training_window_months=12,
        lambda_risk=0.02,
        verbose=False,
    )
    assert len(ensemble.refit_results) == 1
    assert len(ensemble.refit_results[0].agents) == 2
    assert len(ensemble.refit_results[0].histories) == 2
