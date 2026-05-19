"""Walk-forward RL adapter: one RLAgent per (refit_date, seed), ensemble-averaged OOS."""
from __future__ import annotations

import bisect
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aiam.dl.walkforward import generate_refit_dates  # noqa: reuse DL harness date logic
from aiam.rl.agent import RLAgent
from aiam.rl.env import COST_BPS, PortfolioEnv
from aiam.rl.policy import SimplexPolicy
from aiam.rl.trainer import TrainConfig, TrainHistory, train

__all__ = [
    "RLRefitResult",
    "WalkForwardRLEnsemble",
    "fit_walkforward_rl",
    "generate_refit_dates",
]


@dataclass
class RLRefitResult:
    """Trained agents + histories for one refit date."""
    refit_date: pd.Timestamp
    agents: list[RLAgent]
    histories: list[TrainHistory]


@dataclass
class WalkForwardRLEnsemble:
    """Stores RLRefitResults, routes predictions by date, ensemble-averages across seeds."""

    refit_results: list[RLRefitResult]  # sorted ascending by refit_date

    @property
    def refit_dates(self) -> list[pd.Timestamp]:
        return [r.refit_date for r in self.refit_results]

    def _agents_for_date(self, date: pd.Timestamp) -> list[RLAgent]:
        idx = bisect.bisect_right(self.refit_dates, date) - 1
        if idx < 0:
            raise ValueError(
                f"date {date.date()} precedes earliest refit {self.refit_dates[0].date()}"
            )
        return self.refit_results[idx].agents

    def predict_weights_for_state(
        self, state: dict, date: pd.Timestamp
    ) -> np.ndarray:
        """Ensemble-average deterministic weights across seeds for a given date."""
        agents = self._agents_for_date(date)
        weights = np.stack([a.policy.act(state) for a in agents])  # (S, N)
        return weights.mean(axis=0)

    def evaluate_oos(
        self,
        returns: pd.DataFrame,
        oos_start: pd.Timestamp,
        oos_end: pd.Timestamp,
        lookback: int = 20,
    ) -> tuple[pd.Series, pd.DataFrame, dict]:
        """Evaluate ensemble OOS. Returns (net_returns, weights_df, diagnostics)."""
        assets = returns.columns.tolist()
        N = len(assets)
        returns_arr = returns.values.astype(np.float32)       # (T_full, N)
        date_to_pos = {d: i for i, d in enumerate(returns.index)}

        oos_returns = returns.loc[oos_start:oos_end]

        net_rets, weight_rows, turnovers, dates = [], [], [], []
        prev_w = np.ones(N, dtype=np.float32) / N

        for date in oos_returns.index:
            pos = date_to_pos[date]
            start_pos = max(0, pos - lookback)
            trailing = returns_arr[start_pos:pos]             # (≤lookback, N)
            if len(trailing) >= lookback:
                feat = trailing[-lookback:].T.astype(np.float32)  # (N, lookback)
            else:
                feat = np.zeros((N, lookback), dtype=np.float32)

            state = {"features": feat, "weights": prev_w}
            w = self.predict_weights_for_state(state, date)
            w = np.clip(w, 0.0, None)
            w = w / (w.sum() + 1e-12)

            r = oos_returns.loc[date].values.astype(np.float32)
            gross_ret = float(w @ r)
            turnover = float(np.abs(w - prev_w).sum())
            tc = COST_BPS / 10_000.0 * turnover
            net_rets.append(gross_ret - tc)
            weight_rows.append(w.copy())
            turnovers.append(turnover)
            dates.append(date)
            prev_w = w

        weights_df = pd.DataFrame(weight_rows, index=dates, columns=assets)
        ret_series = pd.Series(net_rets, index=dates, name="rl_ensemble")

        diagnostics = {
            "mean_turnover": float(np.mean(turnovers)),
            "weight_std_across_time": float(weights_df.std(axis=0).mean()),
            "n_oos_days": len(dates),
        }
        return ret_series, weights_df, diagnostics


def fit_walkforward_rl(
    returns: pd.DataFrame,
    refit_dates: list[pd.Timestamp],
    config: TrainConfig,
    seeds: Sequence[int],
    hidden_dim: int = 32,
    training_window_months: int = 24,
    lambda_risk: float = 0.02,
    verbose: bool = False,
) -> WalkForwardRLEnsemble:
    """Train one RLAgent per (refit_date, seed) on a trailing returns window.

    For each refit_date D:
      training window = [D − training_window_months, D − 1 day]
    """
    lookback: int = config.max_steps_per_episode if config.max_steps_per_episode else 20
    # Infer lookback from env default (20) — features = rolling-return windows.
    n_features: int = 20
    refit_results: list[RLRefitResult] = []

    for i, refit_date in enumerate(refit_dates):
        train_end = refit_date - pd.Timedelta(days=1)
        train_start = refit_date - pd.DateOffset(months=training_window_months)

        train_returns = returns.loc[
            (returns.index >= train_start) & (returns.index <= train_end)
        ]

        if len(train_returns) < 50:
            if verbose:
                print(f"  Refit {i+1}/{len(refit_dates)}: skipped ({len(train_returns)} rows)")
            continue

        agents: list[RLAgent] = []
        histories: list[TrainHistory] = []
        t0 = time.time()

        for seed in seeds:
            policy = SimplexPolicy(n_features=n_features, hidden_dim=hidden_dim)
            env = PortfolioEnv(train_returns, lambda_risk=lambda_risk)
            seed_config = TrainConfig(
                episodes=config.episodes,
                gamma=config.gamma,
                lr=config.lr,
                entropy_coef=config.entropy_coef,
                grad_clip=config.grad_clip,
                seed=seed,
                max_steps_per_episode=config.max_steps_per_episode,
            )
            history, value_head = train(policy, env, seed_config)
            agent = RLAgent(policy=policy, lookback=n_features, seed=seed)
            agent.history = history
            agent._value_head = value_head
            agents.append(agent)
            histories.append(history)

        refit_results.append(RLRefitResult(
            refit_date=refit_date,
            agents=agents,
            histories=histories,
        ))

        if verbose:
            mean_final_reward = float(np.mean([h.episode_rewards[-1] for h in histories]))
            mean_turnover = float(np.mean([np.mean(h.mean_turnovers[-10:]) for h in histories]))
            print(
                f"  Refit {i+1:2d}/{len(refit_dates)}: "
                f"{train_start.date()} → {train_end.date()} "
                f"({len(train_returns)} days) | "
                f"seeds={len(seeds)} | "
                f"final_reward={mean_final_reward:.5f} | "
                f"mean_to={mean_turnover:.4f} | "
                f"{time.time()-t0:.1f}s"
            )

    return WalkForwardRLEnsemble(refit_results=refit_results)
