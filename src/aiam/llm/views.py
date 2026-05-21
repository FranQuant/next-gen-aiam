from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from aiam.llm.cache import PromptCache
from aiam.llm.client import LLMClient
from aiam.llm.evidence import build_evidence, evidence_to_text
from aiam.llm.prompts import SYSTEM_PROMPT, build_prompt
from aiam.llm.schemas import ParseError, parse_viewset

logger = logging.getLogger(__name__)

_EMPTY = lambda n: (np.zeros((0, n)), np.array([]), np.zeros((0, 0)))  # noqa: E731


class LLMViewGenerator:
    """Black-Litterman view generator backed by an LLM client."""

    def __init__(
        self,
        client: LLMClient,
        *,
        view_uncertainty_scaler: float = 0.05,
        lookbacks: tuple[int, ...] = (21, 63, 252),
        cache: PromptCache | None = None,
        strict: bool = False,
    ) -> None:
        self._client = client
        self._view_uncertainty_scaler = view_uncertainty_scaler
        self._lookbacks = lookbacks
        self._cache = cache
        self._strict = strict

    def __call__(
        self,
        returns: pd.DataFrame,
        asof: pd.Timestamp,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (P, Q, Omega) aligned to returns.columns; empty arrays on failure."""
        n = returns.shape[1]
        universe = list(returns.columns)

        try:
            evidence_df = build_evidence(returns, asof, lookbacks=self._lookbacks)
            evidence_text = evidence_to_text(evidence_df)
            prompt = build_prompt(evidence_text, asof, universe)

            model_id = getattr(self._client, "model", "mock")
            raw = self._cache.get(model_id, SYSTEM_PROMPT, prompt) if self._cache else None

            if raw is None:
                raw = self._client.complete(prompt, system=SYSTEM_PROMPT)
                if self._cache:
                    self._cache.set(model_id, SYSTEM_PROMPT, prompt, raw)

            logger.debug("LLMViewGenerator asof=%s raw_chars=%d", asof.date(), len(raw))
            viewset = parse_viewset(raw)

        except ParseError as exc:
            logger.warning("LLMViewGenerator parse failure at asof=%s: %s", asof.date(), exc)
            if self._strict:
                raise
            return _EMPTY(n)
        except Exception as exc:
            logger.warning("LLMViewGenerator failure at asof=%s: %s", asof.date(), exc)
            if self._strict:
                raise
            return _EMPTY(n)

        col_idx = {c: i for i, c in enumerate(returns.columns)}
        valid_views = [v for v in viewset.views if v.asset in col_idx]
        if not valid_views:
            return _EMPTY(n)

        k = len(valid_views)
        P = np.zeros((k, n))
        Q = np.zeros(k)
        omega_diag = np.zeros(k)
        daily_var = returns.var()

        for i, view in enumerate(valid_views):
            j = col_idx[view.asset]
            P[i, j] = 1.0
            Q[i] = view.expected_excess_return
            # Monotonic: higher confidence → lower uncertainty
            omega_diag[i] = (
                float(daily_var.iloc[j])
                * self._view_uncertainty_scaler
                * (1.0 - view.confidence + 1e-4)
            )

        return P, Q, np.diag(omega_diag)
