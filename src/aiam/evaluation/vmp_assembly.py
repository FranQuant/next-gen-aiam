from __future__ import annotations

import numpy as np
import pandas as pd

_SQRT252 = np.sqrt(252)


def assemble_vmp_returns(
    cached_returns: pd.Series,
    target_vol: float | None = None,
    clip: tuple[float, float] = (0.25, 1.5),
    lookback: int = 21,
    lag: int = 1,
    rf: float = 0.0,
) -> pd.Series:
    """
    Apply Moreira-Muir 2017 volatility management to a base strategy's returns.

    exposure_t = clip( target_vol / realized_vol_{t-lag}, *clip )
    vmp_return_t = exposure_t * (cached_return_t - rf/252) + rf/252

    If target_vol is None, defaults to cached_returns.std() * sqrt(252) — the
    strategy's own long-run realized vol, making VMP a vol-stabilized version
    rather than a vol-targeted-to-X version.
    """
    if target_vol is None:
        target_vol = cached_returns.std() * _SQRT252

    realized_vol = cached_returns.rolling(lookback).std() * _SQRT252

    # shift(lag) enforces no look-ahead; first (lookback + lag - 1) rows are NaN
    # Fill NaN with 1.0 — neutral exposure, so those days use the base return unchanged.
    exposure = (target_vol / realized_vol).shift(lag).clip(*clip).fillna(1.0)

    excess_return = cached_returns - rf / 252
    return exposure * excess_return + rf / 252
