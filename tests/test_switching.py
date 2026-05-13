import numpy as np
import pandas as pd

from aiam.evaluation.switch_assembly import assemble_switch_returns


def test_assemble_switch_returns_synthetic():
    """Verify regime dispatch: regime looked up at BDay-1 before each return date.

    The function mirrors the horse-race convention: weights are set the day before
    the return is recorded, so the regime must be labeled on the rebalancing dates
    (one business day before each return date) to produce the expected dispatch.
    """
    dates = pd.bdate_range("2020-01-02", periods=20)
    rng = np.random.default_rng(7)
    ret_a = rng.normal(0.001, 0.005, 20)
    ret_b = rng.normal(-0.002, 0.008, 20)
    wide = pd.DataFrame({"A": ret_a, "B": ret_b}, index=dates)

    # Label regimes on rebalancing dates (one bday before each return date) so that
    # return dates[i] dispatches using regime[reb_dates[i]].
    reb_dates = dates - pd.offsets.BDay(1)
    regime = pd.Series([0] * 10 + [1] * 10, index=reb_dates)

    rule = {0: "A", 1: "B"}
    result = assemble_switch_returns(wide, regime, rule=rule, default_strategy="A")

    expected = pd.Series(np.concatenate([ret_a[:10], ret_b[10:]]), index=dates)
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_assemble_switch_returns_default_fallback():
    """Dates whose regime is not in the rule fall back to default_strategy."""
    dates = pd.bdate_range("2020-01-02", periods=10)
    rng = np.random.default_rng(42)
    ret_a = rng.normal(0.001, 0.005, 10)
    ret_b = rng.normal(-0.001, 0.005, 10)
    wide = pd.DataFrame({"A": ret_a, "B": ret_b}, index=dates)

    # Regime 99 has no rule entry → should use default "B"
    regime = pd.Series([99] * 10, index=dates)

    result = assemble_switch_returns(wide, regime, rule={0: "A"}, default_strategy="B")

    pd.testing.assert_series_equal(result, pd.Series(ret_b, index=dates), check_names=False)
