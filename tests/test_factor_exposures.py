"""Tests for factor_exposures.py.

These tests focus on two things:
- The rolling regression produces the expected factor loadings when the data is
  perfectly constructed (a clean beta case).
- The function behaves sensibly on edge cases (not enough history, wrong index).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Tests run from various working directories (CI, local, notebooks). Adding the project
# root to sys.path makes imports deterministic without requiring an installed package.
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import src.factor_exposures as fe


def test_compute_factor_exposures_lagged_betas():
    # Use a tiny, controlled monthly grid so we can reason exactly about what the
    # rolling window should be able to fit and when lagged betas should appear.
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")

    # Build factors so the regression is essentially one-dimensional:
    # keeping SMB/HML/RF at zero isolates MKT, which makes the “true beta” unambiguous.
    factors = pd.DataFrame(
        {
            "MKT": [0.01, 0.02, 0.03, 0.04, 0.05],
            "SMB": 0.0,
            "HML": 0.0,
            "RF": 0.0,
        },
        index=dates,
    )

    # Create returns with a known data-generating process (RET = 2 * MKT).
    # This lets the test validate the regression numerics without relying on
    # approximate “real-world” behavior.
    returns = pd.DataFrame(
        {"RET": (2 * factors["MKT"]).values},
        index=pd.MultiIndex.from_product([["AAA"], dates], names=["ticker", "date"]),
    )

    # Use a short minimum window so the unit test produces output quickly, but still
    # requires enough observations to identify an intercept + beta in least squares.
    config = fe.ExposureConfig(min_months=3, max_months=5)
    result = fe.compute_factor_exposures(returns, factors, config=config)

    # The implementation lags exposures by one month to prevent look-ahead bias.
    # Dropping NaNs isolates the dates where the pipeline claims betas are “known”.
    non_na = result.dropna()

    # With min_months=3, the first regression can only be fit at the 3rd observation,
    # and the one-period lag means the first *usable* beta arrives one month later.
    assert list(non_na.index.get_level_values("date")) == [dates[3], dates[4]]

    # Verify the key identification claim: beta_MKT should recover the true value (≈ 2).
    # Using a tolerance avoids brittle failures due to floating point rounding.
    pd.testing.assert_series_equal(
        non_na["beta_MKT"],
        pd.Series(
            [2.0, 2.0],
            index=pd.MultiIndex.from_product([["AAA"], dates[3:]], names=["ticker", "date"]),
        ),
        check_names=False,
        check_exact=False,
        rtol=1e-6,
    )

    # Since SMB/HML never move in this synthetic setup, the model should assign them
    # effectively zero loading (numerical noise aside).
    assert (non_na["beta_SMB"].abs() < 1e-8).all()
    assert (non_na["beta_HML"].abs() < 1e-8).all()


def test_compute_factor_exposures_not_enough_history_returns_empty():
    # Keep the sample intentionally too short relative to min_months so we test the
    # “fail gracefully” behavior instead of producing partial/unstable estimates.
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")

    # Provide the required columns even though the values are trivial; the goal here
    # is to verify window-length guarding, not factor variability.
    factors = pd.DataFrame({"MKT": 0.0, "SMB": 0.0, "HML": 0.0, "RF": 0.0}, index=dates)

    # Minimal well-formed returns panel: correct MultiIndex + required return column.
    returns = pd.DataFrame(
        {"RET": 0.0},
        index=pd.MultiIndex.from_product([["AAA"], dates], names=["ticker", "date"]),
    )

    # Require more history than exists; the function should refuse to estimate rather
    # than emitting misleading betas from an under-identified window.
    config = fe.ExposureConfig(min_months=4, max_months=5)
    result = fe.compute_factor_exposures(returns, factors, config=config)

    # Empty output with a stable schema is important because downstream joins/merges
    # should not crash just because a ticker/date range is too short.
    assert result.empty
    assert list(result.columns) == ["alpha_ff3", "beta_MKT", "beta_SMB", "beta_HML"]


def test_compute_factor_exposures_requires_multiindex():
    # The exposure code groups by ticker and aligns on date; without a (ticker, date)
    # MultiIndex, any results would be ambiguous and likely silently wrong.
    factors = pd.DataFrame({"MKT": [], "SMB": [], "HML": [], "RF": []})
    returns = pd.DataFrame({"RET": []})

    # We want an explicit, user-facing error instead of accidental “success” on a
    # malformed input (which would be very hard to debug downstream).
    with pytest.raises(ValueError):
        fe.compute_factor_exposures(returns, factors)