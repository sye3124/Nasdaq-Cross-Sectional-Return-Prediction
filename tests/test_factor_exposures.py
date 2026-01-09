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

# Add the project root to sys.path so tests can import local modules directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import src.factor_exposures as fe


def test_compute_factor_exposures_lagged_betas():
    # Use a short monthly sample; "ME" makes these month-end timestamps.
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")

    # Build a simple factor panel:
    # - MKT varies each month
    # - SMB/HML/RF are set to zero so only MKT matters
    factors = pd.DataFrame(
        {
            "MKT": [0.01, 0.02, 0.03, 0.04, 0.05],
            "SMB": 0.0,
            "HML": 0.0,
            "RF": 0.0,
        },
        index=dates,
    )

    # Construct returns so excess returns are exactly 2 * MKT.
    # With an intercept in the regression and RF=0, we expect beta_MKT ≈ 2.0.
    returns = pd.DataFrame(
        {"RET": (2 * factors["MKT"]).values},
        index=pd.MultiIndex.from_product([["AAA"], dates], names=["ticker", "date"]),
    )

    # Need at least 3 months to fit a regression; allow up to 5 months in the window.
    config = fe.ExposureConfig(min_months=3, max_months=5)
    result = fe.compute_factor_exposures(returns, factors, config=config)

    # Exposures are shifted by one period, so the first usable beta appears later.
    non_na = result.dropna()

    # With 5 months total and min_months=3, we can fit starting at month 3,
    # but the one-period lag means we only *observe* betas for the last 2 dates.
    assert list(non_na.index.get_level_values("date")) == [dates[3], dates[4]]

    # beta_MKT should be ~2.0 for both available months.
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

    # Other factors are always zero, so their betas should be ~0.
    assert (non_na["beta_SMB"].abs() < 1e-8).all()
    assert (non_na["beta_HML"].abs() < 1e-8).all()


def test_compute_factor_exposures_not_enough_history_returns_empty():
    # Only 3 months of data, but we'll require at least 4 months to fit.
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")

    # Factors exist but carry no variation (still a valid input shape).
    factors = pd.DataFrame({"MKT": 0.0, "SMB": 0.0, "HML": 0.0, "RF": 0.0}, index=dates)

    # Returns for a single ticker across the same dates.
    returns = pd.DataFrame(
        {"RET": 0.0},
        index=pd.MultiIndex.from_product([["AAA"], dates], names=["ticker", "date"]),
    )

    # Require more history than we have.
    config = fe.ExposureConfig(min_months=4, max_months=5)
    result = fe.compute_factor_exposures(returns, factors, config=config)

    # Should return an empty DataFrame with the expected output columns.
    assert result.empty
    assert list(result.columns) == ["alpha_ff3", "beta_MKT", "beta_SMB", "beta_HML"]


def test_compute_factor_exposures_requires_multiindex():
    # The implementation requires returns to be indexed by (ticker, date).
    factors = pd.DataFrame({"MKT": [], "SMB": [], "HML": [], "RF": []})
    returns = pd.DataFrame({"RET": []})

    # Without the required MultiIndex, we expect a clear ValueError.
    with pytest.raises(ValueError):
        fe.compute_factor_exposures(returns, factors)