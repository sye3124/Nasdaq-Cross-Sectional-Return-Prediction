"""Tests for final_dataset.py.

These tests verify that `assemble_modeling_dataset()` correctly:
- creates the next-period return target by shifting returns within each ticker
- merges standardized characteristics, factor loadings, and the target into one panel
- performs cross-sectional normalization by rank when requested
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add the repository root so tests can import local modules without installation.
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import src.final_dataset as final_dataset


def _make_multiindex_frame(values, dates, tickers, col_name):
    """Helper to build a simple (ticker, date)-indexed DataFrame with one column."""
    return pd.DataFrame(
        {col_name: values},
        index=pd.MultiIndex.from_product([tickers, dates], names=["ticker", "date"]),
    )


def test_next_return_shift_and_merge():
    # Two tickers over three month-ends.
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    tickers = ["AAA", "BBB"]

    # Realized returns in panel form.
    returns = _make_multiindex_frame(
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        dates,
        tickers,
        "RET",
    )

    # A single characteristic (e.g., size) to ensure merging works.
    characteristics = _make_multiindex_frame(
        [1, 2, 3, 4, 5, 6],
        dates,
        tickers,
        "size",
    )

    # Factor loadings aligned to the same index.
    factor_loadings = pd.DataFrame(
        {"beta_MKT": 0.5, "beta_SMB": 0.1, "beta_HML": -0.2},
        index=returns.index,
    )

    dataset = final_dataset.assemble_modeling_dataset(returns, characteristics, factor_loadings)

    # The last month cannot have a next-period return, so it should be dropped.
    assert dataset.index.get_level_values("date").max() == dates[-2]

    # next_return should be the within-ticker shift(-1) of RET.
    expected_next = pd.Series(
        [0.02, 0.03, 0.05, 0.06],
        index=pd.MultiIndex.from_product([tickers, dates[:-1]], names=["ticker", "date"]),
        name="next_return",
    )

    pd.testing.assert_series_equal(dataset["next_return"], expected_next)

    # The final dataset should contain: standardized characteristics + betas + target.
    assert set(dataset.columns) == {"size", "beta_MKT", "beta_SMB", "beta_HML", "next_return"}


def test_rank_normalization_cross_section():
    # Three tickers over two months; only the first month survives after next_return drop.
    dates = pd.date_range("2021-01-31", periods=2, freq="ME")
    tickers = ["AAA", "BBB", "CCC"]

    returns = _make_multiindex_frame(
        [0.1, 0.2, 0.3, 0.0, -0.1, 0.4],
        dates,
        tickers,
        "RET",
    )

    # Two characteristics with a known ordering within each cross-section.
    characteristics = pd.DataFrame(
        {
            "value": [1, 3, 2, 5, 4, 6],
            "momentum": [5, 4, 3, 2, 1, 0],
        },
        index=pd.MultiIndex.from_product([tickers, dates], names=["ticker", "date"]),
    )

    # Minimal factor loading included to validate merging doesn't break.
    factor_loadings = pd.DataFrame({"beta_MKT": 1.0}, index=characteristics.index)

    dataset = final_dataset.assemble_modeling_dataset(
        returns,
        characteristics,
        factor_loadings,
        standardize="rank",
    )

    # Only the first month remains because the second month has no next_return.
    first_month = dataset.xs(dates[0], level="date")

    # Rank standardization uses percentile ranks in [0, 1] with method="average".
    # value ordering at first date: AAA=1 (lowest), CCC=2 (middle), BBB=3 (highest)
    assert first_month.loc["AAA", "value"] == pytest.approx(1 / 3)
    assert first_month.loc["BBB", "value"] == pytest.approx(1.0)  # highest
    assert first_month.loc["CCC", "value"] == pytest.approx(2 / 3)

    # momentum ordering at first date: CCC=3 (lowest), BBB=4 (middle), AAA=5 (highest)
    assert first_month.loc["AAA", "momentum"] == pytest.approx(1.0)
    assert first_month.loc["BBB", "momentum"] == pytest.approx(2 / 3)
    assert first_month.loc["CCC", "momentum"] == pytest.approx(1 / 3)