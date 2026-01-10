"""Tests for final_dataset.py.

These tests verify that `assemble_modeling_dataset()` correctly:
- creates the next-period return target by shifting returns within each ticker
- merges characteristics, factor loadings, and the target into one panel
- applies cross-sectional normalization (z-score or rank) without leaking information
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure the test runner can import project modules without installing the package.
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import src.final_dataset as final_dataset


def _make_multiindex_frame(values, dates, tickers, col_name):
    """Helper to build a simple (ticker, date)-indexed DataFrame with one column.

    We use this helper to keep tests focused on *alignment logic* (indexing and
    shifting) rather than on repetitive DataFrame construction boilerplate.
    """
    return pd.DataFrame(
        {col_name: values},
        index=pd.MultiIndex.from_product([tickers, dates], names=["ticker", "date"]),
    )


def test_next_return_shift_and_merge():
    # Use a tiny panel where the correct target and merged shape can be checked by inspection.
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    tickers = ["AAA", "BBB"]

    # Realized returns are the only input needed to define the next-period target.
    returns = _make_multiindex_frame(
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        dates,
        tickers,
        "RET",
    )

    # Characteristics are merged as features; values are arbitrary as long as alignment is clear.
    characteristics = _make_multiindex_frame(
        [1, 2, 3, 4, 5, 6],
        dates,
        tickers,
        "size",
    )

    # Factor loadings are merged as additional features on the same (ticker, date) grid.
    factor_loadings = pd.DataFrame(
        {"beta_MKT": 0.5, "beta_SMB": 0.1, "beta_HML": -0.2},
        index=returns.index,
    )

    dataset = final_dataset.assemble_modeling_dataset(returns, characteristics, factor_loadings)

    # The final month has no "next" return to predict, so it must be removed.
    assert dataset.index.get_level_values("date").max() == dates[-2]

    # The supervised-learning target should be the within-ticker forward shift of realized returns.
    expected_next = pd.Series(
        [0.02, 0.03, 0.05, 0.06],
        index=pd.MultiIndex.from_product([tickers, dates[:-1]], names=["ticker", "date"]),
        name="next_return",
    )

    pd.testing.assert_series_equal(dataset["next_return"], expected_next)

    # The output schema should include standardized numeric characteristics, factor betas, and the target.
    assert set(dataset.columns) == {"size", "beta_MKT", "beta_SMB", "beta_HML", "next_return"}


def test_rank_normalization_cross_section():
    # Construct a panel where the cross-sectional ordering is known so percentile ranks are predictable.
    dates = pd.date_range("2021-01-31", periods=2, freq="ME")
    tickers = ["AAA", "BBB", "CCC"]

    # Two months of returns: the second month will be dropped after the next_return shift.
    returns = _make_multiindex_frame(
        [0.1, 0.0, 0.2, -0.1, 0.3, 0.4],
        dates,
        tickers,
        "RET",
    )

    # Provide multiple characteristics to verify rank normalization is applied per-column within each date.
    characteristics = pd.DataFrame(
    {
        # date-major intended: date1 [1,3,2], date2 [5,4,6]
        # ticker-major required by your index: AAA[1,5], BBB[3,4], CCC[2,6]
        "value": [1, 5, 3, 4, 2, 6],

        # date-major intended: date1 [5,4,3], date2 [2,1,0]
        # ticker-major: AAA[5,2], BBB[4,1], CCC[3,0]
        "momentum": [5, 2, 4, 1, 3, 0],
    },
    index=pd.MultiIndex.from_product([tickers, dates], names=["ticker", "date"]),
    )


    # Include at least one factor loading so merging logic is exercised under rank standardization.
    factor_loadings = pd.DataFrame({"beta_MKT": 1.0}, index=characteristics.index)

    dataset = final_dataset.assemble_modeling_dataset(
        returns,
        characteristics,
        factor_loadings,
        standardize="rank",
    )

    # Only the first month survives because the last month cannot have a next_return target.
    first_month = dataset.xs(dates[0], level="date")

    # Percentile ranks are computed within the cross-section using method="average".
    # value ordering at first date: AAA=1 (lowest), CCC=2 (middle), BBB=3 (highest)
    assert first_month.loc["AAA", "value"] == pytest.approx(1 / 3)
    assert first_month.loc["CCC", "value"] == pytest.approx(2 / 3)
    assert first_month.loc["BBB", "value"] == pytest.approx(1.0)

    # momentum ordering at first date: CCC=3 (lowest), BBB=4 (middle), AAA=5 (highest)
    assert first_month.loc["CCC", "momentum"] == pytest.approx(1 / 3)
    assert first_month.loc["BBB", "momentum"] == pytest.approx(2 / 3)
    assert first_month.loc["AAA", "momentum"] == pytest.approx(1.0)