import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import final_dataset as final_dataset

def _make_multiindex_frame(values, dates, tickers, col_name):
    return pd.DataFrame(
        {col_name: values},
        index=pd.MultiIndex.from_product([tickers, dates], names=["ticker", "date"]),
    )


def test_next_return_shift_and_merge():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    tickers = ["AAA", "BBB"]

    returns = _make_multiindex_frame(
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06], dates, tickers, "RET"
    )

    characteristics = _make_multiindex_frame(
        [1, 2, 3, 4, 5, 6], dates, tickers, "size"
    )

    factor_loadings = pd.DataFrame(
        {"beta_MKT": 0.5, "beta_SMB": 0.1, "beta_HML": -0.2},
        index=returns.index,
    )

    dataset = final_dataset.assemble_modeling_dataset(returns, characteristics, factor_loadings)

    # Last date should be dropped because the next-period return is unavailable
    assert dataset.index.get_level_values("date").max() == dates[-2]

    expected_next = pd.Series(
        [0.02, 0.03, 0.05, 0.06],
        index=pd.MultiIndex.from_product([tickers, dates[:-1]], names=["ticker", "date"]),
        name="next_return",
    )
    pd.testing.assert_series_equal(dataset["next_return"], expected_next)
    assert set(dataset.columns) == {"size", "beta_MKT", "beta_SMB", "beta_HML", "next_return"}


def test_rank_normalization_cross_section():
    dates = pd.date_range("2021-01-31", periods=2, freq="ME")
    tickers = ["AAA", "BBB", "CCC"]

    returns = _make_multiindex_frame(
        [0.1, 0.2, 0.3, 0.0, -0.1, 0.4], dates, tickers, "RET"
    )
    characteristics = pd.DataFrame(
        {
            "value": [1, 3, 2, 5, 4, 6],
            "momentum": [5, 4, 3, 2, 1, 0],
        },
        index=pd.MultiIndex.from_product([tickers, dates], names=["ticker", "date"]),
    )
    factor_loadings = pd.DataFrame(
        {"beta_MKT": 1.0}, index=characteristics.index
    )

    dataset = final_dataset.assemble_modeling_dataset(
        returns, characteristics, factor_loadings, standardize="rank"
    )

    # Only the first month remains because we drop rows without next-period returns
    first_month = dataset.xs(dates[0], level="date")

    # value ranks (per cross-section at first date)
    assert first_month.loc["AAA", "value"] == pytest.approx(1 / 3)
    assert first_month.loc["BBB", "value"] == pytest.approx(2 / 3)
    assert first_month.loc["CCC", "value"] == pytest.approx(1.0)

    # momentum ranks (per cross-section at first date)
    assert first_month.loc["AAA", "momentum"] == pytest.approx(1.0)
    assert first_month.loc["BBB", "momentum"] == pytest.approx(2 / 3)
    assert first_month.loc["CCC", "momentum"] == pytest.approx(1 / 3)
