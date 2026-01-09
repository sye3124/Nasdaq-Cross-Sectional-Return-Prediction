"""Utilities for building the final model-ready panel.

This module takes three aligned inputs—realized returns, cross-sectional
characteristics, and rolling factor loadings—and combines them into a single
monthly dataset suitable for supervised learning.

Key steps:
- normalize all (ticker, date) indices to month-end timestamps
- create the prediction target as the next-month return
- standardize characteristics cross-sectionally within each month (z-score or rank)
- join everything together and drop rows with missing targets/features
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


StandardizeMethod = Literal["zscore", "rank"]


def _validate_multiindex(df: pd.DataFrame, name: str) -> None:
    """Ensure a DataFrame is indexed by ('ticker', 'date').

    Parameters
    ----------
    df
        Input frame to validate.
    name
        Human-readable name used in the error message.
    """
    if not isinstance(df.index, pd.MultiIndex) or df.index.names[:2] != ["ticker", "date"]:
        raise ValueError(f"{name} must use a MultiIndex with levels ('ticker', 'date').")


def _normalize_month_end_index(df: pd.DataFrame) -> pd.DataFrame:
    """Force the date level of a (ticker, date) index onto month-end timestamps.

    This is a small but important hygiene step: upstream pipelines sometimes
    carry month labels as arbitrary days within the month, period values, or
    timestamps that aren't exactly the month end. For modeling we want a single,
    consistent convention so joins behave as expected.

    Returns a sorted copy of the input with a rebuilt MultiIndex.
    """
    idx = df.index
    tick = idx.get_level_values("ticker")
    dt = (
        pd.DatetimeIndex(idx.get_level_values("date"))
        .to_period("M")
        .to_timestamp("M")  # month-end
    )

    out = df.copy()
    out.index = pd.MultiIndex.from_arrays([tick, dt], names=["ticker", "date"])
    return out.sort_index()


def _standardize_characteristics(
    characteristics: pd.DataFrame, method: StandardizeMethod
) -> pd.DataFrame:
    """Standardize numeric characteristics within each date.

    Parameters
    ----------
    characteristics
        Numeric columns indexed by (ticker, date).
    method
        - "zscore": (x - mean) / std within each date (population std, ddof=0)
          with zero-filled results when std is zero / undefined.
        - "rank": percentile ranks within each date (values in [0, 1]).

    Returns
    -------
    pd.DataFrame
        Same shape as the input, standardized per date.
    """

    def zscore(group):
        mean = group.mean()
        std = group.std(ddof=0)

        # Protect against divide-by-zero when a column is constant for a date.
        if hasattr(std, "replace"):  # Series
            std_safe = std.replace(0, np.nan)
        else:  # scalar
            std_safe = np.nan if std == 0 else std

        scaled = (group - mean) / std_safe
        return scaled.fillna(0.0)

    def rank(group):
        # For Series, rank directly; for DataFrame, rank each column.
        if isinstance(group, pd.Series):
            return group.rank(pct=True, method="average")
        return group.apply(lambda col: col.rank(pct=True, method="average"))

    transformer = zscore if method == "zscore" else rank
    return characteristics.groupby(level="date").transform(transformer)


# pylint: disable=too-many-arguments
def assemble_modeling_dataset(
    returns: pd.DataFrame,
    characteristics: pd.DataFrame,
    factor_loadings: pd.DataFrame,
    *,
    return_col: str = "RET",
    next_return_col: str = "next_return",
    standardize: StandardizeMethod = "zscore",
) -> pd.DataFrame:
    """Merge inputs into a single monthly panel for training or backtesting.

    The target is constructed as the *next* period's return for each ticker:
    ``next_return[t] = return[t+1]``. All features remain aligned at time t.

    Characteristics are standardized cross-sectionally within each month using
    either z-scores or percentile ranks. Factor loadings are included as-is.

    Parameters
    ----------
    returns
        Panel indexed by ``('ticker', 'date')`` containing realized returns.
    characteristics
        Panel of cross-sectional signals indexed by ``('ticker', 'date')``.
    factor_loadings
        Panel of rolling factor exposures indexed by ``('ticker', 'date')``.
    return_col
        Column name in ``returns`` holding the realized return.
    next_return_col
        Name of the output target column for next-period returns.
    standardize
        Cross-sectional normalization method: ``"zscore"`` or ``"rank"``.

    Returns
    -------
    pd.DataFrame
        A model-ready panel indexed by ``('ticker', 'date')`` containing:
        standardized characteristics, factor loadings, and ``next_return_col``.
        Rows missing the target or any feature are removed.
    """

    # Guard against typos in the standardization mode.
    if standardize not in {"zscore", "rank"}:
        raise ValueError("standardize must be 'zscore' or 'rank'.")

    # Verify each input uses the expected panel index.
    for df, name in (
        (returns, "returns"),
        (characteristics, "characteristics"),
        (factor_loadings, "factor_loadings"),
    ):
        _validate_multiindex(df, name)

    # Align all time stamps to the same month-end convention.
    returns = _normalize_month_end_index(returns)
    characteristics = _normalize_month_end_index(characteristics)
    factor_loadings = _normalize_month_end_index(factor_loadings)

    # Target: next-period return per ticker.
    next_return = (
        returns[return_col].groupby(level="ticker").shift(-1).to_frame(next_return_col)
    )

    # Only standardize numeric characteristics; keep non-numeric columns out.
    chars = characteristics.select_dtypes(include=[np.number]).copy()
    standardized_chars = _standardize_characteristics(chars, method=standardize)

    # Join features + exposures + target into one panel.
    dataset = pd.concat([standardized_chars, factor_loadings, next_return], axis=1)

    # Keep only rows where we have a target and a full feature vector.
    dataset = dataset.dropna(subset=[next_return_col])
    dataset = dataset.dropna()

    return dataset


__all__ = [
    "assemble_modeling_dataset",
]