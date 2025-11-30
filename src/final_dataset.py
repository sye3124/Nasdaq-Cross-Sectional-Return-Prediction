"""Construction helpers for the final modeling dataset."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


StandardizeMethod = Literal["zscore", "rank"]


def _validate_multiindex(df: pd.DataFrame, name: str) -> None:
    if not isinstance(df.index, pd.MultiIndex) or df.index.names[:2] != ["ticker", "date"]:
        raise ValueError(f"{name} must use a MultiIndex with levels ('ticker', 'date').")


def _standardize_characteristics(
    characteristics: pd.DataFrame, method: StandardizeMethod
) -> pd.DataFrame:

    def zscore(group):
        mean = group.mean()
        std = group.std(ddof=0)

        if hasattr(std, "replace"):     # Series
            std_safe = std.replace(0, np.nan)
        else:                           # scalar
            std_safe = np.nan if std == 0 else std

        scaled = (group - mean) / std_safe
        return scaled.fillna(0.0)

    def rank(group):
        # group is Series (1-column DataFrame case)
        if isinstance(group, pd.Series):
            return group.rank(pct=True, method="average")
        # group is DataFrame (multi-column characteristics)
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
    """Assemble the final panel used for model training.

    Parameters
    ----------
    returns : pd.DataFrame
        Multi-indexed by ``('ticker', 'date')`` with a column for realized returns.
    characteristics : pd.DataFrame
        Cross-sectional signals indexed by ``('ticker', 'date')``.
    factor_loadings : pd.DataFrame
        Rolling betas indexed by ``('ticker', 'date')`` as produced by
        :func:`src.factor_exposures.compute_factor_exposures`.
    return_col : str, optional
        Name of the realized return column in ``returns``. Defaults to ``"RET"``.
    next_return_col : str, optional
        Name of the target column containing the next-period return. Defaults to
        ``"next_return"``.
    standardize : {"zscore", "rank"}, optional
        Method for cross-sectional normalization within each date. ``"zscore"``
        standardizes to mean 0 and standard deviation 1 (using population std),
        while ``"rank"`` converts to percentile ranks in ``[0, 1]``.

    Returns
    -------
    pd.DataFrame
        Multi-indexed by ``('ticker', 'date')`` with the normalized
        characteristics, factor loadings, and next-period returns.
    """

    if standardize not in {"zscore", "rank"}:
        raise ValueError("standardize must be 'zscore' or 'rank'.")

    for df, name in (
        (returns, "returns"),
        (characteristics, "characteristics"),
        (factor_loadings, "factor_loadings"),
    ):
        _validate_multiindex(df, name)

    returns = returns.copy()
    returns.index = pd.MultiIndex.from_arrays(
        [
            returns.index.get_level_values("ticker"),
            pd.to_datetime(returns.index.get_level_values("date")),
        ],
        names=["ticker", "date"],
    )

    next_return = (
        returns[return_col]
        .groupby(level="ticker")
        .shift(-1)
        .to_frame(next_return_col)
    )

    standardized_chars = _standardize_characteristics(characteristics.copy(), method=standardize)

    dataset = pd.concat([standardized_chars, factor_loadings, next_return], axis=1)
    dataset = dataset.dropna(subset=[next_return_col])

    return dataset


__all__ = ["assemble_modeling_dataset",
]