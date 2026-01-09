"""Rolling Fama–French (FF3) factor exposure estimation.

This module estimates time-varying factor exposures by running trailing-window
OLS regressions of each stock’s excess return on the Fama–French three factors:

    excess_return ~ alpha + beta_MKT * MKT + beta_SMB * SMB + beta_HML * HML

The outputs are returned as:

- alpha_ff3: the regression intercept (abnormal return relative to FF3)
- beta_MKT, beta_SMB, beta_HML: factor loadings

The resulting exposures are shifted by one period so that the value stored at
date *t* only uses information available up to *t-1*. This makes the exposures
safe to use as features in cross-sectional models at time *t*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class ExposureConfig:
    """Settings controlling how rolling exposures are computed.

    Parameters
    ----------
    min_months
        Minimum number of observations required to fit a regression.
    max_months
        Maximum trailing window size used when fitting each regression.
    return_col
        Column name in the stock return panel containing the raw return.
    rf_col
        Column name in the factor table containing the risk-free rate.
    factor_cols
        Names of the factor columns to include (in order) in the regression.
    """

    # Window lengths in months.
    min_months: int = 36
    max_months: int = 60
    return_col: str = "RET"
    rf_col: str = "RF"
    factor_cols: tuple[str, str, str] = ("MKT", "SMB", "HML")

    # pylint: disable=missing-function-docstring
    def __post_init__(self) -> None:
        # Basic sanity checks to avoid silent misconfiguration.
        if self.min_months <= 0 or self.max_months <= 0:
            raise ValueError("Window lengths must be positive.")
        if self.min_months > self.max_months:
            raise ValueError("min_months cannot exceed max_months.")


def _fit_rolling_window(window: pd.DataFrame, config: ExposureConfig) -> Optional[pd.Series]:
    """Fit an FF3 regression on one trailing window.

    The input window is expected to contain stock returns, the risk-free rate,
    and the three FF3 factor series. Missing rows are dropped before fitting.

    Parameters
    ----------
    window
        Trailing slice of the merged stock/factor data.
    config
        Column names and minimum/maximum window constraints.

    Returns
    -------
    Optional[pd.Series]
        A Series with keys ``alpha_ff3`` and ``beta_<factor>`` for each factor.
        Returns ``None`` if there is not enough data (or the regression fails).
    """

    # Keep only rows where all regression inputs are present.
    required_cols = [config.return_col, config.rf_col, *config.factor_cols]
    trimmed = window.dropna(subset=required_cols)

    # If we don't have enough observations, skip this window.
    if len(trimmed) < config.min_months:
        return None

    # Response: stock excess return over the risk-free rate.
    excess_returns = trimmed[config.return_col] - trimmed[config.rf_col]

    # Predictors: the factor returns, plus an explicit intercept.
    factors = trimmed[list(config.factor_cols)]
    X = np.column_stack([np.ones(len(factors)), factors.to_numpy()])

    # Solve via least squares. This keeps dependencies minimal and behaves well
    # even if the design matrix is close to singular.
    try:
        coefs, *_ = np.linalg.lstsq(X, excess_returns.to_numpy(), rcond=None)
    except Exception:
        return None

    # Package coefficients in a consistent, model-friendly naming scheme.
    out = {"alpha_ff3": coefs[0]}
    out.update({f"beta_{name}": coefs[i + 1] for i, name in enumerate(config.factor_cols)})
    return pd.Series(out)


def compute_factor_exposures(
    stock_returns: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    config: ExposureConfig = ExposureConfig(),
) -> pd.DataFrame:
    """Compute rolling FF3 exposures for every ticker in a return panel.

    Expected inputs
    ---------------
    - ``stock_returns``: MultiIndex ``('ticker', 'date')`` with a return column
      named by ``config.return_col``.
    - ``factors``: indexed by date with factor columns in ``config.factor_cols``
      and a risk-free rate column ``config.rf_col``.

    The output is lagged by one period within each ticker so that exposures at
    date *t* are estimated only from information up to *t-1*.

    Parameters
    ----------
    stock_returns
        Panel of stock returns indexed by ``('ticker', 'date')``.
    factors
        Time series of factor returns and the risk-free rate indexed by date.
    config
        Controls column names and trailing-window sizes.

    Returns
    -------
    pd.DataFrame
        MultiIndex ``('ticker', 'date')`` with columns ``alpha_ff3``,
        ``beta_MKT``, ``beta_SMB``, and ``beta_HML``.
    """

    # Enforce the expected panel shape: (ticker, date) MultiIndex.
    if not isinstance(stock_returns.index, pd.MultiIndex) or stock_returns.index.names[:2] != [
        "ticker",
        "date",
    ]:
        raise ValueError("stock_returns must use a MultiIndex with levels ('ticker', 'date').")

    # Standardize factor index naming to make joins predictable.
    factors = factors.copy()
    if factors.index.name != "date":
        factors.index = pd.Index(factors.index, name="date")

    results: list[pd.Series] = []

    # Process each ticker independently (rolling windows are not shared).
    for ticker, df_stock in stock_returns.groupby(level="ticker", sort=True):
        stock = df_stock.droplevel("ticker")
        stock.index = pd.to_datetime(stock.index)

        # Align stock returns with factor data on the date index.
        merged = stock.join(factors, how="left").sort_index()

        # Walk forward in time and fit a trailing regression for each endpoint.
        dates: Iterable[pd.Timestamp] = merged.index
        for idx in range(config.min_months - 1, len(merged)):
            window_end = idx + 1  # iloc end is exclusive
            window_start = max(0, window_end - config.max_months)
            window = merged.iloc[window_start:window_end]

            estimates = _fit_rolling_window(window, config)
            if estimates is None:
                continue

            row = estimates.to_dict()
            row["ticker"] = ticker
            row["date"] = dates[idx]
            results.append(row)

    # If nothing was estimable, return an empty frame with the right schema.
    if not results:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])
        return pd.DataFrame(
            columns=["alpha_ff3", "beta_MKT", "beta_SMB", "beta_HML"],
            index=empty_index,
        )

    # Assemble results and shift within each ticker so exposures at t are known at t.
    df_exposures = pd.DataFrame(results).set_index(["ticker", "date"]).sort_index()
    df_exposures = df_exposures.groupby(level="ticker").shift(1).dropna(how="all")

    return df_exposures


__all__ = [
    "ExposureConfig",
    "compute_factor_exposures",
]