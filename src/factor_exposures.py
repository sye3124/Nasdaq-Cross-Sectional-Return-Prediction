"""Utilities for estimating rolling Fama-French factor exposures.

The helpers here compute 36–60 month rolling regressions of stock excess
returns on the Fama-French three factors (MKT, SMB, HML). The resulting
betas are lagged so they are available for use as cross-sectional features
when forming portfolios or predictive models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class ExposureConfig:
    """Configuration for rolling factor exposure estimation."""

    min_months: int = 36
    max_months: int = 60
    return_col: str = "RET"
    rf_col: str = "RF"
    factor_cols: tuple[str, str, str] = ("MKT", "SMB", "HML")

    def __post_init__(self) -> None:
        if self.min_months <= 0 or self.max_months <= 0:
            raise ValueError("Window lengths must be positive.")
        if self.min_months > self.max_months:
            raise ValueError("min_months cannot exceed max_months.")


def _fit_rolling_window(window: pd.DataFrame, config: ExposureConfig) -> Optional[pd.Series]:
    """Fit a single OLS regression for a trailing window.

    Parameters
    ----------
    window : pd.DataFrame
        Window of data with columns for stock return, risk-free rate, and
        factor columns.
    config : ExposureConfig
        Configuration describing column names and window constraints.

    Returns
    -------
    Optional[pd.Series]
        Series containing beta estimates for MKT, SMB, and HML. Returns
        ``None`` when the regression cannot be estimated.
    """

    # Drop rows where any input is missing so OLS receives a complete matrix.
    required_cols = [config.return_col, config.rf_col, *config.factor_cols]
    trimmed = window.dropna(subset=required_cols)
    if len(trimmed) < config.min_months:
        return None

    excess_returns = trimmed[config.return_col] - trimmed[config.rf_col]
    factors = trimmed[list(config.factor_cols)]

    # Build design matrix with an explicit intercept term. Using numpy's least
    # squares keeps the dependency light and is robust even when some factors are
    # constant (e.g., all zeros), which can otherwise trigger singular matrix
    # warnings in higher-level regression helpers.
    X = np.column_stack([np.ones(len(factors)), factors.to_numpy()])

    try:
        coefs, *_ = np.linalg.lstsq(X, excess_returns.to_numpy(), rcond=None)
    except Exception:
        return None

    return pd.Series({f"beta_{name}": coefs[i + 1] for i, name in enumerate(config.factor_cols)})


def compute_factor_exposures(
    stock_returns: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    config: ExposureConfig = ExposureConfig(),
) -> pd.DataFrame:
    """Estimate rolling factor loadings for each stock.

    The function expects ``stock_returns`` to be indexed by ``('ticker', 'date')``
    and to include a return column named in ``config.return_col``. ``factors``
    should be indexed by date with at least the columns specified in
    ``config.factor_cols`` and the risk-free rate column ``config.rf_col``.

    Parameters
    ----------
    stock_returns : pd.DataFrame
        Multi-indexed by ``ticker`` and ``date`` containing stock returns.
    factors : pd.DataFrame
        Indexed by date and containing the factor and risk-free series.
    config : ExposureConfig, optional
        Controls column names and rolling window lengths.

    Returns
    -------
    pd.DataFrame
        Multi-indexed by ``ticker`` and ``date`` with columns ``beta_MKT``,
        ``beta_SMB``, and ``beta_HML``. Betas are lagged by one period so they
        are known at time ``t``.
    """

    if not isinstance(stock_returns.index, pd.MultiIndex) or stock_returns.index.names[:2] != [
        "ticker",
        "date",
    ]:
        raise ValueError("stock_returns must use a MultiIndex with levels ('ticker', 'date').")

    factors = factors.copy()
    if factors.index.name != "date":
        factors.index = pd.Index(factors.index, name="date")

    results: list[pd.Series] = []

    for ticker, df_stock in stock_returns.groupby(level="ticker", sort=True):
        stock = df_stock.droplevel("ticker")
        stock.index = pd.to_datetime(stock.index)
        merged = stock.join(factors, how="left")
        merged = merged.sort_index()

        # Iterate through the sample, estimating betas using trailing windows
        # with a maximum length and a minimum threshold for stability.
        dates: Iterable[pd.Timestamp] = merged.index
        for idx in range(len(merged)):
            window_end = idx + 1  # slice end is exclusive
            window_start = max(0, window_end - config.max_months)
            window = merged.iloc[window_start:window_end]
            estimates = _fit_rolling_window(window, config)
            if estimates is None:
                continue

            result = estimates.copy()
            result.loc["ticker"] = ticker
            result.loc["date"] = dates[idx]
            results.append(result)

    if not results:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])
        return pd.DataFrame(
            columns=["beta_MKT", "beta_SMB", "beta_HML"], index=empty_index
        )

    df_exposures = pd.DataFrame(results).set_index(["ticker", "date"]).sort_index()

    # Lag exposures so they are available at the start of each period.
    df_exposures = df_exposures.groupby(level="ticker").shift(1)

    return df_exposures


__all__ = [
    "ExposureConfig",
    "compute_factor_exposures",
]