from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


ValueMethod = Literal["moving_average", "long_term_reversal"]
ProfitabilityMethod = Literal["rolling_sharpe", "return_over_volatility"]


@dataclass
class FeatureConfig:
    """Configuration for characteristic computation.

    Attributes
    ----------
    price_col : str
        Column containing close or adjusted-close prices.
    volume_col : str
        Column containing daily share volume.
    return_col : str | None
        Optional column containing daily returns. When omitted, returns are
        inferred from ``price_col``.
    value_method : {"moving_average", "long_term_reversal"}
        Choice of value proxy. ``"moving_average"`` computes the negative
        price-to-12-month moving-average ratio, while ``"long_term_reversal"``
        uses negative cumulative returns from months ``t-60`` to ``t-13``.
    profitability_method : {"rolling_sharpe", "return_over_volatility"}
        Choice of profitability proxy. ``"rolling_sharpe"`` uses the 12-month
        rolling Sharpe ratio of monthly returns; ``"return_over_volatility"``
        divides 12-month cumulative returns by 12-month volatility.
    volatility_window_days : int
        Length of the rolling window (in trading days) for realized volatility.
    volatility_min_days : int
        Minimum daily observations required before emitting a volatility
        estimate.
    momentum_min_months : int
        Minimum history required to compute momentum (default 6 months).
    value_min_months : int
        Minimum history required for value proxies (default 6 months).
    reversal_min_months : int
        Minimum history required for long-term reversal (default 24 months).
    profitability_min_months : int
        Minimum history required for profitability metrics (default 6 months).
    investment_min_months : int
        Minimum history required before computing investment.
    """

    price_col: str = "adj_close"
    volume_col: str = "volume"
    return_col: Optional[str] = None
    value_method: ValueMethod = "moving_average"
    profitability_method: ProfitabilityMethod = "rolling_sharpe"
    volatility_window_days: int = 252
    volatility_min_days: int = 63
    momentum_min_months: int = 6
    value_min_months: int = 6
    reversal_min_months: int = 24
    profitability_min_months: int = 6
    investment_min_months: int = 6

    def __post_init__(self) -> None:
        if self.value_method not in {"moving_average", "long_term_reversal"}:
            raise ValueError("value_method must be 'moving_average' or 'long_term_reversal'.")
        if self.profitability_method not in {"rolling_sharpe", "return_over_volatility"}:
            raise ValueError(
                "profitability_method must be 'rolling_sharpe' or 'return_over_volatility'."
            )
        if self.volatility_window_days <= 1 or self.volatility_min_days <= 1:
            raise ValueError("Volatility windows must exceed one day.")


def _validate_multiindex(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.MultiIndex) or df.index.names[:2] != ["ticker", "date"]:
        raise ValueError("Input data must use a MultiIndex with levels ('ticker', 'date').")


def _cumulative_return(window: pd.Series) -> float:
    return float(np.prod(1 + window) - 1)


def _compute_dollar_volume(price: pd.Series, volume: pd.Series) -> pd.Series:
    dollar_vol = price * volume
    dollar_vol = dollar_vol.where(dollar_vol > 0)
    return np.log(dollar_vol)


def _compute_value(price: pd.Series, monthly_return: pd.Series, config: FeatureConfig) -> pd.Series:
    if config.value_method == "moving_average":
        moving_avg = price.rolling(12, min_periods=config.value_min_months).mean()
        divisor = moving_avg.replace(0, np.nan)
        return -(price / divisor)

    # Long-term reversal: negative cumulative returns from t-60 to t-13
    reversal = (
        (1 + monthly_return.shift(13))
        .rolling(48, min_periods=config.reversal_min_months)
        .apply(_cumulative_return, raw=False)
    )
    return -reversal


def _compute_momentum(monthly_return: pd.Series, config: FeatureConfig) -> pd.Series:
    # Cumulative return over months t-12:t-2 (skip t-1)
    return (
        (1 + monthly_return.shift(2))
        .rolling(11, min_periods=config.momentum_min_months)
        .apply(_cumulative_return, raw=False)
    )


def _compute_volatility(daily_return: pd.Series, config: FeatureConfig) -> pd.Series:
    realized_vol = (
        daily_return
        .rolling(config.volatility_window_days, min_periods=config.volatility_min_days)
        .std(ddof=0)
        * np.sqrt(252)
    )
    return realized_vol


def _compute_investment(monthly_volume: pd.Series, config: FeatureConfig) -> pd.Series:
    avg_volume = monthly_volume.rolling(12, min_periods=config.investment_min_months).mean()
    log_avg_volume = np.log(avg_volume.replace(0, np.nan))
    return log_avg_volume.diff()


def _compute_profitability(
    monthly_return: pd.Series,
    monthly_volatility: pd.Series,
    config: FeatureConfig,
) -> pd.Series:
    lagged = monthly_return.shift(1)

    if config.profitability_method == "rolling_sharpe":
        rolling_mean = lagged.rolling(12, min_periods=config.profitability_min_months).mean()
        rolling_std = lagged.rolling(12, min_periods=config.profitability_min_months).std(ddof=0)
        rolling_std = rolling_std.replace(0, np.nan)
        return rolling_mean / rolling_std

    cumulative = (
        (1 + lagged)
        .rolling(12, min_periods=config.profitability_min_months)
        .apply(_cumulative_return, raw=False)
    )
    divisor = monthly_volatility.replace(0, np.nan)
    return cumulative / divisor

def compute_features(raw: pd.DataFrame, *, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Compute cross-sectional characteristics from price and volume data.

    Parameters
    ----------
    raw : pd.DataFrame
        Multi-indexed by ``('ticker', 'date')`` with columns for price,
        volume, and optional daily returns.
    config : FeatureConfig, optional
        Controls column names and method choices for value and profitability.

    Returns
    -------
    pd.DataFrame
        Multi-indexed by ``('ticker', 'date')`` containing columns ``dollar_volume``,
        ``value``, ``momentum``, ``volatility``, ``investment``, and
        ``profitability``. Rows where all features are missing are dropped.
    """

    _validate_multiindex(raw)
    cfg = config or FeatureConfig()

    required_cols = {cfg.price_col, cfg.volume_col}
    missing = required_cols - set(raw.columns)
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(sorted(missing))}.")

    results: list[pd.DataFrame] = []

    for ticker, df_ticker in raw.groupby(level="ticker", sort=True):
        df_sorted = df_ticker.droplevel("ticker").sort_index()
        df_sorted.index = pd.to_datetime(df_sorted.index)

        price = df_sorted[cfg.price_col]
        volume = df_sorted[cfg.volume_col]

        if cfg.return_col and cfg.return_col in df_sorted.columns:
            daily_ret = df_sorted[cfg.return_col].copy()
        else:
            daily_ret = price.pct_change(fill_method=None)

        realized_daily_vol = _compute_volatility(daily_ret, cfg)

        # Month-end series (use one convention everywhere)
        monthly_price = price.resample("ME").last()
        monthly_volume = volume.resample("ME").mean()
        monthly_return = monthly_price.pct_change(fill_method=None)
        monthly_volatility = realized_daily_vol.resample("ME").last()
        monthly_volatility = monthly_volatility.shift(1)

        # Characteristics
        dollar_volume = _compute_dollar_volume(monthly_price, monthly_volume)
        dollar_volume = dollar_volume.shift(1)
        value = _compute_value(monthly_price, monthly_return, cfg)
        momentum = _compute_momentum(monthly_return, cfg)
        investment = _compute_investment(monthly_volume, cfg)

        # Profitability (both methods now use consistent inputs)
        profitability = _compute_profitability(monthly_return, monthly_volatility, cfg)

        features = pd.DataFrame(
            {
                "dollar_volume": dollar_volume,
                "value": value,
                "momentum": momentum,
                "volatility": monthly_volatility,
                "investment": investment,
            }
        )

        features["profitability"] = profitability
        features["ticker"] = ticker
        features = features.set_index("ticker", append=True).reorder_levels(["ticker", "date"])
        results.append(features)

    if not results:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])
        return pd.DataFrame(columns=[
            "dollar_volume",
            "value",
            "momentum",
            "volatility",
            "investment",
            "profitability",
        ], index=empty_index)


    features_db = pd.concat(results).sort_index()
    features_db.index = features_db.index.set_names(["ticker", "date"])

    return features_db.dropna(how="all")


__all__ = ["FeatureConfig", "compute_features"]