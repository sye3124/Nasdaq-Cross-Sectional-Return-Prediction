from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


ValueMethod = Literal["moving_average", "long_term_reversal"]
ProfitabilityMethod = Literal["rolling_sharpe", "return_over_volatility"]


@dataclass
class FeatureConfig:
    """Parameters controlling how each characteristic is computed.

    This module builds a set of monthly, cross-sectional features from daily
    price/volume inputs. Most features are defined using trailing windows and/or
    lagged inputs so they are usable as predictors without peeking into the
    current month.

    Notes on the feature definitions
    --------------------------------
    - dollar_volume: log(price * volume) at month end, then lagged by one month
    - value:
        * "moving_average": -(price / 12m moving average of price)
        * "long_term_reversal": - cumulative return from t-60 to t-13
    - momentum: cumulative return from t-12 to t-2 (skips the most recent month)
    - volatility: annualized realized volatility from daily returns (month-end),
      then lagged by one month
    - investment: change in log(12m average monthly volume)
    - profitability:
        * "rolling_sharpe": 12m mean / 12m std of lagged monthly returns
        * "return_over_volatility": 12m cumulative return / lagged 12m volatility
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
        # Keep config mistakes loud and early.
        if self.value_method not in {"moving_average", "long_term_reversal"}:
            raise ValueError("value_method must be 'moving_average' or 'long_term_reversal'.")
        if self.profitability_method not in {"rolling_sharpe", "return_over_volatility"}:
            raise ValueError(
                "profitability_method must be 'rolling_sharpe' or 'return_over_volatility'."
            )
        if self.volatility_window_days <= 1 or self.volatility_min_days <= 1:
            raise ValueError("Volatility windows must exceed one day.")


def _validate_multiindex(df: pd.DataFrame) -> None:
    """Ensure the input panel is indexed by ('ticker', 'date')."""
    if not isinstance(df.index, pd.MultiIndex) or df.index.names[:2] != ["ticker", "date"]:
        raise ValueError("Input data must use a MultiIndex with levels ('ticker', 'date').")


def _cumulative_return(window: pd.Series) -> float:
    """Compound returns over the window into a single cumulative return."""
    return float(np.prod(1 + window) - 1)


def _compute_dollar_volume(price: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute log dollar volume, treating non-positive values as missing."""
    dollar_vol = price * volume
    dollar_vol = dollar_vol.where(dollar_vol > 0)
    return np.log(dollar_vol)


def _compute_value(price: pd.Series, monthly_return: pd.Series, config: FeatureConfig) -> pd.Series:
    """Compute the chosen value proxy on monthly data."""
    if config.value_method == "moving_average":
        # Cheapness proxy: price relative to its trailing 12-month average.
        moving_avg = price.rolling(12, min_periods=config.value_min_months).mean()
        divisor = moving_avg.replace(0, np.nan)
        return -(price / divisor)

    # Long-term reversal: negative cumulative return from months t-60 to t-13.
    reversal = (
        (1 + monthly_return.shift(13))
        .rolling(48, min_periods=config.reversal_min_months)
        .apply(_cumulative_return, raw=False)
    )
    return -reversal


def _compute_momentum(monthly_return: pd.Series, config: FeatureConfig) -> pd.Series:
    """Compute momentum as cumulative return from t-12 to t-2 (skip t-1)."""
    return (
        (1 + monthly_return.shift(2))
        .rolling(11, min_periods=config.momentum_min_months)
        .apply(_cumulative_return, raw=False)
    )


def _compute_volatility(daily_return: pd.Series, config: FeatureConfig) -> pd.Series:
    """Compute annualized realized volatility from daily returns."""
    realized_vol = (
        daily_return.rolling(config.volatility_window_days, min_periods=config.volatility_min_days)
        .std(ddof=0)
        * np.sqrt(252)
    )
    return realized_vol


def _compute_investment(monthly_volume: pd.Series, config: FeatureConfig) -> pd.Series:
    """Proxy investment with the change in log trailing-average trading activity."""
    avg_volume = monthly_volume.rolling(12, min_periods=config.investment_min_months).mean()
    log_avg_volume = np.log(avg_volume.replace(0, np.nan))
    return log_avg_volume.diff()


def _compute_profitability(
    monthly_return: pd.Series,
    monthly_volatility: pd.Series,
    config: FeatureConfig,
) -> pd.Series:
    """Compute the chosen profitability proxy using monthly inputs."""
    # Use lagged monthly returns so the feature is known at the start of month t.
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
    """Turn daily price/volume panels into monthly cross-sectional features.

    The input is a daily panel indexed by (ticker, date). Each ticker is
    resampled to month-end ("ME") and features are computed on those monthly
    series. Where appropriate, inputs are shifted so the feature at month t
    only uses information available through month t-1.

    Parameters
    ----------
    raw
        Daily panel indexed by ``('ticker', 'date')`` containing at least price
        and volume columns.
    config
        Optional configuration controlling column names, window lengths, and
        which value/profitability definitions to use.

    Returns
    -------
    pd.DataFrame
        Monthly panel indexed by ``('ticker', 'date')`` with columns:
        ``dollar_volume``, ``value``, ``momentum``, ``volatility``,
        ``investment``, and ``profitability``. Rows where every feature is
        missing are removed.
    """

    _validate_multiindex(raw)
    cfg = config or FeatureConfig()

    required_cols = {cfg.price_col, cfg.volume_col}
    missing = required_cols - set(raw.columns)
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(sorted(missing))}.")

    results: list[pd.DataFrame] = []

    # Compute features one ticker at a time to keep indexing and resampling clean.
    for ticker, df_ticker in raw.groupby(level="ticker", sort=True):
        df_sorted = df_ticker.droplevel("ticker").sort_index()
        df_sorted.index = pd.to_datetime(df_sorted.index)

        price = df_sorted[cfg.price_col]
        volume = df_sorted[cfg.volume_col]

        # Use provided returns if available; otherwise compute simple returns from price.
        if cfg.return_col and cfg.return_col in df_sorted.columns:
            daily_ret = df_sorted[cfg.return_col].copy()
        else:
            daily_ret = price.pct_change(fill_method=None)

        realized_daily_vol = _compute_volatility(daily_ret, cfg)

        # Convert daily inputs to month-end series (single convention across all features).
        monthly_price = price.resample("ME").last()
        monthly_volume = volume.resample("ME").mean()
        monthly_return = monthly_price.pct_change(fill_method=None)

        # Volatility is taken at month end and then lagged one month.
        monthly_volatility = realized_daily_vol.resample("ME").last().shift(1)

        # Core characteristics (most are already defined in lag-safe ways).
        dollar_volume = _compute_dollar_volume(monthly_price, monthly_volume).shift(1)
        value = _compute_value(monthly_price, monthly_return, cfg)
        momentum = _compute_momentum(monthly_return, cfg)
        investment = _compute_investment(monthly_volume, cfg)
        profitability = _compute_profitability(monthly_return, monthly_volatility, cfg)

        features = pd.DataFrame(
            {
                "dollar_volume": dollar_volume,
                "value": value,
                "momentum": momentum,
                "volatility": monthly_volatility,
                "investment": investment,
                "profitability": profitability,
            }
        )

        features["ticker"] = ticker
        features = features.set_index("ticker", append=True).reorder_levels(["ticker", "date"])
        results.append(features)

    # Preserve schema even when there are no tickers / no output.
    if not results:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])
        return pd.DataFrame(
            columns=[
                "dollar_volume",
                "value",
                "momentum",
                "volatility",
                "investment",
                "profitability",
            ],
            index=empty_index,
        )

    features_db = pd.concat(results).sort_index()
    features_db.index = features_db.index.set_names(["ticker", "date"])

    return features_db.dropna(how="all")


__all__ = [
    "FeatureConfig", 
    "compute_features"
]