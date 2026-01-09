"""Tests for features.py.

These tests build a small synthetic daily panel and then verify that
`compute_features()` matches the feature definitions in the module.

The key idea:
- We generate daily prices/volumes that end exactly at known month-end values.
- The feature module resamples to month-end ("ME") internally.
- We manually reproduce the same resampling and feature formulas here and
  compare against the function output.
"""

import numpy as np
import pandas as pd

from src.features import FeatureConfig, compute_features


def _build_daily_panel(
    ticker: str,
    monthly_price: pd.Series,
    monthly_volume: pd.Series,
) -> pd.DataFrame:
    """Expand month-end price/volume series into a daily panel.

    For each month-end date in `monthly_price`:
    - create a daily date range from the start of the month to that month-end
    - linearly interpolate prices from the prior month-end to the current month-end
    - fill daily volume with a constant value equal to that month's volume level

    The resulting DataFrame is indexed by ('ticker', 'date'), matching the
    expected input format of `compute_features()`.
    """
    rows = []

    # First month "starts" at its own month-end value; subsequent months ramp from prev -> current.
    prev_price = monthly_price.iloc[0]

    for date, price_end in monthly_price.items():
        # Start of the month (00:00) through the month-end timestamp.
        start = date.to_period("M").start_time
        end = date

        days = pd.date_range(start, end, freq="D")

        # Smooth synthetic path so daily returns exist and volatility can be computed.
        prices = np.linspace(prev_price, price_end, len(days))

        # Simple constant daily volume within the month.
        volumes = np.full(len(days), monthly_volume.loc[date])

        rows.append(pd.DataFrame({"adj_close": prices, "volume": volumes}, index=days))
        prev_price = price_end

    daily = pd.concat(rows)

    # Build the expected MultiIndex: ('ticker', 'date')
    daily["ticker"] = ticker
    daily = daily.reset_index().rename(columns={"index": "date"})
    return daily.set_index(["ticker", "date"]).sort_index()


def test_compute_features_moving_average_and_sharpe():
    # Month-end dates: the feature code uses month-end resampling ("ME").
    month_ends = pd.date_range("2020-01-31", periods=15, freq="ME")

    # Synthetic month-end price and volume paths.
    monthly_price = pd.Series(np.linspace(10.0, 24.0, len(month_ends)), index=month_ends)
    monthly_volume = pd.Series(np.linspace(100.0, 240.0, len(month_ends)), index=month_ends)

    # Expand month-level inputs into daily data, then compute features.
    raw = _build_daily_panel("AAA", monthly_price, monthly_volume)
    features = compute_features(raw)

    # --- Manually reproduce the module's internal monthly inputs ---
    daily = raw.xs("AAA", level="ticker")
    price = daily["adj_close"]
    volume = daily["volume"]

    # Module convention:
    # - price: month-end last
    # - volume: within-month mean
    monthly_price_m = price.resample("ME").last()
    monthly_volume_m = volume.resample("ME").mean()
    monthly_return = monthly_price_m.pct_change()

    # --- Manually reproduce feature definitions (default config) ---

    # Dollar volume: log(price * volume), then lag by one month.
    expected_dollar_volume = np.log(
        (monthly_price_m * monthly_volume_m).where(monthly_price_m * monthly_volume_m > 0)
    ).shift(1)

    # Value (moving average method): negative price / 12M moving average, min 6 months.
    expected_value = -(
        monthly_price_m
        / monthly_price_m.rolling(12, min_periods=6).mean().replace(0, np.nan)
    )

    # Momentum: cumulative return over months t-12..t-2 (skip t-1), min 6 months.
    expected_momentum = (
        (1 + monthly_return.shift(2))
        .rolling(11, min_periods=6)
        .apply(lambda x: np.prod(1 + x) - 1, raw=False)
    )

    # Volatility:
    # - compute realized daily vol from daily returns
    # - resample to month-end (last)
    # - lag by one month
    daily_ret = price.pct_change(fill_method=None)
    realized_daily_vol = daily_ret.rolling(252, min_periods=63).std(ddof=0) * np.sqrt(252)
    expected_volatility = realized_daily_vol.resample("ME").last().shift(1)

    # Investment: diff of log(12M avg volume), min 6 months.
    expected_investment = np.log(
        monthly_volume_m.rolling(12, min_periods=6).mean().replace(0, np.nan)
    ).diff()

    # Profitability (rolling Sharpe):
    # - use lagged monthly returns (shift(1))
    # - rolling mean / rolling std, min 6 months
    lagged = monthly_return.shift(1)
    expected_profitability = (
        lagged.rolling(12, min_periods=6).mean()
        / lagged.rolling(12, min_periods=6).std(ddof=0).replace(0, np.nan)
    )

    # --- Compare output ---
    # `compute_features()` drops rows where all features are NaN.
    # Because several signals are lagged, the first month is fully missing.
    output = features.xs("AAA", level="ticker")

    expected_index = month_ends[1:].rename("date")
    pd.testing.assert_index_equal(output.index, expected_index)

    pd.testing.assert_series_equal(
        output["dollar_volume"],
        expected_dollar_volume.loc[expected_index],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        output["value"],
        expected_value.loc[expected_index],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        output["momentum"],
        expected_momentum.loc[expected_index],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        output["volatility"],
        expected_volatility.loc[expected_index],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        output["investment"],
        expected_investment.loc[expected_index],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        output["profitability"],
        expected_profitability.loc[expected_index],
        check_names=False,
    )