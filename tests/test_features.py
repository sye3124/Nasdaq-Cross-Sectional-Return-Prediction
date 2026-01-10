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

    # We anchor the very first month to its own month-end level so the generated
    # daily path has a well-defined starting price (no missing “previous month”).
    prev_price = monthly_price.iloc[0]

    for date, price_end in monthly_price.items():
        # The feature pipeline resamples from *daily* to *month-end*; building each
        # month as an explicit daily block makes the resampling behavior reproducible.
        start = date.to_period("M").start_time
        end = date
        days = pd.date_range(start, end, freq="D")

        # A smooth price path ensures daily returns exist everywhere, which matters
        # because realized volatility depends on a rolling window of daily returns.
        prices = np.linspace(prev_price, price_end, len(days))

        # Holding volume constant within each month keeps the test focused on the
        # transformation logic (rolling means / logs / diffs), not within-month noise.
        volumes = np.full(len(days), monthly_volume.loc[date])

        rows.append(pd.DataFrame({"adj_close": prices, "volume": volumes}, index=days))
        prev_price = price_end

    daily = pd.concat(rows)

    # The production code expects a (ticker, date) MultiIndex; constructing it here
    # ensures the test exercises the same grouping + resampling pathways.
    daily["ticker"] = ticker
    daily = daily.reset_index().rename(columns={"index": "date"})
    return daily.set_index(["ticker", "date"]).sort_index()


def test_compute_features_moving_average_and_sharpe():
    # Use month-end timestamps because the feature code standardizes everything
    # to a month-end convention when constructing monthly predictors.
    month_ends = pd.date_range("2020-01-31", periods=15, freq="ME")

    # Choose monotone price/volume paths so any deviations we observe come from the
    # feature transformations (lags/rolling windows), not from stochastic noise.
    monthly_price = pd.Series(np.linspace(10.0, 24.0, len(month_ends)), index=month_ends)
    monthly_volume = pd.Series(np.linspace(100.0, 240.0, len(month_ends)), index=month_ends)

    # The core contract: given a well-formed daily panel, compute_features should
    # reproduce the documented monthly signal definitions exactly.
    raw = _build_daily_panel("AAA", monthly_price, monthly_volume)
    features = compute_features(raw)

    # --- Manually reproduce the module's internal monthly inputs ---
    # We rebuild the same intermediate monthly series the module uses so that any
    # mismatch isolates a true definition/implementation discrepancy.
    daily = raw.xs("AAA", level="ticker")
    price = daily["adj_close"]
    volume = daily["volume"]

    # This mirrors the module's resampling choices: price uses the month-end close
    # (last), while volume uses the within-month average (mean).
    monthly_price_m = price.resample("ME").last()
    monthly_volume_m = volume.resample("ME").mean()
    monthly_return = monthly_price_m.pct_change()

    # --- Manually reproduce feature definitions (default config) ---

    # Dollar volume is treated as a liquidity proxy; it is lagged so the feature at
    # month t uses information known at the *start* of month t (avoids look-ahead).
    expected_dollar_volume = np.log(
        (monthly_price_m * monthly_volume_m).where(monthly_price_m * monthly_volume_m > 0)
    ).shift(1)

    # "Value" (cheapness) uses price relative to a trailing average; the negative sign
    # makes “cheaper” stocks score higher, matching common cross-sectional conventions.
    expected_value = -(
        monthly_price_m
        / monthly_price_m.rolling(12, min_periods=6).mean().replace(0, np.nan)
    )

    # Momentum intentionally skips the most recent month to reduce short-term reversal
    # contamination; it compounds returns over the intermediate window.
    expected_momentum = (
        (1 + monthly_return.shift(2))
        .rolling(11, min_periods=6)
        .apply(lambda x: np.prod(1 + x) - 1, raw=False)
    )

    # Volatility is estimated from daily returns (more granular information), then sampled
    # at month-end and lagged so it is available at the start of the next month.
    daily_ret = price.pct_change(fill_method=None)
    realized_daily_vol = daily_ret.rolling(252, min_periods=63).std(ddof=0) * np.sqrt(252)
    expected_volatility = realized_daily_vol.resample("ME").last().shift(1)

    # Investment is proxied by changes in a trailing average of trading activity; the log
    # stabilizes scale and the diff turns it into a growth/change measure.
    expected_investment = np.log(
        monthly_volume_m.rolling(12, min_periods=6).mean().replace(0, np.nan)
    ).diff()

    # Profitability uses only lagged returns so the feature is fully known before the
    # prediction month; the rolling Sharpe-like ratio summarizes return “quality”.
    lagged = monthly_return.shift(1)
    expected_profitability = (
        lagged.rolling(12, min_periods=6).mean()
        / lagged.rolling(12, min_periods=6).std(ddof=0).replace(0, np.nan)
    )

    # --- Compare output ---
    # The production function drops rows where every feature is missing; because several
    # signals are lagged, the first month is structurally unavailable and should be absent.
    output = features.xs("AAA", level="ticker")

    # We expect month 1 to be dropped and the remaining months to align exactly to month-end.
    expected_index = month_ends[1:].rename("date")
    pd.testing.assert_index_equal(output.index, expected_index)

    # Column-by-column equality checks ensure both definitions and alignment (lags/windows)
    # match the spec, not just overall shape or summary statistics.
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