import numpy as np
import pandas as pd

from features import FeatureConfig, compute_features


def _build_daily_panel(ticker: str, monthly_price: pd.Series, monthly_volume: pd.Series) -> pd.DataFrame:
    rows = []
    prev_price = monthly_price.iloc[0]

    for date, price_end in monthly_price.items():
        start = date.to_period("M").start_time
        end = date
        days = pd.date_range(start, end, freq="D")
        prices = np.linspace(prev_price, price_end, len(days))
        volumes = np.full(len(days), monthly_volume.loc[date])
        rows.append(pd.DataFrame({"adj_close": prices, "volume": volumes}, index=days))
        prev_price = price_end

    daily = pd.concat(rows)
    daily["ticker"] = ticker
    daily = daily.reset_index().rename(columns={"index": "date"})
    return daily.set_index(["ticker", "date"]).sort_index()


def test_compute_features_moving_average_and_sharpe():
    # Build month-end series (the module resamples with "ME", i.e. month-end at midnight)
    month_ends = pd.date_range("2020-01-31", periods=15, freq="ME")
    # Keep the synthetic paths the same as before (values aligned by position)
    monthly_price = pd.Series(np.linspace(10.0, 24.0, len(month_ends)), index=month_ends)
    monthly_volume = pd.Series(np.linspace(100.0, 240.0, len(month_ends)), index=month_ends)

    raw = _build_daily_panel("AAA", monthly_price, monthly_volume)
    features = compute_features(raw)

    # Expected monthly inputs as the module constructs them
    # (Price: last of month, Volume: mean of month)
    daily = raw.xs("AAA", level="ticker")
    price = daily["adj_close"]
    volume = daily["volume"]

    monthly_price_m = price.resample("ME").last()
    monthly_volume_m = volume.resample("ME").mean()
    monthly_return = monthly_price_m.pct_change()

    # Module feature definitions
    expected_dollar_volume = np.log((monthly_price_m * monthly_volume_m).where(monthly_price_m * monthly_volume_m > 0)).shift(1)

    expected_value = -(monthly_price_m / monthly_price_m.rolling(12, min_periods=6).mean().replace(0, np.nan))

    expected_momentum = (
        (1 + monthly_return.shift(2))
        .rolling(11, min_periods=6)
        .apply(lambda x: np.prod(1 + x) - 1, raw=False)
    )

    # Volatility in module: daily returns -> rolling std -> resample month-end -> shift(1)
    daily_ret = price.pct_change(fill_method=None)
    realized_daily_vol = (
        daily_ret.rolling(252, min_periods=63).std(ddof=0) * np.sqrt(252)
    )
    expected_volatility = realized_daily_vol.resample("ME").last().shift(1)

    expected_investment = np.log(monthly_volume_m.rolling(12, min_periods=6).mean().replace(0, np.nan)).diff()

    lagged = monthly_return.shift(1)
    expected_profitability = (
        lagged.rolling(12, min_periods=6).mean()
        / lagged.rolling(12, min_periods=6).std(ddof=0).replace(0, np.nan)
    )

    # Output expectations:
    # The module drops rows where all features are missing; with the shift(1),
    # the first month is all-NaN, so we expect months 2..15 (14 rows).
    output = features.xs("AAA", level="ticker")

    expected_index = month_ends[1:]
    expected_index = expected_index.rename("date")
    pd.testing.assert_index_equal(output.index, expected_index)

    pd.testing.assert_series_equal(output["dollar_volume"], expected_dollar_volume.loc[expected_index], check_names=False)
    pd.testing.assert_series_equal(output["value"], expected_value.loc[expected_index], check_names=False)
    pd.testing.assert_series_equal(output["momentum"], expected_momentum.loc[expected_index], check_names=False)
    pd.testing.assert_series_equal(output["volatility"], expected_volatility.loc[expected_index], check_names=False)
    pd.testing.assert_series_equal(output["investment"], expected_investment.loc[expected_index], check_names=False)
    pd.testing.assert_series_equal(output["profitability"], expected_profitability.loc[expected_index], check_names=False)