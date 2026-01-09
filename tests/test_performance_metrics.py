"""Tests for performance_metrics.py.

This test module covers the portfolio analytics helpers:

- `compute_long_short_returns`:
  Builds a top-minus-bottom decile spread per model.

- `summarize_portfolio_performance`:
  Computes summary statistics (mean return, volatility, Sharpe, drawdowns) and
  optionally writes a cumulative return plot.

- `compute_turnover_from_weights`:
  Converts per-ticker portfolio weights into per-period turnover.

- Transaction cost adjustment:
  When turnover weights and bps costs are provided, returns should be reduced by
  turnover * cost_per_trade.

- Sharpe ratio significance tests:
  Includes Jobson–Korkie (optionally without Memmel correction) and a bootstrap
  test for Sharpe differences.
"""

import math

import pandas as pd
import pytest

from src.performance_metrics import (
    bootstrap_sharpe_ratio_difference,
    compute_long_short_returns,
    compute_turnover_from_weights,
    jobson_korkie_test,
    summarize_portfolio_performance,
)
from src.portfolios import compute_decile_portfolio_weights


def test_long_short_and_metrics(tmp_path):
    # Two months of returns for a single model with only bottom and top deciles.
    index = pd.to_datetime(["2020-01-31", "2020-02-29"])
    returns = pd.DataFrame(
        {
            ("model_a", 1): [0.01, 0.02],   # bottom decile
            ("model_a", 10): [0.03, 0.01],  # top decile
        },
        index=index,
    )

    # Long-short should be top minus bottom, per date.
    long_short = compute_long_short_returns(returns)
    expected_long_short = pd.DataFrame(
        {("model_a", "long_short"): [0.02, -0.01]},
        index=index,
    )
    pd.testing.assert_frame_equal(long_short, expected_long_short)

    # Summarize performance and request a cumulative plot to be written.
    metrics, cumulative, drawdowns = summarize_portfolio_performance(
        returns,
        risk_free_rate=0.0,
        periods_per_year=1,  # treat each row as "annual" so no scaling happens
        plot_path=tmp_path / "cum.png",
    )

    # Plot output should exist.
    assert (tmp_path / "cum.png").exists()

    # With periods_per_year=1, "mean_return" should be the raw mean of the series.
    assert metrics.loc[("model_a", 10), "mean_return"] == pd.Series([0.03, 0.01]).mean()

    # Max drawdown is computed from the wealth curve; ensure it matches the returned drawdown series.
    assert (
        metrics.loc[("model_a", "long_short"), "max_drawdown"]
        == drawdowns[("model_a", "long_short")].min()
    )

    # Sharpe uses sample volatility (ddof=1); this is a known value for [0.03, 0.01] with rf=0.
    assert metrics.loc[("model_a", 10), "sharpe_ratio"] == pytest.approx(1.4142135623730951)

    # Cumulative index should match the input dates.
    assert cumulative.index.equals(index)


def test_turnover_from_decile_weights():
    # Create a tiny panel where ticker membership flips between two deciles across months.
    index = pd.MultiIndex.from_product(
        [["A", "B"], pd.to_datetime(["2020-01-31", "2020-02-29"])],
        names=["ticker", "date"],
    )
    panel = pd.DataFrame({"model": [1, 2, 2, 1]}, index=index)

    # Build decile weights (here: 2 deciles), then compute turnover.
    weights = compute_decile_portfolio_weights(panel, model_cols=["model"], n_deciles=2)
    turnover = compute_turnover_from_weights(weights)

    # In the first month, turnover is undefined (no prior weights).
    # In the second month, both deciles fully flip -> turnover = 1.0.
    expected_turnover = pd.DataFrame(
        {
            ("model", 1): [float("nan"), 1.0],
            ("model", 2): [float("nan"), 1.0],
        },
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
    )

    pd.testing.assert_frame_equal(turnover, expected_turnover)


def test_transaction_cost_adjustment():
    # Same setup as the turnover test, but now we apply transaction costs to returns.
    index = pd.MultiIndex.from_product(
        [["A", "B"], pd.to_datetime(["2020-01-31", "2020-02-29"])],
        names=["ticker", "date"],
    )
    panel = pd.DataFrame({"model": [1, 2, 2, 1]}, index=index)

    weights = compute_decile_portfolio_weights(panel, model_cols=["model"], n_deciles=2)
    turnover = compute_turnover_from_weights(weights)

    # Decile returns for two months.
    returns = pd.DataFrame(
        {("model", 1): [0.02, 0.03], ("model", 2): [0.01, 0.02]},
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
    )

    # Apply 10 bps transaction cost. In the implementation this is divided by 10,000.
    metrics, _, _ = summarize_portfolio_performance(
        returns,
        turnover_weights=weights,
        transaction_cost_bps=10,
        risk_free_rate=0.0,
        periods_per_year=1,
    )

    # Expected cost = turnover * (bps / 10,000) = turnover * 0.001.
    expected_cost = turnover.reindex(returns.index).fillna(0.0) * 0.001
    adjusted_returns = returns - expected_cost

    # Mean returns in the metrics should reflect the cost-adjusted return series.
    expected_mean = adjusted_returns.mean()
    assert metrics.loc[("model", 1), "mean_return"] == expected_mean[("model", 1)]
    assert metrics.loc[("model", 2), "mean_return"] == expected_mean[("model", 2)]


def test_jobson_korkie_difference():
    # Two model return series with enough observations for Sharpe computation and a JK test.
    returns = pd.DataFrame(
        {
            "model_a": [0.02, 0.01, 0.015, 0.005, 0.018, 0.0],
            "model_b": [0.01, 0.0, 0.005, 0.002, 0.006, -0.002],
        },
        index=pd.date_range("2020-01-31", periods=6, freq="ME"),
    )

    # Disable Memmel correction here because the expected values are benchmarked for the uncorrected statistic.
    result = jobson_korkie_test(
        returns,
        model_1="model_a",
        model_2="model_b",
        risk_free_rate=0.0,
        periods_per_year=12,
        memmel_correction=False,
    )

    # These values are deterministic given the hard-coded input series.
    assert result["periods"] == 6
    assert result["sharpe_diff"] == pytest.approx(2.2662684197995935)
    assert result["jk_stat"] == pytest.approx(1.2718862228771095)
    assert result["p_value"] == pytest.approx(0.20341354758736774)


def test_bootstrap_sharpe_ratio_difference():
    # Same return series as above; now validate the percentile bootstrap logic.
    returns = pd.DataFrame(
        {
            "model_a": [0.02, 0.01, 0.015, 0.005, 0.018, 0.0],
            "model_b": [0.01, 0.0, 0.005, 0.002, 0.006, -0.002],
        },
        index=pd.date_range("2020-01-31", periods=6, freq="ME"),
    )

    # Run the library function.
    result = bootstrap_sharpe_ratio_difference(
        returns,
        model_1="model_a",
        model_2="model_b",
        n_bootstrap=500,
        random_state=123,
        risk_free_rate=0.0,
        periods_per_year=12,
    )

    # Recompute the bootstrap distribution manually to validate mean/std/p-value.
    aligned = returns[["model_a", "model_b"]]
    n = len(aligned)

    import numpy as np  # local import to mirror how the test was originally written

    rng = np.random.default_rng(123)
    diffs = []
    for _ in range(500):
        # Resample indices with replacement.
        idx = rng.integers(0, n, size=n)
        sample = aligned.iloc[idx]

        # Annualized Sharpe ratio with sample std (ddof=1).
        sr1 = sample["model_a"].mean() / sample["model_a"].std(ddof=1) * math.sqrt(12)
        sr2 = sample["model_b"].mean() / sample["model_b"].std(ddof=1) * math.sqrt(12)
        diffs.append(sr1 - sr2)

    boot_diffs = np.array(diffs)

    # Check reported stats against our manual reconstruction.
    assert result["periods"] == 6
    assert result["sharpe_diff"] == pytest.approx(2.2662684197995935)
    assert result["bootstrap_mean"] == pytest.approx(boot_diffs.mean())
    assert result["bootstrap_std"] == pytest.approx(boot_diffs.std(ddof=1))

    # Two-sided p-value from the bootstrap distribution (percentile logic).
    assert result["p_value"] == pytest.approx(
        2
        * min(
            (boot_diffs >= result["sharpe_diff"]).mean(),
            (boot_diffs <= result["sharpe_diff"]).mean(),
        )
    )