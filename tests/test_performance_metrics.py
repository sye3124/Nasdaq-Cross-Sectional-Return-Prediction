"""Tests for performance_metrics.py.

This module validates the portfolio analytics and inference utilities used
throughout the project. The tests are designed to be small, deterministic,
and interpretable, so failures are easy to diagnose.

Covered components
------------------
- compute_long_short_returns
    Constructs top-minus-bottom (long–short) portfolios from decile returns.

- summarize_portfolio_performance
    Produces core performance statistics (mean return, volatility, Sharpe ratio,
    drawdowns) and optionally saves cumulative return plots.

- compute_turnover_from_weights
    Translates per-ticker portfolio weights into per-period turnover measures.

- Transaction cost adjustment
    Verifies that portfolio returns are correctly reduced by
    turnover × transaction_cost_bps.

- Sharpe ratio significance tests
    Includes both analytical (Jobson–Korkie, with optional Memmel correction)
    and resampling-based (bootstrap) tests for Sharpe differences.
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
    """End-to-end check of long–short construction and performance summaries."""
    # Two periods of returns for a single model with bottom and top deciles only.
    index = pd.to_datetime(["2020-01-31", "2020-02-29"])
    returns = pd.DataFrame(
        {
            ("model_a", 1): [0.01, 0.02],   # bottom decile
            ("model_a", 10): [0.03, 0.01],  # top decile
        },
        index=index,
    )

    # Long–short = top minus bottom, computed per date.
    long_short = compute_long_short_returns(returns)
    expected_long_short = pd.DataFrame(
        {("model_a", "long_short"): [0.02, -0.01]},
        index=index,
    )
    pd.testing.assert_frame_equal(long_short, expected_long_short)

    # Summarize performance and request a cumulative plot.
    metrics, cumulative, drawdowns = summarize_portfolio_performance(
        returns,
        risk_free_rate=0.0,
        periods_per_year=1,  # treat each row as an annual observation
        plot_path=tmp_path / "cum.png",
    )

    # Plot output should be created.
    assert (tmp_path / "cum.png").exists()

    # With periods_per_year=1, the mean return is just the sample mean.
    assert metrics.loc[("model_a", 10), "mean_return"] == pd.Series([0.03, 0.01]).mean()

    # Max drawdown reported in the metrics must match the drawdown series minimum.
    assert (
        metrics.loc[("model_a", "long_short"), "max_drawdown"]
        == drawdowns[("model_a", "long_short")].min()
    )

    # Sharpe ratio uses sample volatility (ddof=1); this value is deterministic here.
    assert metrics.loc[("model_a", 10), "sharpe_ratio"] == pytest.approx(1.4142135623730951)

    # Cumulative returns should preserve the original date index.
    assert cumulative.index.equals(index)


def test_turnover_from_decile_weights():
    """Verify turnover computation from per-ticker decile weights."""
    index = pd.MultiIndex.from_product(
        [["A", "B"], pd.to_datetime(["2020-01-31", "2020-02-29"])],
        names=["ticker", "date"],
    )
    panel = pd.DataFrame({"model": [1, 2, 2, 1]}, index=index)

    # Build decile weights (two deciles) and compute turnover.
    weights = compute_decile_portfolio_weights(panel, model_cols=["model"], n_deciles=2)
    turnover = compute_turnover_from_weights(weights)

    # First period has no prior holdings; second period fully flips membership.
    expected_turnover = pd.DataFrame(
        {
            ("model", 1): [float("nan"), 1.0],
            ("model", 2): [float("nan"), 1.0],
        },
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
    )

    pd.testing.assert_frame_equal(turnover, expected_turnover)


def test_transaction_cost_adjustment():
    """Ensure transaction costs reduce returns in proportion to turnover."""
    index = pd.MultiIndex.from_product(
        [["A", "B"], pd.to_datetime(["2020-01-31", "2020-02-29"])],
        names=["ticker", "date"],
    )
    panel = pd.DataFrame({"model": [1, 2, 2, 1]}, index=index)

    weights = compute_decile_portfolio_weights(panel, model_cols=["model"], n_deciles=2)
    turnover = compute_turnover_from_weights(weights)

    returns = pd.DataFrame(
        {("model", 1): [0.02, 0.03], ("model", 2): [0.01, 0.02]},
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
    )

    metrics, _, _ = summarize_portfolio_performance(
        returns,
        turnover_weights=weights,
        transaction_cost_bps=10,  # 10 bps = 0.001
        risk_free_rate=0.0,
        periods_per_year=1,
    )

    expected_cost = turnover.reindex(returns.index).fillna(0.0) * 0.001
    adjusted_returns = returns - expected_cost
    expected_mean = adjusted_returns.mean()

    assert metrics.loc[("model", 1), "mean_return"] == expected_mean[("model", 1)]
    assert metrics.loc[("model", 2), "mean_return"] == expected_mean[("model", 2)]


def test_jobson_korkie_difference():
    """Check the uncorrected Jobson–Korkie Sharpe difference test."""
    returns = pd.DataFrame(
        {
            "model_a": [0.02, 0.01, 0.015, 0.005, 0.018, 0.0],
            "model_b": [0.01, 0.0, 0.005, 0.002, 0.006, -0.002],
        },
        index=pd.date_range("2020-01-31", periods=6, freq="ME"),
    )

    result = jobson_korkie_test(
        returns,
        model_1="model_a",
        model_2="model_b",
        risk_free_rate=0.0,
        periods_per_year=12,
        memmel_correction=False,
    )

    assert result["periods"] == 6
    assert result["sharpe_diff"] == pytest.approx(2.2662684197995935)
    assert result["jk_stat"] == pytest.approx(1.2718862228771095)
    assert result["p_value"] == pytest.approx(0.20341354758736774)


def test_bootstrap_sharpe_ratio_difference():
    """Validate bootstrap inference for Sharpe ratio differences."""
    returns = pd.DataFrame(
        {
            "model_a": [0.02, 0.01, 0.015, 0.005, 0.018, 0.0],
            "model_b": [0.01, 0.0, 0.005, 0.002, 0.006, -0.002],
        },
        index=pd.date_range("2020-01-31", periods=6, freq="ME"),
    )

    result = bootstrap_sharpe_ratio_difference(
        returns,
        model_1="model_a",
        model_2="model_b",
        n_bootstrap=500,
        random_state=123,
        risk_free_rate=0.0,
        periods_per_year=12,
    )

    aligned = returns[["model_a", "model_b"]]
    n = len(aligned)

    import numpy as np

    rng = np.random.default_rng(123)
    diffs = []
    for _ in range(500):
        idx = rng.integers(0, n, size=n)
        sample = aligned.iloc[idx]

        sr1 = sample["model_a"].mean() / sample["model_a"].std(ddof=1) * math.sqrt(12)
        sr2 = sample["model_b"].mean() / sample["model_b"].std(ddof=1) * math.sqrt(12)
        diffs.append(sr1 - sr2)

    boot_diffs = np.array(diffs)

    assert result["periods"] == 6
    assert result["sharpe_diff"] == pytest.approx(2.2662684197995935)
    assert result["bootstrap_mean"] == pytest.approx(boot_diffs.mean())
    assert result["bootstrap_std"] == pytest.approx(boot_diffs.std(ddof=1))
    assert result["p_value"] == pytest.approx(
        2
        * min(
            (boot_diffs >= result["sharpe_diff"]).mean(),
            (boot_diffs <= result["sharpe_diff"]).mean(),
        )
    )