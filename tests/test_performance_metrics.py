import pandas as pd
import pytest

from src.performance_metrics import (
    compute_long_short_returns,
    compute_turnover_from_weights,
    summarize_portfolio_performance,
)
from src.portfolios import compute_decile_portfolio_weights


def test_long_short_and_metrics(tmp_path):
    index = pd.to_datetime(["2020-01-31", "2020-02-29"])
    returns = pd.DataFrame(
        {
            ("model_a", 1): [0.01, 0.02],
            ("model_a", 10): [0.03, 0.01],
        },
        index=index,
    )

    long_short = compute_long_short_returns(returns)
    expected_long_short = pd.DataFrame({("model_a", "long_short"): [0.02, -0.01]}, index=index)
    pd.testing.assert_frame_equal(long_short, expected_long_short)

    metrics, cumulative, drawdowns = summarize_portfolio_performance(
        returns,
        risk_free_rate=0.0,
        periods_per_year=1,
        plot_path=tmp_path / "cum.png",
    )

    assert (tmp_path / "cum.png").exists()

    # Mean returns are per-period with periods_per_year=1 so no scaling occurs
    assert metrics.loc[("model_a", 10), "mean_return"] == pd.Series([0.03, 0.01]).mean()
    assert metrics.loc[("model_a", "long_short"), "max_drawdown"] == drawdowns[("model_a", "long_short")].min()

    # Sharpe uses sample volatility; approximate expected value
    assert metrics.loc[("model_a", 10), "sharpe_ratio"] == pytest.approx(1.4142135623730951)

    assert cumulative.index.equals(index)


def test_turnover_from_decile_weights():
    index = pd.MultiIndex.from_product(
        [["A", "B"], pd.to_datetime(["2020-01-31", "2020-02-29"])], names=["ticker", "date"]
    )
    panel = pd.DataFrame({"model": [1, 2, 2, 1]}, index=index)

    weights = compute_decile_portfolio_weights(panel, model_cols=["model"], n_deciles=2)
    turnover = compute_turnover_from_weights(weights)

    expected_turnover = pd.DataFrame(
        {
            ("model", 1): [float("nan"), 0.25],
            ("model", 2): [float("nan"), 0.25],
        },
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
    )

    pd.testing.assert_frame_equal(turnover, expected_turnover)


def test_transaction_cost_adjustment():
    index = pd.MultiIndex.from_product(
        [["A", "B"], pd.to_datetime(["2020-01-31", "2020-02-29"])], names=["ticker", "date"]
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
        transaction_cost_bps=10,
        risk_free_rate=0.0,
        periods_per_year=1,
    )

    expected_cost = turnover.reindex(returns.index).fillna(0.0) * 0.001

    adjusted_returns = returns - expected_cost
    expected_mean = adjusted_returns.mean()

    assert metrics.loc[("model", 1), "mean_return"] == expected_mean[("model", 1)]
    assert metrics.loc[("model", 2), "mean_return"] == expected_mean[("model", 2)]
