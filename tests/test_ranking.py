"""Tests for ranking.py.

These tests validate `convert_predictions_to_rankings`, which converts a panel of
model predictions into cross-sectional percentile ranks.

Covered behaviors:
- Ranking raw predicted returns within each date.
- Ranking based on cross-sectional z-scores (including the constant cross-section case).
- Optional risk adjustment by dividing predictions by a risk metric (volatility or beta)
  before applying the ranking step.
"""

from pathlib import Path
import sys

import pandas as pd
import pytest

# Add repo root so `src.*` imports work when running tests directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.ranking import RankingConfig, convert_predictions_to_rankings


def _make_index(dates, tickers) -> pd.MultiIndex:
    """Convenience helper to create the expected ('ticker', 'date') MultiIndex."""
    return pd.MultiIndex.from_product([tickers, dates], names=["ticker", "date"])


def test_rank_by_predicted_return():
    # Two cross-sections (two month-ends), three tickers each.
    dates = pd.date_range("2022-01-31", periods=2, freq="ME")
    tickers = ["AAA", "BBB", "CCC"]

    # Note: _make_index uses (tickers, dates) product order, so the rows are:
    # (AAA,d1), (AAA,d2), (BBB,d1), (BBB,d2), (CCC,d1), (CCC,d2)
    preds = pd.DataFrame(
        {"pred": [0.1, 0.3, 0.2, 0.35, 0.05, 0.4]},
        index=_make_index(dates, tickers),
    )

    # Default basis="return": rank the raw predictions within each date.
    config = RankingConfig(prediction_col="pred", output_col="rank")
    ranks = convert_predictions_to_rankings(preds, config=config)

    # In month 1 (date[0]): BBB has highest pred (0.2), AAA is middle (0.1), CCC is lowest (0.05).
    first_month = ranks.xs(dates[0], level="date")
    assert first_month.loc["BBB", "rank"] == pytest.approx(1.0)
    assert first_month.loc["AAA", "rank"] == pytest.approx(2 / 3)
    assert first_month.loc["CCC", "rank"] == pytest.approx(1 / 3)

    # In month 2 (date[1]): CCC is highest (0.4), BBB is middle (0.35), AAA is lowest (0.3).
    second_month = ranks.xs(dates[1], level="date")
    assert second_month.loc["CCC", "rank"] == pytest.approx(1.0)
    assert second_month.loc["BBB", "rank"] == pytest.approx(2 / 3)
    assert second_month.loc["AAA", "rank"] == pytest.approx(1 / 3)


def test_rank_by_zscore_handles_constant_cross_section():
    # A constant cross-section should not blow up z-score computation (std=0).
    dates = pd.date_range("2023-01-31", periods=1, freq="ME")
    tickers = ["AAA", "BBB", "CCC"]
    preds = pd.DataFrame({"pred": [0.0, 0.0, 0.0]}, index=_make_index(dates, tickers))

    # basis="zscore": compute cross-sectional z-scores per date, then rank them.
    config = RankingConfig(prediction_col="pred", output_col="zrank", basis="zscore")
    ranks = convert_predictions_to_rankings(preds, config=config)

    month = ranks.xs(dates[0], level="date")

    # All z-scores are identical (filled to 0.0), so ranks should also be identical
    # under method="average": the average rank among 3 items is 2/3 in percentile terms.
    assert month["zrank"].nunique() == 1
    assert month["zrank"].iloc[0] == pytest.approx(2 / 3)


def test_risk_adjustment_with_volatility():
    # Risk adjustment divides prediction by a risk metric before ranking.
    dates = pd.date_range("2024-01-31", periods=1, freq="ME")
    tickers = ["AAA", "BBB", "CCC"]
    preds = pd.DataFrame(
        {
            "pred": [0.1, 0.2, 0.15],
            "pred_vol": [0.1, 0.4, 0.05],
        },
        index=_make_index(dates, tickers),
    )

    # Adjusted scores: AAA=1.0, BBB=0.5, CCC=3.0 -> CCC highest, AAA middle, BBB lowest.
    config = RankingConfig(
        prediction_col="pred",
        output_col="risk_rank",
        risk_adjust="volatility",
        volatility_col="pred_vol",
    )
    ranks = convert_predictions_to_rankings(preds, config=config)
    month = ranks.xs(dates[0], level="date")

    assert month.loc["CCC", "risk_rank"] == pytest.approx(1.0)
    assert month.loc["AAA", "risk_rank"] == pytest.approx(2 / 3)
    assert month.loc["BBB", "risk_rank"] == pytest.approx(1 / 3)


def test_risk_adjustment_with_beta_and_zscore_basis():
    # Combine risk adjustment (by beta) with z-score basis.
    dates = pd.date_range("2025-01-31", periods=2, freq="ME")
    tickers = ["AAA", "BBB"]

    # Rows are (AAA,d1), (AAA,d2), (BBB,d1), (BBB,d2).
    preds = pd.DataFrame(
        {
            "pred": [0.1, 0.4, 0.3, 0.1],
            "pred_beta": [1.0, 1.0, 2.0, 2.0],
        },
        index=_make_index(dates, tickers),
    )

    # After beta adjustment:
    # d1: AAA=0.1/1=0.1, BBB=0.3/2=0.15 -> BBB should rank higher
    # d2: AAA=0.4/1=0.4, BBB=0.1/2=0.05 -> AAA should rank higher
    #
    # basis="zscore" then standardizes within each date; with 2 names, percentile ranks
    # should be {0.5, 1.0} under method="average".
    config = RankingConfig(
        prediction_col="pred",
        output_col="beta_rank",
        basis="zscore",
        risk_adjust="beta",
        beta_col="pred_beta",
        rank_method="average",
    )
    ranks = convert_predictions_to_rankings(preds, config=config)

    first_month = ranks.xs(dates[0], level="date")
    assert first_month.loc["BBB", "beta_rank"] == pytest.approx(1.0)
    assert first_month.loc["AAA", "beta_rank"] == pytest.approx(0.5)

    second_month = ranks.xs(dates[1], level="date")
    assert second_month.loc["AAA", "beta_rank"] == pytest.approx(1.0)
    assert second_month.loc["BBB", "beta_rank"] == pytest.approx(0.5)