from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.ranking import RankingConfig, convert_predictions_to_rankings


def _make_index(dates, tickers):
    return pd.MultiIndex.from_product([tickers, dates], names=["ticker", "date"])


def test_rank_by_predicted_return():
    dates = pd.date_range("2022-01-31", periods=2, freq="ME")
    tickers = ["AAA", "BBB", "CCC"]
    preds = pd.DataFrame(
    {"pred": [0.1, 0.3, 0.2, 0.35, 0.05, 0.4]},
    index=_make_index(dates, tickers),
    )

    config = RankingConfig(prediction_col="pred", output_col="rank")
    ranks = convert_predictions_to_rankings(preds, config=config)

    first_month = ranks.xs(dates[0], level="date")
    assert first_month.loc["BBB", "rank"] == pytest.approx(1.0)
    assert first_month.loc["AAA", "rank"] == pytest.approx(2 / 3)
    assert first_month.loc["CCC", "rank"] == pytest.approx(1 / 3)

    second_month = ranks.xs(dates[1], level="date")
    assert second_month.loc["CCC", "rank"] == pytest.approx(1.0)
    assert second_month.loc["BBB", "rank"] == pytest.approx(2 / 3)
    assert second_month.loc["AAA", "rank"] == pytest.approx(1 / 3)


def test_rank_by_zscore_handles_constant_cross_section():
    dates = pd.date_range("2023-01-31", periods=1, freq="ME")
    tickers = ["AAA", "BBB", "CCC"]
    preds = pd.DataFrame({"pred": [0.0, 0.0, 0.0]}, index=_make_index(dates, tickers))

    config = RankingConfig(prediction_col="pred", output_col="zrank", basis="zscore")
    ranks = convert_predictions_to_rankings(preds, config=config)

    month = ranks.xs(dates[0], level="date")
    assert month["zrank"].nunique() == 1
    assert month["zrank"].iloc[0] == pytest.approx(2 / 3)


def test_risk_adjustment_with_volatility():
    dates = pd.date_range("2024-01-31", periods=1, freq="ME")
    tickers = ["AAA", "BBB", "CCC"]
    preds = pd.DataFrame(
        {
            "pred": [0.1, 0.2, 0.15],
            "pred_vol": [0.1, 0.4, 0.05],
        },
        index=_make_index(dates, tickers),
    )

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
    dates = pd.date_range("2025-01-31", periods=2, freq="ME")
    tickers = ["AAA", "BBB"]
    preds = pd.DataFrame(
    {
        # rows: (AAA,d1), (AAA,d2), (BBB,d1), (BBB,d2)
        "pred":      [0.1, 0.4, 0.3, 0.1],
        "pred_beta": [1.0, 1.0, 2.0, 2.0],
    },
    index=_make_index(dates, tickers),
    )

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