"""Tests for the rolling training / tuning utilities in training_scheme.py.

This file focuses on three related helpers:

1) `rolling_oos_predictions`
   - Walks forward through time and produces out-of-sample (OOS) predictions.
   - Uses only data strictly before the prediction month.
   - Supports either rolling windows (bounded by max_train_months) or expanding windows.

2) `WindowConfig`
   - Controls when predictions begin (`min_train_months`) and how much history is used.

3) `rolling_time_series_tuning`
   - Performs time-series cross-validation inside each training window to pick the best
     model from a list of candidates, then refits and predicts the next month.
   - Stores both the predictions and the identity/diagnostics of the chosen model.

The tests use tiny synthetic panels where we know the exact linear relationship.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add repo root so `src.*` imports work when running tests directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.training_scheme import (
    CandidateModel,
    WindowConfig,
    rolling_oos_predictions,
    rolling_time_series_tuning,
)


def _build_panel(values_by_date: dict[pd.Timestamp, dict[str, float]]) -> pd.DataFrame:
    """Create a single-ticker monthly panel from a {date: {col: val}} mapping.

    This helper is intentionally minimal: it constructs a MultiIndex with one
    ticker ("AAA") and one row per date. Each row is a dict containing feature(s)
    and the target column.
    """
    entries = []
    for date, val in values_by_date.items():
        entries.append((("AAA", date), val))

    index, vals = zip(*entries)
    return pd.DataFrame(vals, index=pd.MultiIndex.from_tuples(index, names=["ticker", "date"]))


class _DummyLinearModel:
    """A tiny "model" used to make tuning behavior deterministic.

    It ignores y during fit and predicts `weight * X[:, 0]`.
    """

    def __init__(self, weight: float):
        self.weight = weight

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[:, 0] * self.weight


def test_rolling_predictions_respect_min_and_max_window():
    # Build 6 monthly observations for one ticker where y = 2x exactly.
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    panel = _build_panel({dates[i]: {"feature": i + 1, "next_return": 2 * (i + 1)} for i in range(len(dates))})

    # Predictions start only after we have min_train_months months of history.
    # With min_train_months=3, the first OOS prediction happens at dates[3].
    cfg = WindowConfig(min_train_months=3, max_train_months=4, prediction_col="pred")
    preds = rolling_oos_predictions(panel, ["feature"], window_config=cfg)

    expected_dates = dates[3:]
    assert list(preds.index.get_level_values("date")) == list(expected_dates)

    # With no model_factory provided, rolling_oos_predictions falls back to OLS.
    # The OLS should recover y = 2x perfectly and emit one prediction per OOS month.
    expected_values = 2 * (np.arange(4, 7))  # features at dates[3:] are 4, 5, 6
    np.testing.assert_allclose(preds["pred"].values, expected_values)


def test_expanding_window_grows_history():
    # Constant feature = 1.0, but the target changes over time.
    # With an intercept, the OLS forecast is effectively the mean of past y values.
    dates = pd.date_range("2021-01-31", periods=4, freq="ME")
    panel = _build_panel(
        {
            dates[0]: {"feature": 1.0, "next_return": 1.0},
            dates[1]: {"feature": 1.0, "next_return": 1.0},
            dates[2]: {"feature": 1.0, "next_return": 3.0},
            dates[3]: {"feature": 1.0, "next_return": 5.0},
        }
    )

    # Expanding=True means we always start training at the beginning of the sample.
    cfg = WindowConfig(min_train_months=2, max_train_months=2, expanding=True)
    preds = rolling_oos_predictions(panel, ["feature"], window_config=cfg)

    # For the last month, the training set includes all prior months (3 points).
    # Since X is constant (all ones), the best OLS fit predicts the mean of y:
    # mean([1, 1, 3]) = 5/3.
    last_pred = preds.xs(dates[3], level="date")
    assert last_pred.iloc[0]["prediction"] == pytest.approx(5 / 3)


def test_time_series_tuning_selects_best_candidate_and_tracks_models():
    # Multi-ticker panel where next_return is exactly 2 * feature.
    # We provide two dummy candidates: one underfits (weight=1) and one is correct (weight=2).
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    records = []
    for i, date in enumerate(dates):
        for ticker, offset in [("AAA", 1.0), ("BBB", 2.0)]:
            feature = i + offset
            records.append({"ticker": ticker, "date": date, "feature": feature, "next_return": 2 * feature})

    panel = pd.DataFrame.from_records(records).set_index(["ticker", "date"])

    candidates = [
        CandidateModel("underfit", lambda: _DummyLinearModel(1.0)),
        CandidateModel("best", lambda: _DummyLinearModel(2.0)),
    ]

    # Start predicting at dates[3] because min_train_months=3.
    cfg = WindowConfig(min_train_months=3, max_train_months=4, prediction_col="cv_pred")
    result = rolling_time_series_tuning(
        panel,
        ["feature"],
        candidates=candidates,
        window_config=cfg,
        cv_folds=2,
    )

    preds = result.predictions
    assert not preds.empty
    assert set(preds.columns) == {"cv_pred"}

    expected_dates = dates[cfg.min_train_months :]
    assert list(preds.index.get_level_values("date").unique()) == list(expected_dates)

    # For each OOS month, we should have recorded which candidate was selected and why.
    for date in expected_dates:
        model_info = result.best_models[pd.to_datetime(date)]
        assert model_info.name == "best"
        assert set(model_info.cv_scores) == {"spearman", "r2", "mae"}

        # The chosen model should predict exactly 2 * feature for that month.
        oos_pred = preds.xs(date, level="date")["cv_pred"]
        true_feature = panel.xs(date, level="date")["feature"]
        np.testing.assert_allclose(oos_pred.values, 2 * true_feature.values)