"""Tests for regularized models in training_scheme.py.

This file validates `rolling_regularized_predictions`, which generates rolling/expanding
out-of-sample forecasts using cross-validated regularized linear regressions.

What we check:
- The function works for each supported regularized model type: Ridge, Lasso, Elastic Net.
- Predictions begin only after `min_train_months` distinct months are available.
- On a synthetic dataset with a deterministic linear relationship, the fitted models
  should closely track the true next_return values out of sample (within tolerance).
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
    RegularizedModelConfig,
    WindowConfig,
    rolling_regularized_predictions,
)


def _build_panel() -> pd.DataFrame:
    """Build a simple panel where next_return is exactly linear in the features.

    Index: MultiIndex ('ticker', 'date')
    Columns:
      - x1, x2: deterministic features varying by ticker and time
      - next_return: 1 + 2*x1 - x2 (no noise)

    With enough training history, a linear model with mild regularization should
    reproduce the mapping very closely.
    """
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    tickers = ["AAA", "BBB", "CCC"]

    records = []
    for i, date in enumerate(dates, start=1):
        for j, ticker in enumerate(tickers, start=1):
            x1 = i + j
            x2 = 0.5 * j
            next_return = 1.0 + 2 * x1 - x2
            records.append((ticker, date, x1, x2, next_return))

    df = pd.DataFrame(records, columns=["ticker", "date", "x1", "x2", "next_return"])
    return df.set_index(["ticker", "date"]).sort_index()


@pytest.mark.parametrize("model_type", ["ridge", "lasso", "elasticnet"])
def test_regularized_models_track_linear_relation(model_type):
    panel = _build_panel()

    # Small rolling window to keep the test quick and ensure we produce OOS rows.
    window_cfg = WindowConfig(min_train_months=2, max_train_months=3, prediction_col="pred")

    # Keep regularization very light so the synthetic linear relationship is recovered.
    model_cfg = RegularizedModelConfig(
        model_type=model_type,
        alphas=[1e-6, 1e-4, 1e-2],
        l1_ratios=[0.5, 0.8] if model_type == "elasticnet" else None,
        cv_folds=3,
        max_iter=5000,
    )

    preds = rolling_regularized_predictions(
        panel,
        ["x1", "x2"],
        window_config=window_cfg,
        model_config=model_cfg,
    )

    # With min_train_months=2, the first prediction occurs at the 3rd distinct month.
    expected_dates = panel.index.get_level_values("date").unique().sort_values()[2:]
    assert list(preds.index.get_level_values("date").unique()) == list(expected_dates)

    # Compare predictions to the true target for the same (ticker, date) rows.
    expected = panel.loc[preds.index, "next_return"].values

    # Allow small numeric slack because the pipeline includes scaling and CV selection.
    np.testing.assert_allclose(preds["pred"].values, expected, rtol=1e-4, atol=1e-6)