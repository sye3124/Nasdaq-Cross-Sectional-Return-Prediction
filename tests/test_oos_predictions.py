"""Tests for oos_predictions.py.

This file checks that `generate_oos_predictions_all_models()`:
- produces a single merged panel containing predictions from all supported model families
  (OLS, regularized linear models, and random forest)
- respects the rolling window configuration (no predictions before min_train_months)
- optionally writes the merged output to disk
- yields forecasts that are meaningfully related to the target on a simple synthetic dataset
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add repo root so tests can run without installing the package.
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.training_scheme import WindowConfig
from src.oos_predictions import generate_oos_predictions_all_models


def _build_synthetic_panel() -> pd.DataFrame:
    """Create a tiny panel where next_return is a simple linear function of features.

    The panel is indexed by ('ticker', 'date') and includes:
    - x1, x2: deterministic features
    - next_return: a clean linear combination of x1 and x2

    Because the data is noiseless and stable, all models should generate
    predictions that strongly correlate with realized_return.
    """
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    tickers = ["AAA", "BBB"]

    records = []
    for i, date in enumerate(dates, start=1):
        for j, ticker in enumerate(tickers, start=1):
            # Simple deterministic feature map.
            x1 = i + j
            x2 = j

            # Target is linear in the features.
            next_ret = 0.5 * x1 + 0.25 * x2
            records.append((ticker, date, x1, x2, next_ret))

    df = pd.DataFrame(records, columns=["ticker", "date", "x1", "x2", "next_return"])
    return df.set_index(["ticker", "date"]).sort_index()


def test_generate_predictions_all_models(tmp_path):
    panel = _build_synthetic_panel()

    # Configure a small rolling window so the test has enough OOS dates quickly.
    cfg = WindowConfig(min_train_months=2, max_train_months=3, prediction_col="pred")

    # Write merged predictions to disk to ensure persistence works.
    output_file = tmp_path / "preds.csv"

    dataset = generate_oos_predictions_all_models(
        panel,
        ["x1", "x2"],
        window_config=cfg,
        output_path=output_file,
    )

    # The helper should create one column per model plus the realized return column.
    expected_cols = {
        "ols_pred",
        "ridge_pred",
        "lasso_pred",
        "elasticnet_pred",
        "random_forest_pred",
        "realized_return",
    }
    assert expected_cols.issubset(dataset.columns)

    # Predictions should only start after we have `min_train_months` distinct months.
    prediction_dates = dataset.index.get_level_values("date").unique()
    first_allowed = panel.index.get_level_values("date").unique().sort_values()[cfg.min_train_months]
    assert prediction_dates.min() >= first_allowed

    # The output CSV should exist and contain at least the core columns.
    assert output_file.exists()
    saved = pd.read_csv(output_file)
    assert {"ols_pred", "realized_return"}.issubset(set(saved.columns))

    # Basic sanity check: with a deterministic linear DGP, forecasts should be highly correlated
    # with realized returns once we drop rows with missing predictions.
    filled = dataset.dropna()
    assert filled.shape[0] > 0

    correlations = filled.corr()["realized_return"].drop("realized_return")
    assert np.all(correlations > 0.8)