"""Tests for oos_predictions.py.

These tests focus on the *behavioral guarantees* of `generate_oos_predictions_all_models()`:

- It should emit one merged panel that includes forecasts from every supported
  model family under a shared rolling-window protocol.
- It must respect the information barrier imposed by `min_train_months`: we
  should not see any out-of-sample predictions before enough history exists.
- When an `output_path` is provided, it should persist the merged panel so the
  downstream analysis pipeline can run without recomputing predictions.
- On a simple synthetic dataset where the target is a stable, noiseless function
  of the features, all model forecasts should line up strongly with realized outcomes.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Allow tests to import `src.*` modules when running from the repository root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.training_scheme import WindowConfig
from src.oos_predictions import generate_oos_predictions_all_models


def _build_synthetic_panel() -> pd.DataFrame:
    """Create a small panel with a deterministic relationship between features and target.

    We use a clean, time-stable data-generating process so that once each model
    has enough history, its forecasts should track realized returns closely.
    This avoids flaky tests driven by random noise.
    """
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    tickers = ["AAA", "BBB"]

    records = []
    for i, date in enumerate(dates, start=1):
        for j, ticker in enumerate(tickers, start=1):
            # Deterministic features that vary across both time and tickers.
            x1 = i + j
            x2 = j

            # Deterministic target: predictable by construction.
            next_ret = 0.5 * x1 + 0.25 * x2
            records.append((ticker, date, x1, x2, next_ret))

    df = pd.DataFrame(records, columns=["ticker", "date", "x1", "x2", "next_return"])
    return df.set_index(["ticker", "date"]).sort_index()


def test_generate_predictions_all_models(tmp_path):
    panel = _build_synthetic_panel()

    # Use a short training requirement so the test has multiple OOS months to check.
    cfg = WindowConfig(min_train_months=2, max_train_months=3, prediction_col="pred")

    # Persisting predictions is part of the public behavior: downstream notebooks/scripts
    # should be able to load the merged CSV without rerunning rolling estimation.
    output_file = tmp_path / "preds.csv"

    dataset = generate_oos_predictions_all_models(
        panel,
        ["x1", "x2"],
        window_config=cfg,
        output_path=output_file,
    )

    # The merged panel should include one column per model plus the realized outcome column.
    expected_cols = {
        "ols_pred",
        "ridge_pred",
        "lasso_pred",
        "elasticnet_pred",
        "random_forest_pred",
        "realized_return",
    }
    assert expected_cols.issubset(dataset.columns)

    # Enforce the "no peeking" rule: predictions can only begin once at least
    # `min_train_months` *distinct months* are available for training.
    prediction_dates = dataset.index.get_level_values("date").unique()
    all_dates = panel.index.get_level_values("date").unique().sort_values()
    first_allowed = all_dates[cfg.min_train_months]
    assert prediction_dates.min() >= first_allowed

    # Verify that persistence works (the write side-effect is a contract).
    assert output_file.exists()
    saved = pd.read_csv(output_file)
    assert {"ols_pred", "realized_return"}.issubset(set(saved.columns))

    # If the synthetic relationship is perfectly predictable, forecasts should align
    # strongly with realized returns wherever all values are present.
    filled = dataset.dropna()
    assert filled.shape[0] > 0

    correlations = filled.corr()["realized_return"].drop("realized_return")
    assert np.all(correlations > 0.8)