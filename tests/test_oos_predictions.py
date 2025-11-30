from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(ROOT))

from src.training_scheme import WindowConfig
from src.oos_predictions import generate_oos_predictions_all_models


def _build_synthetic_panel():
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    tickers = ["AAA", "BBB"]

    records = []
    for i, date in enumerate(dates, start=1):
        for j, ticker in enumerate(tickers, start=1):
            x1 = i + j
            x2 = j
            next_ret = 0.5 * x1 + 0.25 * x2
            records.append((ticker, date, x1, x2, next_ret))

    df = pd.DataFrame(records, columns=["ticker", "date", "x1", "x2", "next_return"])
    df = df.set_index(["ticker", "date"])
    return df


def test_generate_predictions_all_models(tmp_path):
    panel = _build_synthetic_panel()

    cfg = WindowConfig(min_train_months=2, max_train_months=3, prediction_col="pred")
    output_file = tmp_path / "preds.csv"

    dataset = generate_oos_predictions_all_models(
        panel, ["x1", "x2"], window_config=cfg, output_path=output_file
    )

    expected_cols = {
        "ols_pred",
        "ridge_pred",
        "lasso_pred",
        "elasticnet_pred",
        "random_forest_pred",
        "realized_return",
    }
    assert expected_cols.issubset(dataset.columns)

    # All models should produce at least one prediction and align on the index
    prediction_dates = dataset.index.get_level_values("date").unique()
    assert prediction_dates.min() >= panel.index.get_level_values("date").unique()[cfg.min_train_months]

    # Save-to-disk path should exist and contain the merged dataset
    assert output_file.exists()
    saved = pd.read_csv(output_file)
    assert set(saved.columns) >= {"ols_pred", "realized_return"}

    # Basic accuracy check: predictions should correlate with the true returns
    filled = dataset.dropna()
    assert filled.shape[0] > 0
    correlations = filled.corr()["realized_return"].drop("realized_return")
    assert np.all(correlations > 0.8)