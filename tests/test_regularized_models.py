import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.training_scheme import (
    RegularizedModelConfig,
    WindowConfig,
    rolling_regularized_predictions,
)


def _build_panel():
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    tickers = ["AAA", "BBB", "CCC"]

    records = []
    for i, date in enumerate(dates, start=1):
        for j, ticker in enumerate(tickers, start=1):
            x1 = i + j
            x2 = 0.5 * j
            next_return = 1.0 + 2 * x1 - x2
            records.append((ticker, date, x1, x2, next_return))

    idx = pd.MultiIndex.from_tuples(
        [(t, d) for t, d, *_ in records], names=["ticker", "date"]
    )
    df = pd.DataFrame(records, columns=["ticker", "date", "x1", "x2", "next_return"])
    df = df.set_index(idx)
    return df


@pytest.mark.parametrize("model_type", ["ridge", "lasso", "elasticnet"])
def test_regularized_models_track_linear_relation(model_type):
    panel = _build_panel()

    window_cfg = WindowConfig(min_train_months=2, max_train_months=3, prediction_col="pred")
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

    expected_dates = panel.index.get_level_values("date").unique()[2:]
    assert list(preds.index.get_level_values("date").unique()) == list(expected_dates)

    expected = panel.loc[preds.index, "next_return"].values
    np.testing.assert_allclose(preds["pred"].values, expected, rtol=1e-4, atol=1e-6)