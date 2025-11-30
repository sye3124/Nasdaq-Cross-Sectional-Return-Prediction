import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.training_scheme import WindowConfig, rolling_oos_predictions


def _build_panel(values_by_date):
    entries = []
    for date, val in values_by_date.items():
        entries.append((("AAA", date), val))
    index, vals = zip(*entries)
    return pd.DataFrame(
        vals, index=pd.MultiIndex.from_tuples(index, names=["ticker", "date"])
    )


def test_rolling_predictions_respect_min_and_max_window():
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    panel = _build_panel(
        {
            dates[i]: {"feature": i + 1, "next_return": 2 * (i + 1)}
            for i in range(len(dates))
        }
    )

    cfg = WindowConfig(min_train_months=3, max_train_months=4, prediction_col="pred")
    preds = rolling_oos_predictions(panel, ["feature"], window_config=cfg)

    expected_dates = dates[3:]
    assert list(preds.index.get_level_values("date")) == list(expected_dates)

    # Model should recover the linear relation y = 2x and store one prediction per OOS date.
    expected_values = 2 * (np.arange(4, 7))
    np.testing.assert_allclose(preds["pred"].values, expected_values)


def test_expanding_window_grows_history():
    dates = pd.date_range("2021-01-31", periods=4, freq="ME")
    panel = _build_panel(
        {
            dates[0]: {"feature": 1.0, "next_return": 1.0},
            dates[1]: {"feature": 1.0, "next_return": 1.0},
            dates[2]: {"feature": 1.0, "next_return": 3.0},
            dates[3]: {"feature": 1.0, "next_return": 5.0},
        }
    )

    cfg = WindowConfig(min_train_months=2, max_train_months=2, expanding=True)
    preds = rolling_oos_predictions(panel, ["feature"], window_config=cfg)

    # With an expanding window, the final prediction should use all three prior months,
    # resulting in the mean of [1, 1, 3] = 5/3 with a single feature equal to one.
    last_pred = preds.xs(dates[3], level="date")
    assert last_pred.iloc[0]["prediction"] == pytest.approx(5 / 3)
