import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.linear_models import CrossSectionalOLSConfig, cross_sectional_ols


def _make_panel():
    dates = pd.to_datetime(["2020-01-31", "2020-02-29"])  # ME freq but explicit dates ok
    idx = pd.MultiIndex.from_product([list("ABCD"), dates], names=["ticker", "date"])
    df = pd.DataFrame(index=idx)
    df["x1"] = [1, 2, 3, 4, 2, 4, 6, 8]
    df["x2"] = [0.5, 1.5, 2.0, 3.0, 1.0, 1.0, 2.0, 2.0]

    # date 1: y = 1 + 2*x1 - 1*x2
    # date 2: y = 0.5 + 1.5*x1 + 0.5*x2
    target = pd.Series(index=idx, dtype=float)
    target.loc[(slice(None), dates[0])] = 1 + 2 * df.loc[(slice(None), dates[0]), "x1"] - df.loc[(slice(None), dates[0]), "x2"]
    target.loc[(slice(None), dates[1])] = 0.5 + 1.5 * df.loc[(slice(None), dates[1]), "x1"] + 0.5 * df.loc[(slice(None), dates[1]), "x2"]

    df["next_return"] = target
    return df


def test_cross_sectional_predictions_match_true_model():
    panel = _make_panel()
    preds = cross_sectional_ols(panel, ["x1", "x2"])

    # Predictions should match the constructed linear relationship exactly
    pd.testing.assert_index_equal(preds.index, panel.index)
    np.testing.assert_allclose(preds["prediction"].values, panel["next_return"].values)


def test_cross_sectional_rank_output():
    panel = _make_panel()
    cfg = CrossSectionalOLSConfig(prediction_type="rank", prediction_col="rank")
    ranks = cross_sectional_ols(panel, ["x1", "x2"], config=cfg)

    # Ranks should be percentiles within each date
    expected_ranks = panel.groupby(level="date")["next_return"].rank(pct=True, method="average")
    pd.testing.assert_series_equal(ranks["rank"], expected_ranks.rename("rank"))