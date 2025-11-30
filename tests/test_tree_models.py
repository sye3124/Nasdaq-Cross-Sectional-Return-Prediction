import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.tree_models import (
    GradientBoostingConfig,
    RandomForestConfig,
    cross_sectional_gradient_boosting,
    cross_sectional_random_forest,
)


def _make_panel():
    dates = pd.to_datetime(["2020-01-31", "2020-02-29"])
    idx = pd.MultiIndex.from_product([list("ABCD"), dates], names=["ticker", "date"])
    df = pd.DataFrame(index=idx)
    df["x1"] = [1, 2, 3, 4, 2, 4, 6, 8]
    df["x2"] = [0.5, 1.5, 2.0, 3.0, 1.0, 1.0, 2.0, 2.0]

    target = pd.Series(index=idx, dtype=float)
    target.loc[(slice(None), dates[0])] = 1 + 2 * df.loc[(slice(None), dates[0]), "x1"] - df.loc[(slice(None), dates[0]), "x2"]
    target.loc[(slice(None), dates[1])] = 0.5 + 1.5 * df.loc[(slice(None), dates[1]), "x1"] + 0.5 * df.loc[(slice(None), dates[1]), "x2"]
    df["next_return"] = target
    return df


def test_random_forest_fits_cross_section_exactly():
    panel = _make_panel()
    cfg = RandomForestConfig(n_estimators=1, max_depth=None, bootstrap=False, random_state=0)
    preds = cross_sectional_random_forest(panel, ["x1", "x2"], config=cfg)

    pd.testing.assert_index_equal(preds.index, panel.index)
    np.testing.assert_allclose(preds[cfg.prediction_col].values, panel["next_return"].values)


def test_random_forest_rank_output():
    panel = _make_panel()
    cfg = RandomForestConfig(prediction_type="rank", prediction_col="rank", n_estimators=1, bootstrap=False)
    ranks = cross_sectional_random_forest(panel, ["x1", "x2"], config=cfg)

    expected = panel.groupby(level="date")["next_return"].rank(pct=True, method="average")
    pd.testing.assert_series_equal(ranks["rank"], expected.rename("rank"))


def test_gradient_boosting_predictions_track_target():
    pytest.importorskip("xgboost")
    panel = _make_panel()
    cfg = GradientBoostingConfig(
        booster="xgboost",
        n_estimators=50,
        max_depth=3,
        learning_rate=0.2,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=0,
    )

    preds = cross_sectional_gradient_boosting(panel, ["x1", "x2"], config=cfg)
    mae = np.mean(np.abs(preds[cfg.prediction_col].values - panel["next_return"].values))
    assert mae < 1e-3