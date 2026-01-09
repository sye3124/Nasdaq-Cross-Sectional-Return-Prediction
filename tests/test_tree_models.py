"""Tests for tree_models.py.

These tests cover the cross-sectional tree-based estimators that fit *separate*
models for each date in a ('ticker', 'date') panel.

What we check:
- Random forests can fit a simple deterministic cross-section and return predictions
  aligned to the original panel index.
- The optional "rank" output mode returns percentile ranks within each date.
- Gradient boosting (XGBoost) produces predictions that closely track the target on
  a small synthetic dataset (skipped if xgboost is not installed).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add repo root so `src.*` imports work when running tests directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.tree_models import (
    GradientBoostingConfig,
    RandomForestConfig,
    cross_sectional_gradient_boosting,
    cross_sectional_random_forest,
)


def _make_panel() -> pd.DataFrame:
    """Create a tiny two-date panel with a deterministic target per date.

    We build two cross-sections (two month-ends) for tickers A-D:
      - date 1: y = 1 + 2*x1 - 1*x2
      - date 2: y = 0.5 + 1.5*x1 + 0.5*x2

    This gives us a known "ground truth" to compare against predictions.
    """
    dates = pd.to_datetime(["2020-01-31", "2020-02-29"])
    idx = pd.MultiIndex.from_product([list("ABCD"), dates], names=["ticker", "date"])

    df = pd.DataFrame(index=idx)
    df["x1"] = [1, 2, 3, 4, 2, 4, 6, 8]
    df["x2"] = [0.5, 1.5, 2.0, 3.0, 1.0, 1.0, 2.0, 2.0]

    # Build target piecewise by date to ensure the model must fit each date separately.
    target = pd.Series(index=idx, dtype=float)
    target.loc[(slice(None), dates[0])] = (
        1 + 2 * df.loc[(slice(None), dates[0]), "x1"] - df.loc[(slice(None), dates[0]), "x2"]
    )
    target.loc[(slice(None), dates[1])] = (
        0.5 + 1.5 * df.loc[(slice(None), dates[1]), "x1"] + 0.5 * df.loc[(slice(None), dates[1]), "x2"]
    )

    df["next_return"] = target
    return df


def test_random_forest_fits_cross_section_exactly():
    panel = _make_panel()

    # Use a very small forest to keep the test fast/deterministic.
    # With max_depth=None and bootstrap=False, a single tree can perfectly memorize this toy dataset.
    cfg = RandomForestConfig(n_estimators=1, max_depth=None, bootstrap=False, random_state=0)
    preds = cross_sectional_random_forest(panel, ["x1", "x2"], config=cfg)

    # Predictions should exist for every row and keep the same MultiIndex ordering.
    pd.testing.assert_index_equal(preds.index, panel.index)

    # On this tiny dataset, a single tree should match the target essentially exactly.
    np.testing.assert_allclose(preds[cfg.prediction_col].values, panel["next_return"].values)


def test_random_forest_rank_output():
    panel = _make_panel()

    # Rank mode returns percentile ranks within each cross-section (date).
    cfg = RandomForestConfig(prediction_type="rank", prediction_col="rank", n_estimators=1, bootstrap=False)
    ranks = cross_sectional_random_forest(panel, ["x1", "x2"], config=cfg)

    # If the forest predicts exactly, ranking predictions equals ranking the true target.
    expected = panel.groupby(level="date")["next_return"].rank(pct=True, method="average")
    pd.testing.assert_series_equal(ranks["rank"], expected.rename("rank"))


def test_gradient_boosting_predictions_track_target():
    # Skip cleanly if the optional dependency isn't installed in the environment.
    pytest.importorskip("xgboost")

    panel = _make_panel()

    # Use an XGBoost regressor with enough trees to fit the toy mapping closely.
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

    # We don't require exact equality here, just that the fit is very tight.
    mae = np.mean(np.abs(preds[cfg.prediction_col].values - panel["next_return"].values))
    assert mae < 1e-3