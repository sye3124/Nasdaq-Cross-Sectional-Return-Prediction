"""Tests for linear_models.py.

These tests validate the core behavior of `cross_sectional_ols`:

- When the target is generated from an exact linear model (with an intercept),
  the per-date OLS fit should reproduce the target perfectly.
- When `prediction_type="rank"`, the output should be percentile ranks within
  each date's cross-section, matching pandas' ranking semantics.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add repository root to the import path so `src.*` imports work in-place.
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.linear_models import CrossSectionalOLSConfig, cross_sectional_ols


def _make_panel() -> pd.DataFrame:
    """Create a small two-date panel with a known linear target per date."""
    # Explicit month-end dates; using "ME" would also be fine, but this is deterministic.
    dates = pd.to_datetime(["2020-01-31", "2020-02-29"])

    # 4 tickers across 2 dates -> 8 rows total.
    idx = pd.MultiIndex.from_product([list("ABCD"), dates], names=["ticker", "date"])
    df = pd.DataFrame(index=idx)

    # Two features with enough variation to fit a stable regression each date.
    df["x1"] = [1, 2, 3, 4, 2, 4, 6, 8]
    df["x2"] = [0.5, 1.5, 2.0, 3.0, 1.0, 1.0, 2.0, 2.0]

    # Construct the target so each date has its own true linear relationship.
    #
    # date 1: y = 1 + 2*x1 - 1*x2
    # date 2: y = 0.5 + 1.5*x1 + 0.5*x2
    target = pd.Series(index=idx, dtype=float)

    # Fill in targets for date 1.
    target.loc[(slice(None), dates[0])] = (
        1
        + 2 * df.loc[(slice(None), dates[0]), "x1"]
        - df.loc[(slice(None), dates[0]), "x2"]
    )

    # Fill in targets for date 2.
    target.loc[(slice(None), dates[1])] = (
        0.5
        + 1.5 * df.loc[(slice(None), dates[1]), "x1"]
        + 0.5 * df.loc[(slice(None), dates[1]), "x2"]
    )

    # The modeling API expects the target column to be called "next_return" by default.
    df["next_return"] = target
    return df


def test_cross_sectional_predictions_match_true_model():
    # With a perfectly linear DGP (and enough observations), OLS should fit exactly.
    panel = _make_panel()
    preds = cross_sectional_ols(panel, ["x1", "x2"])

    # The function should return predictions for every row in the original panel.
    pd.testing.assert_index_equal(preds.index, panel.index)

    # Because the target is exactly linear in x1/x2 with an intercept, predictions match perfectly.
    np.testing.assert_allclose(preds["prediction"].values, panel["next_return"].values)


def test_cross_sectional_rank_output():
    # When configured for ranks, the model still fits OLS, but returns percentile ranks per date.
    panel = _make_panel()
    cfg = CrossSectionalOLSConfig(prediction_type="rank", prediction_col="rank")
    ranks = cross_sectional_ols(panel, ["x1", "x2"], config=cfg)

    # Since predictions match next_return exactly in this synthetic setup, ranking either gives the same result.
    expected_ranks = panel.groupby(level="date")["next_return"].rank(pct=True, method="average")

    pd.testing.assert_series_equal(ranks["rank"], expected_ranks.rename("rank"))