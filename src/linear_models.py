"""Cross-sectional linear baselines for return prediction.

The routines here fit a separate ordinary least squares (OLS) regression at each
date using the available cross-section of tickers:

    next_return ~ alpha + b1 * x1 + ... + bk * xk

The fitted coefficients for a given date are then used to generate predictions
for that same cross-section. Optionally, predictions can be converted to
percentile ranks within each date (handy for long/short portfolio construction
or rank-based evaluation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

PredictionType = Literal["prediction", "rank"]


def _validate_multiindex(panel: pd.DataFrame) -> None:
    """Require a (ticker, date) MultiIndex so we can group cleanly by date."""
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.names[:2] != ["ticker", "date"]:
        raise ValueError("panel must be indexed by ('ticker', 'date').")


@dataclass
class CrossSectionalOLSConfig:
    """Configuration for cross-sectional OLS runs.

    Parameters
    ----------
    prediction_col
        Name of the output column in the returned DataFrame.
    prediction_type
        - "prediction": return raw fitted values
        - "rank": return percentile ranks within each date
    rank_method
        Ranking method forwarded to ``pandas.Series.rank`` when producing ranks.
    """

    prediction_col: str = "prediction"
    prediction_type: PredictionType = "prediction"
    rank_method: str = "average"

    def __post_init__(self) -> None:
        # Catch typos early (e.g. "ranks" vs "rank").
        if self.prediction_type not in {"prediction", "rank"}:
            raise ValueError("prediction_type must be 'prediction' or 'rank'.")


def _fit_single_cross_section(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    *,
    rank: bool,
    rank_method: str,
) -> pd.Series:
    """Fit OLS on one date's cross-section and emit predictions for that date.

    The model is fit using rows with both features and the target present.
    Predictions are then produced for rows with features present (the target is
    not required at prediction time). If ``rank=True``, predictions are
    converted to percentile ranks within the cross-section.
    """
    # Rows usable for fitting need both X and y.
    fit = df.dropna(subset=[*feature_cols, target_col])
    # Heuristic: require a little breathing room beyond just "barely identified".
    if len(fit) < len(feature_cols) + 2:
        return pd.Series(dtype=float)

    # Design matrix: intercept + features.
    X_fit = fit[list(feature_cols)].to_numpy()
    y_fit = fit[target_col].to_numpy()
    X_fit = np.column_stack([np.ones(X_fit.shape[0]), X_fit])

    # Least-squares OLS; lightweight and good enough for a baseline.
    coefs, *_ = np.linalg.lstsq(X_fit, y_fit, rcond=None)

    # Rows usable for prediction only need features.
    pred_df = df.dropna(subset=[*feature_cols])
    if pred_df.empty:
        return pd.Series(dtype=float)

    # Build prediction matrix with the same intercept convention.
    X_pred = pred_df[list(feature_cols)].to_numpy()
    X_pred = np.column_stack([np.ones(X_pred.shape[0]), X_pred])
    preds = X_pred @ coefs

    # Return as a Series aligned to the original (ticker, date) index.
    pred_series = pd.Series(preds, index=pred_df.index)
    if rank:
        # Ranking is done within the cross-section for this date.
        pred_series = pred_series.rank(pct=True, method=rank_method)

    return pred_series


def cross_sectional_ols(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    target_col: str = "next_return",
    config: CrossSectionalOLSConfig | None = None,
) -> pd.DataFrame:
    """Run one cross-sectional OLS per date and return predictions or ranks.

    Parameters
    ----------
    panel
        Panel indexed by ``('ticker', 'date')`` containing regressors and the target.
    feature_cols
        Feature columns used as regressors in each date's regression.
    target_col
        Name of the dependent variable column (default: ``"next_return"``).
    config
        Output settings (column name and whether to return levels or ranks).

    Returns
    -------
    pd.DataFrame
        Panel indexed by ``('ticker', 'date')`` with a single prediction column.
        Dates where the regression could not be estimated are skipped.
    """
    _validate_multiindex(panel)
    cfg = config or CrossSectionalOLSConfig()

    results: list[pd.Series] = []

    # Fit a separate regression for each date's cross-section.
    for date, df_date in panel.groupby(level="date", sort=True):
        # Work on a copy so we can safely subset columns.
        df_date = df_date.copy()

        # Only keep what the regression needs.
        df_date = df_date[[*feature_cols, target_col]]

        # Fit and predict for this date.
        pred = _fit_single_cross_section(
            df_date,
            feature_cols=feature_cols,
            target_col=target_col,
            rank=cfg.prediction_type == "rank",
            rank_method=cfg.rank_method,
        )

        # Some dates won't have enough usable observations; skip those.
        if not pred.empty:
            results.append(pred.rename(cfg.prediction_col))

    # If nothing was produced, return an empty frame with the right schema.
    if not results:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])
        return pd.DataFrame(columns=[cfg.prediction_col], index=empty_index)

    # Stack all date-level Series into one output panel.
    output = pd.concat(results).to_frame()
    output.index = pd.MultiIndex.from_arrays(
        [
            output.index.get_level_values("ticker"),
            pd.to_datetime(output.index.get_level_values("date")),
        ],
        names=["ticker", "date"],
    )
    return output.sort_index()


__all__ = [
    "CrossSectionalOLSConfig", 
    "cross_sectional_ols"
]