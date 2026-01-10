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
    # These models are “one regression per month”; without a strict (ticker, date)
    # index contract, group-by behavior can silently change and invalidate results.
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
        # Preventing mis-typed modes keeps experiments reproducible; otherwise a
        # small spelling mistake could quietly change the output semantics.
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
    # Training requires both predictors and outcomes; restricting to complete rows
    # avoids implicitly imputing missing values and makes coefficients interpretable.
    fit = df.dropna(subset=[*feature_cols, target_col])

    # Underidentified (or barely identified) regressions produce unstable betas
    # that can dominate predictions, so we require a small buffer beyond k+1 params.
    if len(fit) < len(feature_cols) + 2:
        return pd.Series(dtype=float)

    # Adding an intercept prevents forcing the regression through the origin,
    # which would bias fitted returns if features are not mean-zero in a month.
    X_fit = fit[list(feature_cols)].to_numpy()
    y_fit = fit[target_col].to_numpy()
    X_fit = np.column_stack([np.ones(X_fit.shape[0]), X_fit])

    # Using least squares keeps this baseline lightweight and dependency-free;
    # we want a simple, transparent benchmark rather than a heavy modeling stack.
    coefs, *_ = np.linalg.lstsq(X_fit, y_fit, rcond=None)

    # At “prediction time” we only require features; allowing missing targets here
    # lets the same function be used even if some tickers lack next_return labels.
    pred_df = df.dropna(subset=[*feature_cols])
    if pred_df.empty:
        return pd.Series(dtype=float)

    # We reuse the exact same intercept convention so train/predict are consistent;
    # otherwise fitted coefficients would not map correctly to the prediction matrix.
    X_pred = pred_df[list(feature_cols)].to_numpy()
    X_pred = np.column_stack([np.ones(X_pred.shape[0]), X_pred])
    preds = X_pred @ coefs

    # Returning a Series with the original index preserves (ticker, date) alignment,
    # which is essential for later merges, ranking, and portfolio construction.
    pred_series = pd.Series(preds, index=pred_df.index)
    if rank:
        # Cross-sectional ranking focuses on ordering (useful for deciles/long-short)
        # and makes outputs robust to level shifts across months.
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

    # Estimating month-by-month keeps the baseline aligned with the predictive task:
    # we care about cross-sectional relationships that can vary over time.
    for date, df_date in panel.groupby(level="date", sort=True):
        # Copying avoids chained-assignment and makes it safe to subset aggressively
        # without mutating the original panel used by other models.
        df_date = df_date.copy()

        # Limiting to necessary columns reduces memory footprint and prevents
        # accidental leakage from unrelated columns in the panel.
        df_date = df_date[[*feature_cols, target_col]]

        # Keeping the estimator and the per-date loop separate makes it easy to
        # reuse the fitting logic in other cross-sectional baselines.
        pred = _fit_single_cross_section(
            df_date,
            feature_cols=feature_cols,
            target_col=target_col,
            rank=cfg.prediction_type == "rank",
            rank_method=cfg.rank_method,
        )

        # Some months genuinely cannot be estimated (too few stocks or too much missingness);
        # skipping them avoids manufacturing predictions from unreliable fits.
        if not pred.empty:
            results.append(pred.rename(cfg.prediction_col))

    # Returning an empty-but-well-typed frame keeps downstream merges predictable
    # and makes “no predictions available” an explicit, non-crashing state.
    if not results:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])
        return pd.DataFrame(columns=[cfg.prediction_col], index=empty_index)

    # Concatenating by index preserves the original (ticker, date) identifiers across months,
    # which is crucial for evaluating forecasts and forming portfolios consistently.
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
    "cross_sectional_ols",
]