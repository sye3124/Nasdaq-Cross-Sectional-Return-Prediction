"""Forecasting accuracy metrics for model comparison."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


RealizedCol = str


def _validate_multiindex(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.MultiIndex) or df.index.names[:2] != ["ticker", "date"]:
        raise ValueError("predictions must be indexed by ('ticker', 'date').")


def _validate_realized_column(df: pd.DataFrame, realized_col: RealizedCol) -> None:
    if realized_col not in df.columns:
        raise KeyError(f"Missing realized return column '{realized_col}'.")


def _align_predictions(
    predictions: pd.Series, realized: pd.Series
) -> pd.DataFrame:
    aligned = pd.concat([predictions, realized], axis=1, join="inner").dropna()
    aligned.index = pd.MultiIndex.from_arrays(
        [
            aligned.index.get_level_values("ticker"),
            pd.to_datetime(aligned.index.get_level_values("date")),
        ],
        names=["ticker", "date"],
    )
    return aligned.sort_index()


def compute_oos_r2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Compute out-of-sample R² using a mean baseline."""

    true_series = pd.Series(y_true)
    pred_series = pd.Series(y_pred)

    if true_series.empty or pred_series.empty or len(true_series) != len(pred_series):
        return np.nan

    errors = true_series - pred_series
    sse = (errors**2).sum()
    baseline = true_series.mean()
    sst = ((true_series - baseline) ** 2).sum()
    if sst == 0:
        return np.nan
    return 1 - sse / sst


def _daily_spearman_rank_corr(aligned: pd.DataFrame) -> pd.Series:
    """Compute cross-sectional Spearman correlations per date."""

    results = {}
    for date, group in aligned.groupby(level="date"):
        if len(group) < 2:
            continue
        pred_ranks = group.iloc[:, 0].rank(pct=True)
        realized_ranks = group.iloc[:, 1].rank(pct=True)
        corr = pred_ranks.corr(realized_ranks, method="pearson")
        results[pd.to_datetime(date)] = corr
    return pd.Series(results).sort_index()


def evaluate_forecasting_accuracy(
    panel: pd.DataFrame,
    *,
    realized_col: RealizedCol = "realized_return",
    model_cols: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate forecasting accuracy across multiple models.

    The function expects a multi-indexed panel keyed by ``('ticker', 'date')``
    containing realized returns and one column per model forecast. Summary
    metrics (out-of-sample R², MAE, MSE, and mean cross-sectional Spearman rank
    correlation) are returned alongside a per-date Spearman series for each
    model, enabling cross-sectional analysis.

    Returns
    -------
    summary : pd.DataFrame
        Rows correspond to models with columns ``['oos_r2', 'mae', 'mse',
        'spearman_rank_corr']``.
    spearman_by_date : pd.DataFrame
        Rows are dates and columns are models containing daily Spearman rank
        correlations. Empty when no valid correlations are available.
    """

    _validate_multiindex(panel)
    _validate_realized_column(panel, realized_col)

    realized = panel[realized_col]
    realized.index = pd.MultiIndex.from_arrays(
        [realized.index.get_level_values("ticker"), pd.to_datetime(realized.index.get_level_values("date"))],
        names=["ticker", "date"],
    )

    candidate_cols = [c for c in panel.columns if c != realized_col]
    if model_cols is not None:
        candidate_cols = [c for c in candidate_cols if c in set(model_cols)]

    metrics: dict[str, dict[str, float]] = {}
    spearman_frames: list[pd.Series] = []

    for col in candidate_cols:
        aligned = _align_predictions(panel[col], realized)
        if aligned.empty:
            metrics[col] = {"oos_r2": np.nan, "mae": np.nan, "mse": np.nan, "spearman_rank_corr": np.nan}
            continue

        errors = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        mae = errors.abs().mean()
        mse = (errors**2).mean()
        r2 = compute_oos_r2(aligned.iloc[:, 1], aligned.iloc[:, 0])

        daily_spearman = _daily_spearman_rank_corr(aligned)
        spearman_mean = daily_spearman.mean()

        metrics[col] = {
            "oos_r2": r2,
            "mae": mae,
            "mse": mse,
            "spearman_rank_corr": spearman_mean,
        }
        if not daily_spearman.empty:
            spearman_frames.append(daily_spearman.rename(col))

    summary = pd.DataFrame(metrics).T
    summary.index.name = "model"

    spearman_by_date = pd.concat(spearman_frames, axis=1) if spearman_frames else pd.DataFrame()
    spearman_by_date.index = pd.to_datetime(spearman_by_date.index)

    return summary.sort_index(), spearman_by_date.sort_index()


__all__ = [
    "compute_oos_r2",
    "evaluate_forecasting_accuracy",
]
