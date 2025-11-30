"""Utilities for converting model predictions into cross-sectional ranks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

PredictionBasis = Literal["return", "zscore"]
RiskAdjustment = Literal["volatility", "beta", None]


@dataclass
class RankingConfig:
    """Configuration controlling cross-sectional ranking behavior.

    Attributes
    ----------
    prediction_col : str
        Name of the column containing model predictions to rank.
    output_col : str
        Name of the output column containing percentile ranks in ``[0, 1]``.
    basis : {"return", "zscore"}
        Whether to rank raw predictions (``"return"``) or their cross-sectional
        z-scores (``"zscore"``) computed per date.
    risk_adjust : {"volatility", "beta", None}
        Optional risk adjustment applied before ranking. When set, predictions
        are divided by the specified risk metric column.
    volatility_col : str | None
        Column containing predicted volatility. Required when
        ``risk_adjust == "volatility"``.
    beta_col : str | None
        Column containing predicted beta. Required when ``risk_adjust == "beta"``.
    rank_method : str
        Method forwarded to :meth:`pandas.Series.rank` when generating
        percentile ranks.
    """

    prediction_col: str = "prediction"
    output_col: str = "prediction_rank"
    basis: PredictionBasis = "return"
    risk_adjust: RiskAdjustment = None
    volatility_col: str | None = None
    beta_col: str | None = None
    rank_method: str = "average"

    def __post_init__(self) -> None:
        if self.basis not in {"return", "zscore"}:
            raise ValueError("basis must be 'return' or 'zscore'.")
        if self.risk_adjust not in {"volatility", "beta", None}:
            raise ValueError("risk_adjust must be 'volatility', 'beta', or None.")
        if self.risk_adjust == "volatility" and not self.volatility_col:
            raise ValueError("volatility_col is required when risk_adjust='volatility'.")
        if self.risk_adjust == "beta" and not self.beta_col:
            raise ValueError("beta_col is required when risk_adjust='beta'.")


def _validate_multiindex(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.MultiIndex) or df.index.names[:2] != ["ticker", "date"]:
        raise ValueError("predictions must be indexed by ('ticker', 'date').")


def _cross_sectional_zscore(series: pd.Series) -> pd.Series:
    grouped = series.groupby(level="date")
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    zscore = (series - mean) / std
    return zscore.fillna(0.0)


def convert_predictions_to_rankings(
    predictions: pd.DataFrame,
    *,
    config: RankingConfig | None = None,
) -> pd.DataFrame:
    """Convert predicted returns to cross-sectional percentile ranks.

    Parameters
    ----------
    predictions : pd.DataFrame
        Multi-indexed by ``('ticker', 'date')`` with at least the prediction
        column configured in :class:`RankingConfig`.
    config : RankingConfig, optional
        Controls the basis for ranking, output column name, and optional
        risk-adjustment inputs.

    Returns
    -------
    pd.DataFrame
        Multi-indexed by ``('ticker', 'date')`` with a single column containing
        percentile ranks in ``[0, 1]`` for each cross-section. Empty when no
        usable predictions are available.
    """

    _validate_multiindex(predictions)
    cfg = config or RankingConfig()

    if cfg.prediction_col not in predictions.columns:
        raise KeyError(f"Missing prediction column '{cfg.prediction_col}'.")

    pred_series = predictions[cfg.prediction_col].dropna()
    if pred_series.empty:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])
        return pd.DataFrame(columns=[cfg.output_col], index=empty_index)

    if cfg.risk_adjust == "volatility":
        adjust_col = cfg.volatility_col or ""
    elif cfg.risk_adjust == "beta":
        adjust_col = cfg.beta_col or ""
    else:
        adjust_col = None

    if adjust_col:
        if adjust_col not in predictions.columns:
            raise KeyError(f"Missing risk adjustment column '{adjust_col}'.")
        adjust_series = predictions.loc[pred_series.index, adjust_col]
        combined = pd.concat([pred_series, adjust_series], axis=1).dropna()
        if combined.empty:
            empty_index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])
            return pd.DataFrame(columns=[cfg.output_col], index=empty_index)
        pred_series = combined[cfg.prediction_col] / combined[adjust_col]

    if cfg.basis == "zscore":
        scoring_series = _cross_sectional_zscore(pred_series)
    else:
        scoring_series = pred_series

    ranks = scoring_series.groupby(level="date").rank(pct=True, method=cfg.rank_method)
    output = ranks.rename(cfg.output_col).to_frame()
    output.index = pd.MultiIndex.from_arrays(
        [output.index.get_level_values("ticker"), pd.to_datetime(output.index.get_level_values("date"))],
        names=["ticker", "date"],
    )
    return output.sort_index()


__all__ = ["RankingConfig", "convert_predictions_to_rankings"]