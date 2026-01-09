"""Convert model predictions into cross-sectional percentile ranks.

Many portfolio construction steps work better with *ranks* than with raw return
forecasts (which can be noisy and hard to compare across dates). This module
takes a prediction panel indexed by ``('ticker', 'date')`` and produces
cross-sectional percentile ranks in ``[0, 1]`` for each date.

Optional knobs:
- rank raw predictions or rank cross-sectional z-scores
- divide predictions by a risk proxy (volatility or beta) before ranking
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

PredictionBasis = Literal["return", "zscore"]
RiskAdjustment = Literal["volatility", "beta", None]


@dataclass
class RankingConfig:
    """Settings controlling how predictions are turned into ranks.

    Parameters
    ----------
    prediction_col
        Column in the input DataFrame containing the model forecasts.
    output_col
        Name of the output column containing percentile ranks in ``[0, 1]``.
    basis
        - "return": rank predictions directly
        - "zscore": z-score within each date first, then rank
    risk_adjust
        Optional risk adjustment to apply before ranking:
        - "volatility": divide by ``volatility_col``
        - "beta": divide by ``beta_col``
        - None: no adjustment
    volatility_col
        Column name used when ``risk_adjust == "volatility"``.
    beta_col
        Column name used when ``risk_adjust == "beta"``.
    rank_method
        Method forwarded to ``pandas.Series.rank`` when computing percentiles.
    """

    prediction_col: str = "prediction"
    output_col: str = "prediction_rank"
    basis: PredictionBasis = "return"
    risk_adjust: RiskAdjustment = None
    volatility_col: str | None = None
    beta_col: str | None = None
    rank_method: str = "average"

    def __post_init__(self) -> None:
        # Validate the small set of allowed options to prevent silent misconfig.
        if self.basis not in {"return", "zscore"}:
            raise ValueError("basis must be 'return' or 'zscore'.")
        if self.risk_adjust not in {"volatility", "beta", None}:
            raise ValueError("risk_adjust must be 'volatility', 'beta', or None.")
        if self.risk_adjust == "volatility" and not self.volatility_col:
            raise ValueError("volatility_col is required when risk_adjust='volatility'.")
        if self.risk_adjust == "beta" and not self.beta_col:
            raise ValueError("beta_col is required when risk_adjust='beta'.")


def _validate_multiindex(df: pd.DataFrame) -> None:
    """Require predictions to be indexed by ('ticker', 'date')."""
    if not isinstance(df.index, pd.MultiIndex) or df.index.names[:2] != ["ticker", "date"]:
        raise ValueError("predictions must be indexed by ('ticker', 'date').")


def _cross_sectional_zscore(series: pd.Series) -> pd.Series:
    """Compute per-date z-scores, filling undefined values with 0.

    If a date's cross-section has zero dispersion (std == 0) the z-score is
    undefined; we treat those as 0 so the downstream ranking remains stable.
    """
    grouped = series.groupby(level="date")
    mean = grouped.transform("mean")
    std = grouped.transform(lambda x: x.std(ddof=0)).replace(0, np.nan)
    z = (series - mean) / std
    return z.fillna(0.0)


def convert_predictions_to_rankings(
    predictions: pd.DataFrame,
    *,
    config: RankingConfig | None = None,
) -> pd.DataFrame:
    """Convert predictions into cross-sectional percentile ranks per date.

    Steps:
    1) select the prediction column and drop missing values
    2) optionally apply a risk adjustment (divide by volatility/beta)
    3) optionally z-score within each date
    4) rank within each date to get percentiles in [0, 1]

    Returns an empty frame (with the correct schema) when no usable predictions
    are available.
    """
    _validate_multiindex(predictions)
    cfg = config or RankingConfig()

    if cfg.prediction_col not in predictions.columns:
        raise KeyError(f"Missing prediction column '{cfg.prediction_col}'.")

    # Start from the prediction series and keep only observed values.
    pred_series = predictions[cfg.prediction_col].dropna()
    if pred_series.empty:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])
        return pd.DataFrame(columns=[cfg.output_col], index=empty_index)

    # Decide which column (if any) we use for risk adjustment.
    if cfg.risk_adjust == "volatility":
        adjust_col = cfg.volatility_col
    elif cfg.risk_adjust == "beta":
        adjust_col = cfg.beta_col
    else:
        adjust_col = None

    if adjust_col is not None:
        # Only align on tickers/dates where we have both prediction and risk proxy.
        if adjust_col not in predictions.columns:
            raise KeyError(f"Missing risk adjustment column '{adjust_col}'.")

        adjust_series = predictions.loc[pred_series.index, adjust_col]
        combined = pd.concat([pred_series, adjust_series], axis=1).dropna()

        if combined.empty:
            empty_index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])
            return pd.DataFrame(columns=[cfg.output_col], index=empty_index)

        # Avoid division by zero and discard infinities.
        den = combined[adjust_col].replace(0, np.nan)
        pred_series = (combined[cfg.prediction_col] / den).replace([np.inf, -np.inf], np.nan).dropna()

        if pred_series.empty:
            empty_index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])
            return pd.DataFrame(columns=[cfg.output_col], index=empty_index)

    # Choose the scoring series that will actually be ranked.
    scoring_series = _cross_sectional_zscore(pred_series) if cfg.basis == "zscore" else pred_series

    # Percentile ranks within each date (ties handled via cfg.rank_method).
    ranks = scoring_series.groupby(level="date").rank(pct=True, method=cfg.rank_method)

    output = ranks.rename(cfg.output_col).to_frame()
    output.index = pd.MultiIndex.from_arrays(
        [
            output.index.get_level_values("ticker"),
            pd.to_datetime(output.index.get_level_values("date")),
        ],
        names=["ticker", "date"],
    )
    return output.sort_index()


__all__ = [
    "RankingConfig", 
    "convert_predictions_to_rankings"
]