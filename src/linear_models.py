"""Cross-sectional linear models for return prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

PredictionType = Literal["prediction", "rank"]


def _validate_multiindex(panel: pd.DataFrame) -> None:
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.names[:2] != ["ticker", "date"]:
        raise ValueError("panel must be indexed by ('ticker', 'date').")


@dataclass
class CrossSectionalOLSConfig:
    """Configuration for cross-sectional OLS estimation.

    Attributes
    ----------
    prediction_col : str
        Name of the output column containing predicted returns or ranks.
    prediction_type : {"prediction", "rank"}
        Whether to return level predictions or percentile ranks within each
        cross-section.
    rank_method : str
        Ranking method passed to :meth:`pandas.Series.rank` when
        ``prediction_type`` is ``"rank"``.
    """

    prediction_col: str = "prediction"
    prediction_type: PredictionType = "prediction"
    rank_method: str = "average"

    def __post_init__(self) -> None:
        if self.prediction_type not in {"prediction", "rank"}:
            raise ValueError("prediction_type must be 'prediction' or 'rank'.")


def _fit_single_cross_section(
    features: pd.DataFrame,
    target_col: str,
    *,
    rank: bool,
    rank_method: str,
) -> pd.Series:
    """Estimate OLS on one date and return predictions (or ranks)."""

    complete = features.dropna(subset=[target_col])
    if complete.empty:
        return pd.Series(dtype=float)

    X = complete.drop(columns=[target_col]).to_numpy()
    y = complete[target_col].to_numpy()

    X_design = np.column_stack([np.ones(len(X)), X])
    coefs, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    preds = X_design @ coefs

    pred_series = pd.Series(preds, index=complete.index)
    if rank:
        pred_series = pred_series.groupby(level="date").rank(pct=True, method=rank_method)
    return pred_series


def cross_sectional_ols(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    target_col: str = "next_return",
    config: CrossSectionalOLSConfig | None = None,
) -> pd.DataFrame:
    """Run cross-sectional OLS at each date and return predictions or ranks.

    Parameters
    ----------
    panel : pd.DataFrame
        Multi-indexed by ``('ticker', 'date')`` containing features and the
        target column.
    feature_cols : Sequence[str]
        Columns to use as regressors in each date's cross-sectional regression.
    target_col : str, optional
        Name of the dependent variable column. Defaults to ``"next_return"``.
    config : CrossSectionalOLSConfig, optional
        Controls the output column name and whether predictions are returned as
        levels or percentile ranks.

    Returns
    -------
    pd.DataFrame
        Multi-indexed by ``('ticker', 'date')`` with one column containing the
        predicted returns or ranks for each date where estimation was possible.
    """

    _validate_multiindex(panel)
    cfg = config or CrossSectionalOLSConfig()

    results: list[pd.Series] = []
    for date, df_date in panel.groupby(level="date", sort=True):
        subset = df_date[[*feature_cols, target_col]].dropna(subset=feature_cols)
        if subset.empty:
            continue

        design = subset[list(feature_cols) + [target_col]]
        design.index = pd.MultiIndex.from_product(
            [design.index.get_level_values("ticker"), [date]], names=["ticker", "date"]
        )
        pred = _fit_single_cross_section(
            design[[*feature_cols, target_col]],
            target_col,
            rank=cfg.prediction_type == "rank",
            rank_method=cfg.rank_method,
        )
        if not pred.empty:
            results.append(pred.rename(cfg.prediction_col))

    if not results:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])
        return pd.DataFrame(columns=[cfg.prediction_col], index=empty_index)

    output = pd.concat(results).to_frame()
    output.index = pd.MultiIndex.from_arrays(
        [output.index.get_level_values("ticker"), pd.to_datetime(output.index.get_level_values("date"))],
        names=["ticker", "date"],
    )
    return output.sort_index()


__all__ = ["CrossSectionalOLSConfig", "cross_sectional_ols"]