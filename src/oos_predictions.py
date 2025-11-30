"""Utilities for generating out-of-sample predictions across multiple models."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from training_scheme import (
    RegularizedModelConfig,
    WindowConfig,
    rolling_oos_predictions,
    rolling_regularized_predictions,
)


def _concat_predictions(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    usable = [df for df in frames if not df.empty]
    if not usable:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])
        return pd.DataFrame(index=empty_index)

    combined = pd.concat(usable, axis=1, join="outer")
    combined.index = pd.MultiIndex.from_arrays(
        [
            combined.index.get_level_values("ticker"),
            pd.to_datetime(combined.index.get_level_values("date")),
        ],
        names=["ticker", "date"],
    )
    return combined.sort_index()


def generate_oos_predictions_all_models(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    target_col: str = "next_return",
    window_config: WindowConfig | None = None,
    output_path: str | Path | None = None,
    realized_col: str = "realized_return",
) -> pd.DataFrame:
    """Generate rolling out-of-sample predictions for several model families.

    The helper trains on the historical data available strictly before each date
    ``t``, predicts returns for all stocks observed at ``t``, and aggregates the
    predictions from linear, regularized, and tree-based estimators. The output
    pairs each model's forecast with the realized next-period return so the
    resulting panel can be used directly for performance evaluation.

    Parameters
    ----------
    panel : pd.DataFrame
        Multi-indexed by ``('ticker', 'date')`` containing the target and
        feature columns.
    feature_cols : Sequence[str]
        Columns used as regressors in every model.
    target_col : str, optional
        Name of the next-period return column. Defaults to ``"next_return"``.
    window_config : WindowConfig, optional
        Configuration controlling the expanding/rolling training window.
        Defaults to :class:`WindowConfig`.
    output_path : str or Path, optional
        When provided, the merged dataset of predictions and realized returns is
        written to this location as a CSV file.
    realized_col : str, optional
        Column name used for realized returns in the output.

    Returns
    -------
    pd.DataFrame
        Multi-indexed by ``('ticker', 'date')`` with one column per model plus
        the realized return column. Rows where every model failed to produce a
        prediction are dropped.
    """

    cfg = window_config or WindowConfig()

    ols_cfg = replace(cfg, prediction_col="ols_pred")
    ridge_cfg = replace(cfg, prediction_col="ridge_pred")
    lasso_cfg = replace(cfg, prediction_col="lasso_pred")
    enet_cfg = replace(cfg, prediction_col="elasticnet_pred")
    rf_cfg = replace(cfg, prediction_col="random_forest_pred")

    ridge_model_cfg = RegularizedModelConfig(model_type="ridge", cv_folds=3)
    lasso_model_cfg = RegularizedModelConfig(model_type="lasso", cv_folds=3)
    enet_model_cfg = RegularizedModelConfig(
        model_type="elasticnet",
        l1_ratios=(0.5, 0.9),
        cv_folds=3,
    )

    rf_factory = lambda: RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        bootstrap=True,
        random_state=0,
        n_jobs=-1,
    )

    predictions = _concat_predictions(
        [
            rolling_oos_predictions(
                panel,
                feature_cols,
                target_col=target_col,
                window_config=ols_cfg,
            ),
            rolling_regularized_predictions(
                panel,
                feature_cols,
                target_col=target_col,
                window_config=ridge_cfg,
                model_config=ridge_model_cfg,
            ),
            rolling_regularized_predictions(
                panel,
                feature_cols,
                target_col=target_col,
                window_config=lasso_cfg,
                model_config=lasso_model_cfg,
            ),
            rolling_regularized_predictions(
                panel,
                feature_cols,
                target_col=target_col,
                window_config=enet_cfg,
                model_config=enet_model_cfg,
            ),
            rolling_oos_predictions(
                panel,
                feature_cols,
                target_col=target_col,
                window_config=rf_cfg,
                model_factory=rf_factory,
            ),
        ]
    )

    realized = panel[[target_col]].rename(columns={target_col: realized_col})
    realized.index = pd.MultiIndex.from_arrays(
        [realized.index.get_level_values("ticker"), pd.to_datetime(realized.index.get_level_values("date"))],
        names=["ticker", "date"],
    )

    merged = predictions.join(realized, how="inner")

    prediction_cols = [col for col in merged.columns if col != realized_col]
    merged = merged.dropna(subset=prediction_cols, how="all")
    merged = merged.sort_index()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(path)

    return merged


__all__ = ["generate_oos_predictions_all_models"]
