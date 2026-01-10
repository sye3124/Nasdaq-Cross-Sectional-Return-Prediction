"""Generate rolling out-of-sample predictions for a small model “zoo”.

This module is a thin orchestration layer around the training utilities in
``training_scheme``. For each month t it:

1) trains each model using data strictly before t,
2) predicts for all tickers observed at t,
3) stacks each model’s predictions into a single panel,
4) attaches the realized next-period return as the evaluation target.

The output is ready to feed into evaluation routines (R², IC/Spearman, DM tests,
etc.).
"""

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
    """Column-wise concatenate a collection of prediction panels.

    Empty frames are ignored. If everything is empty, an empty DataFrame with a
    (ticker, date) MultiIndex is returned.
    """
    usable = [df for df in frames if not df.empty]

    # Keeping the empty-output case explicit avoids downstream code having to
    # special-case “no predictions” situations (common in short samples).
    if not usable:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])
        return pd.DataFrame(index=empty_index)

    # We want model columns to align on the same (ticker, date) universe without
    # letting missing forecasts from one model erase forecasts from another.
    combined = pd.concat(usable, axis=1, join="outer")

    # Normalizing index dtypes makes joins and group-bys stable across pipelines
    # (period vs datetime vs string dates can otherwise create “same month, different key” bugs).
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
    """Produce rolling out-of-sample predictions from several model families.

    The function runs multiple estimators (OLS, ridge, lasso, elastic net, and a
    random forest) under the same rolling/expanding training scheme. Each model
    gets its own prediction column name so the results can be compared side by
    side.

    Parameters
    ----------
    panel
        Panel indexed by ``('ticker', 'date')`` containing features and the target.
    feature_cols
        Feature columns used as regressors in every model.
    target_col
        Column in ``panel`` containing the next-period return to be predicted.
    window_config
        Controls the rolling/expanding training window and related settings.
    output_path
        If provided, writes the merged prediction panel to CSV at this path.
    realized_col
        Column name used for realized returns in the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        Panel indexed by ``('ticker', 'date')`` with one column per model
        prediction plus ``realized_col``. Rows where *all* model predictions are
        missing are removed.
    """
    cfg = window_config or WindowConfig()

    # Distinct column names are critical for fair side-by-side evaluation: they
    # prevent accidental overwrites and make comparisons explicit in saved outputs.
    ols_cfg = replace(cfg, prediction_col="ols_pred")
    ridge_cfg = replace(cfg, prediction_col="ridge_pred")
    lasso_cfg = replace(cfg, prediction_col="lasso_pred")
    enet_cfg = replace(cfg, prediction_col="elasticnet_pred")
    rf_cfg = replace(cfg, prediction_col="random_forest_pred")

    # Using a shared CV structure across regularized models keeps the comparison
    # “apples-to-apples”: differences should come from the estimator, not from tuning effort.
    ridge_model_cfg = RegularizedModelConfig(model_type="ridge", cv_folds=3)
    lasso_model_cfg = RegularizedModelConfig(model_type="lasso", cv_folds=3, max_iter=100_000)
    enet_model_cfg = RegularizedModelConfig(
        model_type="elasticnet",
        l1_ratios=(0.5, 0.9),
        cv_folds=3,
        max_iter=100_000,
    )

    # A factory ensures each window gets a fresh estimator, which avoids “memory”
    # from prior fits accidentally influencing later windows (especially for ensembles).
    rf_factory = lambda: RandomForestRegressor(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=20,
        max_features="sqrt",
        bootstrap=True,
        random_state=0,
        n_jobs=-1,
    )

    # Running every model through the same rolling scheme enforces the key OOS rule:
    # predictions at date t must be trained only on information strictly before t.
    predictions = _concat_predictions(
        [
            rolling_oos_predictions(
                panel,
                feature_cols,
                target_col=target_col,
                window_config=ols_cfg,  # plain OLS baseline
            ),
            rolling_regularized_predictions(
                panel,
                feature_cols,
                target_col=target_col,
                window_config=ridge_cfg,
                model_config=ridge_model_cfg,  # ridge regression
            ),
            rolling_regularized_predictions(
                panel,
                feature_cols,
                target_col=target_col,
                window_config=lasso_cfg,
                model_config=lasso_model_cfg,  # lasso regression
            ),
            rolling_regularized_predictions(
                panel,
                feature_cols,
                target_col=target_col,
                window_config=enet_cfg,
                model_config=enet_model_cfg,  # elastic net regression
            ),
            rolling_oos_predictions(
                panel,
                feature_cols,
                target_col=target_col,
                window_config=rf_cfg,
                model_factory=rf_factory,  # random forest regressor
            ),
        ]
    )

    # Including realized outcomes in the same output panel makes evaluation pipelines
    # simple and less error-prone (no separate merge step where indices can misalign).
    realized = panel[[target_col]].rename(columns={target_col: realized_col})
    realized.index = pd.MultiIndex.from_arrays(
        [
            realized.index.get_level_values("ticker"),
            pd.to_datetime(realized.index.get_level_values("date")),
        ],
        names=["ticker", "date"],
    )

    # Inner join ensures we only evaluate forecasts on dates where the target is observed,
    # which avoids biasing metrics via missing-outcome periods.
    merged = predictions.join(realized, how="inner")

    # Keeping rows where at least one model produced a forecast avoids discarding
    # useful partial results while still removing fully empty “no prediction” rows.
    prediction_cols = [col for col in merged.columns if col != realized_col]
    merged = merged.dropna(subset=prediction_cols, how="all").sort_index()

    # Persisting to disk is optional because notebooks often do exploratory runs;
    # writing only when requested keeps the function pure and test-friendly.
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(path)

    return merged


__all__ = [
    "generate_oos_predictions_all_models"
]