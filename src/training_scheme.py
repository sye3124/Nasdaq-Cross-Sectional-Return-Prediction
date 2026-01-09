"""Training utilities for rolling/expanding estimation and out-of-sample prediction.

The core pattern used throughout this project is:

- For each month t (a cross-section of tickers):
    - train using only months strictly before t
    - predict for all tickers observed at t
- Repeat, rolling forward through time.

This module provides:
- a simple OLS fallback predictor (numpy least squares)
- rolling/expanding out-of-sample prediction loops
- regularized regressions (ridge/lasso/elastic net) with CV inside each window
- an optional “model selection” routine that picks among candidate model families
  via time-series cross-validation on past data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class WindowConfig:
    """Settings for how much history to use at each prediction date.

    Parameters
    ----------
    min_train_months
        Minimum number of *distinct months* required before we emit the first
        out-of-sample prediction.
    max_train_months
        Maximum lookback for a rolling window. Ignored when ``expanding=True``.
    expanding
        If True, training starts from the beginning and grows over time.
        If False, training uses a trailing window capped by ``max_train_months``.
    prediction_col
        Name used for the prediction column in the returned DataFrame.
    """

    min_train_months: int = 60
    max_train_months: int = 120
    expanding: bool = False
    prediction_col: str = "prediction"

    def __post_init__(self) -> None:
        # Simple sanity checks so we fail early with a helpful error.
        if self.min_train_months <= 0:
            raise ValueError("min_train_months must be positive.")
        if self.max_train_months <= 0:
            raise ValueError("max_train_months must be positive.")
        if self.min_train_months > self.max_train_months and not self.expanding:
            raise ValueError("min_train_months cannot exceed max_train_months for rolling windows.")


def _ols_predict(train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray) -> np.ndarray:
    """Fit OLS with an intercept on (train_X, train_y) and predict on test_X."""
    # Add an explicit intercept column so the model can learn a constant term.
    X_design = np.column_stack([np.ones(train_X.shape[0]), train_X])

    # Use least squares directly (fast, no extra dependencies).
    coefs, *_ = np.linalg.lstsq(X_design, train_y, rcond=None)

    # Apply the fitted coefficients to the test set (with the same intercept column).
    X_test = np.column_stack([np.ones(test_X.shape[0]), test_X])
    return X_test @ coefs


def _validate_multiindex(panel: pd.DataFrame) -> None:
    """Require a panel indexed by ('ticker', 'date')."""
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.names[:2] != ["ticker", "date"]:
        raise ValueError("panel must be indexed by ('ticker', 'date').")


def rolling_oos_predictions(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    target_col: str = "next_return",
    window_config: WindowConfig | None = None,
    model_factory: Callable[[], object] | None = None,
) -> pd.DataFrame:
    """Generate rolling/expanding out-of-sample predictions.

    Notes
    -----
    - Uses *distinct months* from the date index to define windows.
    - Training uses dates strictly before the prediction date.
    - If no ``model_factory`` is provided, falls back to OLS via numpy least squares.
    """
    _validate_multiindex(panel)
    cfg = window_config or WindowConfig()

    # Work with distinct dates so min_train_months is measured in months, not rows/tickers.
    dates = panel.index.get_level_values("date").unique().sort_values()
    preds: list[pd.Series] = []

    # First prediction occurs once we have cfg.min_train_months months of history.
    for idx in range(cfg.min_train_months, len(dates)):
        current_date = dates[idx]

        # Decide where the training window starts (expanding from 0, or rolling).
        train_start = 0 if cfg.expanding else max(0, idx - cfg.max_train_months)
        train_dates = dates[train_start:idx]  # strictly before current_date

        # Pull the training slice and require complete feature + target data.
        train_mask = panel.index.get_level_values("date").isin(train_dates)
        train_df = panel.loc[train_mask].dropna(subset=[*feature_cols, target_col])
        if train_df.empty:
            continue

        train_X = train_df[list(feature_cols)].to_numpy()
        train_y = train_df[target_col].to_numpy()

        # Fit the requested model, or use the OLS fallback.
        if model_factory:
            model = model_factory()
            model.fit(train_X, train_y)
            predictor = model.predict
        else:
            predictor = lambda X: _ols_predict(train_X, train_y, X)

        # Cross-section at the prediction date: we only need features to score.
        oos_df = panel.xs(current_date, level="date", drop_level=False)
        oos_features = oos_df[list(feature_cols)].dropna()
        if oos_features.empty:
            continue

        # Predict for all tickers observed at current_date with complete features.
        oos_pred = predictor(oos_features.to_numpy())

        # Store predictions as a Series indexed like the input panel.
        pred_series = pd.Series(oos_pred, index=oos_features.index, name=cfg.prediction_col)
        preds.append(pred_series)

    # If nothing was produced, return an empty frame with the expected schema.
    if not preds:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])
        return pd.DataFrame(columns=[cfg.prediction_col], index=empty_index)

    # Concatenate all per-date Series into one panel-shaped DataFrame.
    result = pd.concat(preds).to_frame()
    result.index = pd.MultiIndex.from_arrays(
        [
            result.index.get_level_values("ticker"),
            pd.to_datetime(result.index.get_level_values("date")),
        ],
        names=["ticker", "date"],
    )
    return result.sort_index()


@dataclass
class RegularizedModelConfig:
    """Settings for rolling regularized regressions.

    Parameters
    ----------
    model_type
        Which penalty family to use: "ridge", "lasso", or "elasticnet".
    alphas
        Candidate regularization strengths searched by the CV estimator.
    l1_ratios
        Candidate L1 ratios for elastic net (required if model_type="elasticnet").
    cv_folds
        Number of folds for cross-validation inside each training window.
    max_iter
        Max iterations for coordinate descent solvers (lasso/elastic net).
    random_state
        Seed for stochastic solvers (when applicable).
    scoring
        Scoring string (kept mainly for API compatibility; RidgeCV uses its own CV logic).
    """

    model_type: Literal["lasso", "ridge", "elasticnet"] = "ridge"
    alphas: Sequence[float] = tuple(np.logspace(-4, 1, 10))
    l1_ratios: Sequence[float] | None = None
    cv_folds: int = 5
    max_iter: int = 10000
    random_state: int | None = 0
    scoring: str | None = "neg_mean_squared_error"

    def __post_init__(self) -> None:
        # Keep validation lightweight but explicit.
        if self.model_type == "elasticnet" and not self.l1_ratios:
            raise ValueError("l1_ratios must be provided for elasticnet models.")
        if self.model_type not in {"lasso", "ridge", "elasticnet"}:
            raise ValueError("model_type must be 'lasso', 'ridge', or 'elasticnet'.")


@dataclass
class CandidateModel:
    """Description of a model option used during rolling model selection."""

    name: str
    factory: Callable[[], object]


@dataclass
class BestPeriodModel:
    """Stores the selected model and its CV diagnostics for one prediction date."""

    name: str
    model: object
    cv_scores: dict[str, float]


@dataclass
class ModelSelectionResult:
    """Output of rolling model selection.

    predictions
        Out-of-sample predictions, indexed by ('ticker', 'date').
    best_models
        Mapping from prediction date -> selected model + scores for that date.
    """

    predictions: pd.DataFrame
    best_models: dict[pd.Timestamp, BestPeriodModel]


def _build_regularized_model(cfg: RegularizedModelConfig):
    """Create a (scaler + CV regularized regression) pipeline."""
    # Scaling is important for lasso/elastic net and generally helps ridge too.
    scaler = StandardScaler(with_mean=True, with_std=True)

    if cfg.model_type == "ridge":
        model = RidgeCV(alphas=cfg.alphas, cv=cfg.cv_folds)
    elif cfg.model_type == "lasso":
        model = LassoCV(alphas=cfg.alphas, cv=cfg.cv_folds, max_iter=cfg.max_iter)
    elif cfg.model_type == "elasticnet":
        model = ElasticNetCV(
            l1_ratio=cfg.l1_ratios,
            alphas=cfg.alphas,
            cv=cfg.cv_folds,
            max_iter=cfg.max_iter,
        )
    else:
        # Should be unreachable due to __post_init__ validation.
        raise ValueError(f"Unknown model_type={cfg.model_type}")

    return Pipeline([("scaler", scaler), ("model", model)])


def rolling_regularized_predictions(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    target_col: str = "next_return",
    window_config: WindowConfig | None = None,
    model_config: RegularizedModelConfig | None = None,
) -> pd.DataFrame:
    """Generate rolling/expanding predictions using regularized regressions.

    This mirrors ``rolling_oos_predictions`` but fits a cross-validated regularized
    model inside each training window, then predicts the next cross-section.
    """
    _validate_multiindex(panel)

    cfg = window_config or WindowConfig()
    model_cfg = model_config or RegularizedModelConfig()

    # Distinct monthly dates define the rolling schedule.
    dates = panel.index.get_level_values("date").unique().sort_values()
    preds: list[pd.Series] = []

    for idx in range(cfg.min_train_months, len(dates)):
        current_date = dates[idx]

        train_start = 0 if cfg.expanding else max(0, idx - cfg.max_train_months)
        train_dates = dates[train_start:idx]

        train_mask = panel.index.get_level_values("date").isin(train_dates)
        train_df = panel.loc[train_mask].dropna(subset=[*feature_cols, target_col])
        if train_df.empty:
            continue

        train_x = train_df[list(feature_cols)].to_numpy()
        train_y = train_df[target_col].to_numpy()

        # Fit a scaled regularized model with CV-selected hyperparameters.
        model = _build_regularized_model(model_cfg)
        model.fit(train_x, train_y)

        # Score the current_date cross-section (features only).
        oos_df = panel.xs(current_date, level="date", drop_level=False)
        oos_features = oos_df[list(feature_cols)].dropna()
        if oos_features.empty:
            continue

        oos_pred = model.predict(oos_features.to_numpy())
        preds.append(pd.Series(oos_pred, index=oos_features.index, name=cfg.prediction_col))

    if not preds:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])
        return pd.DataFrame(columns=[cfg.prediction_col], index=empty_index)

    result = pd.concat(preds).to_frame()
    result.index = pd.MultiIndex.from_arrays(
        [
            result.index.get_level_values("ticker"),
            pd.to_datetime(result.index.get_level_values("date")),
        ],
        names=["ticker", "date"],
    )
    return result.sort_index()


def _safe_mean(values: list[float], *, default: float) -> float:
    """Mean that returns a default when the input list is empty."""
    if not values:
        return default
    return float(np.nanmean(values))


def _time_series_cv_scores(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    cv_folds: int,
    model_factory: Callable[[], object],
) -> dict[str, float]:
    """Evaluate a model using time-ordered CV over distinct months.

    The CV is performed over unique months (not rows), using TimeSeriesSplit:
    earlier months are used to predict later months.
    """
    unique_dates = train_df.index.get_level_values("date").unique().sort_values()
    if len(unique_dates) <= 1:
        return {"spearman": -np.inf, "r2": -np.inf, "mae": np.inf}

    # Cap the number of splits to what's feasible for the available history.
    splits = min(cv_folds, len(unique_dates) - 1)
    if splits < 2:
        return {"spearman": -np.inf, "r2": -np.inf, "mae": np.inf}

    splitter = TimeSeriesSplit(n_splits=splits)

    spearman_scores: list[float] = []
    r2_scores: list[float] = []
    mae_scores: list[float] = []

    for train_idx, val_idx in splitter.split(unique_dates):
        train_dates = unique_dates[train_idx]
        val_dates = unique_dates[val_idx]

        # Build fold datasets by filtering months.
        fit_mask = train_df.index.get_level_values("date").isin(train_dates)
        val_mask = train_df.index.get_level_values("date").isin(val_dates)

        fit_df = train_df.loc[fit_mask]
        val_df = train_df.loc[val_mask]
        if fit_df.empty or val_df.empty:
            continue

        # Fit on the training months...
        model = model_factory()
        model.fit(fit_df[list(feature_cols)].to_numpy(), fit_df[target_col].to_numpy())

        # ...and predict on the validation months.
        val_x = val_df[list(feature_cols)].to_numpy()
        val_y = val_df[target_col].to_numpy()
        preds = model.predict(val_x)

        # Compute Spearman correlation month-by-month, then average.
        val_tmp = val_df.copy()
        val_tmp["_pred"] = preds
        by_month = val_tmp.groupby(level="date").apply(
            lambda g: stats.spearmanr(g[target_col].to_numpy(), g["_pred"].to_numpy()).correlation
        )
        spearman = float(np.nanmean(by_month.to_numpy()))

        # Standard regression metrics on pooled validation observations.
        r2 = r2_score(val_y, preds) if len(val_y) > 1 else np.nan
        mae = mean_absolute_error(val_y, preds)

        # Store fold scores, making sure NaNs don't accidentally win comparisons.
        spearman_scores.append(np.nan_to_num(spearman, nan=-np.inf))
        r2_scores.append(np.nan_to_num(r2, nan=-np.inf))
        mae_scores.append(mae)

    return {
        "spearman": _safe_mean(spearman_scores, default=-np.inf),
        "r2": _safe_mean(r2_scores, default=-np.inf),
        "mae": _safe_mean(mae_scores, default=np.inf),
    }


def rolling_time_series_tuning(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    candidates: Sequence[CandidateModel],
    target_col: str = "next_return",
    window_config: WindowConfig | None = None,
    cv_folds: int = 3,
) -> ModelSelectionResult:
    """Roll forward with time-series CV-based model selection.

    For each prediction month t:
      1) gather training data from months < t (rolling or expanding)
      2) run time-series CV for each candidate model on the training set
      3) pick the best candidate using a lexicographic key:
            (Spearman, R2, -MAE)
      4) refit the best model on all training data
      5) score the cross-section at t and store predictions
    """
    if not candidates:
        raise ValueError("At least one CandidateModel must be provided.")

    _validate_multiindex(panel)
    cfg = window_config or WindowConfig()

    # Distinct dates define the rolling schedule.
    dates = panel.index.get_level_values("date").unique().sort_values()

    preds: list[pd.Series] = []
    best_models: dict[pd.Timestamp, BestPeriodModel] = {}

    for idx in range(cfg.min_train_months, len(dates)):
        current_date = dates[idx]

        # Define the training window in month units.
        train_start = 0 if cfg.expanding else max(0, idx - cfg.max_train_months)
        train_dates = dates[train_start:idx]

        train_mask = panel.index.get_level_values("date").isin(train_dates)
        train_df = panel.loc[train_mask].dropna(subset=[*feature_cols, target_col])
        if train_df.empty:
            continue

        best_candidate: CandidateModel | None = None
        best_scores: dict[str, float] | None = None
        best_key: tuple[float, float, float] | None = None

        # Evaluate each candidate model on the current training window.
        for candidate in candidates:
            scores = _time_series_cv_scores(train_df, feature_cols, target_col, cv_folds, candidate.factory)

            # Higher Spearman, higher R2, lower MAE.
            score_key = (scores["spearman"], scores["r2"], -scores["mae"])

            if best_key is None or score_key > best_key:
                best_candidate = candidate
                best_scores = scores
                best_key = score_key

        if best_candidate is None or best_scores is None:
            continue

        # Refit best model on all available training data for this month.
        model = best_candidate.factory()
        model.fit(train_df[list(feature_cols)].to_numpy(), train_df[target_col].to_numpy())

        # Store the chosen model and diagnostics for inspection/debugging.
        best_models[pd.to_datetime(current_date)] = BestPeriodModel(
            name=best_candidate.name,
            model=model,
            cv_scores=best_scores,
        )

        # Score the cross-section at current_date (features only).
        oos_df = panel.xs(current_date, level="date", drop_level=False)
        oos_features = oos_df[list(feature_cols)].dropna()
        if oos_features.empty:
            continue

        oos_pred = model.predict(oos_features.to_numpy())
        preds.append(pd.Series(oos_pred, index=oos_features.index, name=cfg.prediction_col))

    if not preds:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])
        return ModelSelectionResult(
            predictions=pd.DataFrame(columns=[cfg.prediction_col], index=empty_index),
            best_models={},
        )

    result = pd.concat(preds).to_frame()
    result.index = pd.MultiIndex.from_arrays(
        [
            result.index.get_level_values("ticker"),
            pd.to_datetime(result.index.get_level_values("date")),
        ],
        names=["ticker", "date"],
    )
    return ModelSelectionResult(predictions=result.sort_index(), best_models=best_models)


__all__ = [
    "WindowConfig",
    "rolling_oos_predictions",
    "RegularizedModelConfig",
    "rolling_regularized_predictions",
    "CandidateModel",
    "BestPeriodModel",
    "ModelSelectionResult",
    "rolling_time_series_tuning",
]