"""Helpers for rolling/expanding training, validation, and OOS prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class WindowConfig:
    """Configuration for rolling/expanding model estimation.

    Attributes
    ----------
    min_train_months : int
        Minimum number of distinct months required before the first prediction.
    max_train_months : int
        Maximum lookback window for rolling estimation. Ignored when ``expanding``
        is ``True``.
    expanding : bool
        When ``True``, grow the window from the start date; otherwise use a
        trailing window capped by ``max_train_months``.
    prediction_col : str
        Name of the output column containing stored predictions.
    """

    min_train_months: int = 60
    max_train_months: int = 120
    expanding: bool = False
    prediction_col: str = "prediction"

    def __post_init__(self) -> None:
        if self.min_train_months <= 0:
            raise ValueError("min_train_months must be positive.")
        if self.max_train_months <= 0:
            raise ValueError("max_train_months must be positive.")
        if self.min_train_months > self.max_train_months and not self.expanding:
            raise ValueError("min_train_months cannot exceed max_train_months for rolling windows.")


def _ols_predict(train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray) -> np.ndarray:
    """Fit an OLS model with an intercept and predict on ``test_X``."""

    X_design = np.column_stack([np.ones(len(train_X)), train_X])
    coefs, *_ = np.linalg.lstsq(X_design, train_y, rcond=None)

    X_test = np.column_stack([np.ones(len(test_X)), test_X])
    return X_test @ coefs


def _validate_multiindex(panel: pd.DataFrame) -> None:
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.names[:2] != ["ticker", "date"]:
        raise ValueError("panel must be indexed by ('ticker', 'date').")


def rolling_oos_predictions(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    target_col: str = "next_return",
    window_config: WindowConfig | None = None,
    model_factory: Callable[[], None] | None = None,
) -> pd.DataFrame:
    """Generate rolling or expanding out-of-sample predictions."""

    _validate_multiindex(panel)

    cfg = window_config or WindowConfig()

    # Use DISTINCT, SORTED dates so "months" means unique months, not rows
    dates = panel.index.get_level_values("date").unique().sort_values()
    preds: list[pd.Series] = []

    # First OOS prediction at dates[cfg.min_train_months]
    for idx in range(cfg.min_train_months, len(dates)):
        current_date = dates[idx]

        # Training window: all dates strictly before current_date
        if cfg.expanding:
            train_start = 0
        else:
            train_start = max(0, idx - cfg.max_train_months)

        train_dates = dates[train_start:idx]

        train_mask = panel.index.get_level_values("date").isin(train_dates)
        train_df = panel.loc[train_mask]
        train_df = train_df.dropna(subset=[*feature_cols, target_col])
        if train_df.empty:
            continue

        train_X = train_df[list(feature_cols)].to_numpy()
        train_y = train_df[target_col].to_numpy()

        if model_factory:
            model = model_factory()
            model.fit(train_X, train_y)
            predictor = model.predict
        else:
            predictor = lambda X: _ols_predict(train_X, train_y, X)

        # OOS features at current_date
        oos_df = panel.xs(current_date, level="date", drop_level=False)
        oos_features = oos_df[list(feature_cols)].dropna()
        if oos_features.empty:
            continue

        oos_pred = predictor(oos_features.to_numpy())
        pred_series = pd.Series(oos_pred, index=oos_features.index, name=cfg.prediction_col)
        preds.append(pred_series)

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


@dataclass
class RegularizedModelConfig:
    """Configuration for rolling regularized regressions.

    Attributes
    ----------
    model_type : {"lasso", "ridge", "elasticnet"}
        Choice of regularization penalty.
    alphas : Sequence[float]
        Candidate regularization strengths used during cross-validation.
    l1_ratios : Sequence[float] | None
        Candidate L1 ratios for elastic net. Required when ``model_type`` is
        ``"elasticnet"`` and ignored otherwise.
    cv_folds : int
        Number of cross-validation folds applied inside each training window.
    max_iter : int
        Maximum iterations for the coordinate-descent solvers.
    random_state : int | None
        Random seed used by stochastic solvers (lasso and elastic net).
    """

    model_type: Literal["lasso", "ridge", "elasticnet"] = "ridge"
    alphas: Sequence[float] = tuple(np.logspace(-4, 1, 10))
    l1_ratios: Sequence[float] | None = None
    cv_folds: int = 5
    max_iter: int = 10000
    random_state: int | None = 0
    # NEW: scoring used for RidgeCV (lasso/elasticnet ignore this)
    scoring: str | None = "neg_mean_squared_error"

    def __post_init__(self) -> None:
        if self.model_type == "elasticnet" and not self.l1_ratios:
            raise ValueError("l1_ratios must be provided for elasticnet models.")
        if self.model_type not in {"lasso", "ridge", "elasticnet"}:
            raise ValueError("model_type must be 'lasso', 'ridge', or 'elasticnet'.")

@dataclass
class CandidateModel:
    """Wrapper describing a candidate estimator for hyperparameter tuning."""

    name: str
    factory: Callable[[], object]


@dataclass
class BestPeriodModel:
    """Stores the best-performing model and CV diagnostics for one period."""

    name: str
    model: object
    cv_scores: dict[str, float]


@dataclass
class ModelSelectionResult:
    """Results from rolling time-series hyperparameter tuning."""

    predictions: pd.DataFrame
    best_models: dict[pd.Timestamp, BestPeriodModel]


def _build_regularized_model(cfg: RegularizedModelConfig):
    """Create a cross-validated regularized regression wrapped in a pipeline."""

    if cfg.model_type == "ridge":
        base_model = RidgeCV(
            alphas=cfg.alphas,
            cv=cfg.cv_folds,
            fit_intercept=True,
            scoring=cfg.scoring,  # <- use MSE-based scoring
        )
    elif cfg.model_type == "lasso":
        base_model = LassoCV(
            alphas=cfg.alphas,
            cv=cfg.cv_folds,
            max_iter=cfg.max_iter,
            random_state=cfg.random_state,
        )
    else:
        base_model = ElasticNetCV(
            alphas=cfg.alphas,
            l1_ratio=cfg.l1_ratios,
            cv=cfg.cv_folds,
            max_iter=cfg.max_iter,
            random_state=cfg.random_state,
        )

    return make_pipeline(StandardScaler(), base_model)


def rolling_regularized_predictions(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    target_col: str = "next_return",
    window_config: WindowConfig | None = None,
    model_config: RegularizedModelConfig | None = None,
) -> pd.DataFrame:
    """Generate rolling/expanding predictions using regularized regressions.

    The helper mirrors :func:`rolling_oos_predictions` but cross-validates
    regularization hyperparameters inside each training window before fitting
    the final model used to score the next period's cross-section.
    """

    _validate_multiindex(panel)

    cfg = window_config or WindowConfig()
    model_cfg = model_config or RegularizedModelConfig()

    # Use DISTINCT, SORTED dates so min_train_months is in "months", not rows
    dates = panel.index.get_level_values("date").unique().sort_values()

    preds: list[pd.Series] = []

    # Start at cfg.min_train_months so the first prediction is at dates[min_train_months]
    for idx in range(cfg.min_train_months, len(dates)):
        current_date = dates[idx]

        if cfg.expanding:
            train_start = 0
        else:
            train_start = max(0, idx - cfg.max_train_months)

        # Training dates are strictly BEFORE current_date
        train_dates = dates[train_start:idx]

        # Mask panel to those training dates
        train_mask = panel.index.get_level_values("date").isin(train_dates)
        train_df = panel.loc[train_mask]
        train_df = train_df.dropna(subset=[*feature_cols, target_col])
        if train_df.empty:
            continue

        train_x = train_df[list(feature_cols)].to_numpy()
        train_y = train_df[target_col].to_numpy()

        model = _build_regularized_model(model_cfg)
        model.fit(train_x, train_y)

        # OOS prediction on the current date cross-section
        oos_df = panel.xs(current_date, level="date", drop_level=False)
        oos_features = oos_df[list(feature_cols)].dropna()
        if oos_features.empty:
            continue

        oos_pred = model.predict(oos_features.to_numpy())
        pred_series = pd.Series(oos_pred, index=oos_features.index, name=cfg.prediction_col)
        preds.append(pred_series)

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
    """Evaluate a model with time-ordered CV over distinct months."""

    unique_dates = train_df.index.get_level_values("date").unique().sort_values()
    if len(unique_dates) <= 1:
        return {"spearman": -np.inf, "r2": -np.inf, "mae": np.inf}

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

        fit_mask = train_df.index.get_level_values("date").isin(train_dates)
        val_mask = train_df.index.get_level_values("date").isin(val_dates)

        fit_df = train_df.loc[fit_mask]
        val_df = train_df.loc[val_mask]

        if fit_df.empty or val_df.empty:
            continue

        model = model_factory()
        model.fit(fit_df[list(feature_cols)].to_numpy(), fit_df[target_col].to_numpy())

        val_x = val_df[list(feature_cols)].to_numpy()
        val_y = val_df[target_col].to_numpy()
        preds = model.predict(val_x)

        spearman = stats.spearmanr(val_y, preds).correlation
        r2 = r2_score(val_y, preds) if len(val_y) > 1 else np.nan
        mae = mean_absolute_error(val_y, preds)

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
    """Roll forward with time-series CV-based hyperparameter selection.

    Each out-of-sample month uses only past data for cross-validated model
    selection. Candidates are ranked by (i) Spearman rank correlation, (ii)
    out-of-sample R^2, and (iii) mean absolute error. The best-scoring model
    is then refit on the full training window and used to score the next
    cross-section. The chosen model and diagnostics are stored per period.
    """

    if not candidates:
        raise ValueError("At least one CandidateModel must be provided.")

    _validate_multiindex(panel)
    cfg = window_config or WindowConfig()

    dates = panel.index.get_level_values("date").unique().sort_values()
    preds: list[pd.Series] = []
    best_models: dict[pd.Timestamp, BestPeriodModel] = {}

    for idx in range(cfg.min_train_months, len(dates)):
        current_date = dates[idx]

        train_start = 0 if cfg.expanding else max(0, idx - cfg.max_train_months)
        train_dates = dates[train_start:idx]

        train_mask = panel.index.get_level_values("date").isin(train_dates)
        train_df = panel.loc[train_mask].dropna(subset=[*feature_cols, target_col])
        if train_df.empty:
            continue

        best_candidate: CandidateModel | None = None
        best_scores: dict[str, float] | None = None
        best_key: tuple[float, float, float] | None = None

        for candidate in candidates:
            scores = _time_series_cv_scores(train_df, feature_cols, target_col, cv_folds, candidate.factory)
            score_key = (scores["spearman"], scores["r2"], -scores["mae"])

            if best_key is None or score_key > best_key:
                best_candidate = candidate
                best_scores = scores
                best_key = score_key

        if best_candidate is None or best_scores is None:
            continue

        model = best_candidate.factory()
        model.fit(train_df[list(feature_cols)].to_numpy(), train_df[target_col].to_numpy())

        best_models[pd.to_datetime(current_date)] = BestPeriodModel(
            name=best_candidate.name,
            model=model,
            cv_scores=best_scores,
        )

        oos_df = panel.xs(current_date, level="date", drop_level=False)
        oos_features = oos_df[list(feature_cols)].dropna()
        if oos_features.empty:
            continue

        oos_pred = model.predict(oos_features.to_numpy())
        pred_series = pd.Series(oos_pred, index=oos_features.index, name=cfg.prediction_col)
        preds.append(pred_series)

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
