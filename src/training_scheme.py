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
        # These checks prevent silently using nonsensical windows (which can otherwise
        # produce “valid-looking” outputs that are actually just artifacts).
        if self.min_train_months <= 0:
            raise ValueError("min_train_months must be positive.")
        if self.max_train_months <= 0:
            raise ValueError("max_train_months must be positive.")
        # For rolling windows, a min length larger than the cap would guarantee no predictions;
        # we fail fast so the user doesn’t debug “empty outputs” later.
        if self.min_train_months > self.max_train_months and not self.expanding:
            raise ValueError("min_train_months cannot exceed max_train_months for rolling windows.")


def _ols_predict(train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray) -> np.ndarray:
    """Fit OLS with an intercept on (train_X, train_y) and predict on test_X."""
    # Including an intercept avoids forcing the regression through the origin,
    # which would bias predictions if features are not mean-zero.
    X_design = np.column_stack([np.ones(train_X.shape[0]), train_X])

    # Numpy least-squares gives a stable baseline with minimal moving parts,
    # which is useful both for performance and for debugging.
    coefs, *_ = np.linalg.lstsq(X_design, train_y, rcond=None)

    # The test matrix must match the training design (same intercept convention),
    # otherwise predictions would be systematically mis-scaled.
    X_test = np.column_stack([np.ones(test_X.shape[0]), test_X])
    return X_test @ coefs


def _validate_multiindex(panel: pd.DataFrame) -> None:
    """Require a panel indexed by ('ticker', 'date')."""
    # The entire rolling scheme depends on grouping/slicing by these two levels;
    # validating early keeps errors clear and localized.
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

    # Using unique dates enforces a month-based notion of “history length” so the
    # window size isn’t distorted by changes in the number of tickers over time.
    dates = panel.index.get_level_values("date").unique().sort_values()
    preds: list[pd.Series] = []

    # We only start emitting OOS predictions once there is enough history to
    # make training meaningful and reduce early-sample instability.
    for idx in range(cfg.min_train_months, len(dates)):
        current_date = dates[idx]

        # The expanding/rolling choice controls the bias–variance trade-off:
        # expanding uses more data (lower variance), rolling adapts faster (lower bias to regime changes).
        train_start = 0 if cfg.expanding else max(0, idx - cfg.max_train_months)
        train_dates = dates[train_start:idx]  # strictly before current_date to avoid look-ahead

        # Dropping incomplete rows ensures the model never “learns” from partially missing
        # inputs, which would make comparisons across models inconsistent.
        train_mask = panel.index.get_level_values("date").isin(train_dates)
        train_df = panel.loc[train_mask].dropna(subset=[*feature_cols, target_col])
        if train_df.empty:
            continue

        train_X = train_df[list(feature_cols)].to_numpy()
        train_y = train_df[target_col].to_numpy()

        # A factory lets us swap models without changing the rolling logic; the OLS fallback
        # provides a deterministic baseline when we want maximum transparency.
        if model_factory:
            model = model_factory()
            model.fit(train_X, train_y)
            predictor = model.predict
        else:
            predictor = lambda X: _ols_predict(train_X, train_y, X)

        # At prediction time we only need features (targets are unknown by definition);
        # this keeps the scoring step leakage-safe.
        oos_df = panel.xs(current_date, level="date", drop_level=False)
        oos_features = oos_df[list(feature_cols)].dropna()
        if oos_features.empty:
            continue

        # Predicting on the full cross-section at current_date yields one month of
        # genuinely out-of-sample forecasts suitable for backtests/IC evaluation.
        oos_pred = predictor(oos_features.to_numpy())

        # Keeping the original MultiIndex makes downstream joins (to realized returns,
        # rankings, portfolios) simple and less error-prone.
        pred_series = pd.Series(oos_pred, index=oos_features.index, name=cfg.prediction_col)
        preds.append(pred_series)

    # Returning an empty but correctly-shaped frame avoids special-casing downstream
    # code when the sample is too short or heavily missing.
    if not preds:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])
        return pd.DataFrame(columns=[cfg.prediction_col], index=empty_index)

    # Concatenating preserves the per-ticker, per-month alignment and produces a
    # single “prediction panel” compatible with the evaluation modules.
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
        # The goal is to prevent silent “defaulting” to an unintended model family;
        # these guardrails make misconfiguration obvious.
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
    # Regularization strength depends on feature scale; standardizing inside the pipeline
    # makes hyperparameter choices meaningful and comparable across windows.
    scaler = StandardScaler(with_mean=True, with_std=True)

    if cfg.model_type == "ridge":
        model = RidgeCV(alphas=cfg.alphas, cv=cfg.cv_folds)
    elif cfg.model_type == "lasso":
        # Coordinate descent can need more iterations on noisy/collinear panels; max_iter is a stability knob.
        model = LassoCV(alphas=cfg.alphas, cv=cfg.cv_folds, max_iter=cfg.max_iter)
    elif cfg.model_type == "elasticnet":
        # Elastic net interpolates between ridge and lasso; searching l1_ratio helps when sparsity varies by window.
        model = ElasticNetCV(
            l1_ratio=cfg.l1_ratios,
            alphas=cfg.alphas,
            cv=cfg.cv_folds,
            max_iter=cfg.max_iter,
        )
    else:
        # This branch exists only to keep the function total; validation should prevent reaching it.
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

    # The rolling schedule is defined in calendar time (months), so we anchor everything
    # on unique dates rather than on raw row counts.
    dates = panel.index.get_level_values("date").unique().sort_values()
    preds: list[pd.Series] = []

    for idx in range(cfg.min_train_months, len(dates)):
        current_date = dates[idx]

        # This keeps the training set leakage-safe and lets the caller pick between
        # a regime-adaptive rolling window and a variance-reducing expanding window.
        train_start = 0 if cfg.expanding else max(0, idx - cfg.max_train_months)
        train_dates = dates[train_start:idx]

        # Consistent missing-data handling is important here because CV inside the window
        # is sensitive to which observations are considered “available.”
        train_mask = panel.index.get_level_values("date").isin(train_dates)
        train_df = panel.loc[train_mask].dropna(subset=[*feature_cols, target_col])
        if train_df.empty:
            continue

        train_x = train_df[list(feature_cols)].to_numpy()
        train_y = train_df[target_col].to_numpy()

        # Refitting CV inside each window adapts regularization strength to the local sample
        # (useful when the signal-to-noise ratio changes over time).
        model = _build_regularized_model(model_cfg)
        model.fit(train_x, train_y)

        # At time t we only know features, so predictions are produced using feature-complete rows only.
        oos_df = panel.xs(current_date, level="date", drop_level=False)
        oos_features = oos_df[list(feature_cols)].dropna()
        if oos_features.empty:
            continue

        oos_pred = model.predict(oos_features.to_numpy())
        preds.append(pd.Series(oos_pred, index=oos_features.index, name=cfg.prediction_col))

    # A schema-preserving empty return keeps the rest of the pipeline from needing
    # “if empty: …” checks everywhere.
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
    # Some CV folds can be skipped due to empty slices; returning a default here prevents
    # “empty list” crashes and makes the model-selection rule well-defined.
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

    # TimeSeriesSplit needs enough temporal blocks; we cap splits so we don’t create
    # degenerate folds (which would yield noisy or undefined scores).
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

        # Filtering by dates (not by rows) preserves the intended time ordering and prevents
        # “future” observations from entering the training fold.
        fit_mask = train_df.index.get_level_values("date").isin(train_dates)
        val_mask = train_df.index.get_level_values("date").isin(val_dates)

        fit_df = train_df.loc[fit_mask]
        val_df = train_df.loc[val_mask]
        if fit_df.empty or val_df.empty:
            continue

        # Each fold refits a fresh model so the score reflects true generalization across time,
        # not state carried from previous folds.
        model = model_factory()
        model.fit(fit_df[list(feature_cols)].to_numpy(), fit_df[target_col].to_numpy())

        # Validating on later months mimics the actual forecasting task more closely than
        # random CV, which would leak time information in financial panels.
        val_x = val_df[list(feature_cols)].to_numpy()
        val_y = val_df[target_col].to_numpy()
        preds = model.predict(val_x)

        # Spearman captures cross-sectional ranking ability (portfolio relevance) and is
        # less sensitive to outliers than level-based metrics.
        val_tmp = val_df.copy()
        val_tmp["_pred"] = preds
        by_month = val_tmp.groupby(level="date").apply(
            lambda g: stats.spearmanr(g[target_col].to_numpy(), g["_pred"].to_numpy()).correlation
        )
        spearman = float(np.nanmean(by_month.to_numpy()))

        # R2/MAE provide complementary “level accuracy” views alongside rank-based skill.
        r2 = r2_score(val_y, preds) if len(val_y) > 1 else np.nan
        mae = mean_absolute_error(val_y, preds)

        # Converting NaNs to pessimistic values prevents undefined folds from accidentally
        # being treated as “good” during candidate comparison.
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

    # Defining the schedule off unique months keeps “one step = one month” consistent,
    # regardless of how many tickers are listed in any particular month.
    dates = panel.index.get_level_values("date").unique().sort_values()

    preds: list[pd.Series] = []
    best_models: dict[pd.Timestamp, BestPeriodModel] = {}

    for idx in range(cfg.min_train_months, len(dates)):
        current_date = dates[idx]

        # Converting idx arithmetic into a month window guarantees we never train on (or tune against)
        # information from current_date or beyond.
        train_start = 0 if cfg.expanding else max(0, idx - cfg.max_train_months)
        train_dates = dates[train_start:idx]

        train_mask = panel.index.get_level_values("date").isin(train_dates)
        train_df = panel.loc[train_mask].dropna(subset=[*feature_cols, target_col])
        if train_df.empty:
            continue

        best_candidate: CandidateModel | None = None
        best_scores: dict[str, float] | None = None
        best_key: tuple[float, float, float] | None = None

        # Scoring candidates within the same window makes the selection “local” to that point in time,
        # which is important when the best bias–variance tradeoff changes across regimes.
        for candidate in candidates:
            scores = _time_series_cv_scores(train_df, feature_cols, target_col, cv_folds, candidate.factory)

            # The lexicographic rule prioritizes ranking skill first (portfolio relevance),
            # then level fit, then error magnitude, giving a stable tie-breaking structure.
            score_key = (scores["spearman"], scores["r2"], -scores["mae"])

            if best_key is None or score_key > best_key:
                best_candidate = candidate
                best_scores = scores
                best_key = score_key

        if best_candidate is None or best_scores is None:
            continue

        # Refitting on all training data uses every available observation once selection is done,
        # improving stability compared to using only one CV fold’s fit.
        model = best_candidate.factory()
        model.fit(train_df[list(feature_cols)].to_numpy(), train_df[target_col].to_numpy())

        # Persisting the chosen model and CV diagnostics makes the selection process auditable
        # (useful for debugging and for reporting).
        best_models[pd.to_datetime(current_date)] = BestPeriodModel(
            name=best_candidate.name,
            model=model,
            cv_scores=best_scores,
        )

        # Predictions at current_date are produced from features only, matching the real
        # forecasting setting and keeping the output leakage-safe.
        oos_df = panel.xs(current_date, level="date", drop_level=False)
        oos_features = oos_df[list(feature_cols)].dropna()
        if oos_features.empty:
            continue

        oos_pred = model.predict(oos_features.to_numpy())
        preds.append(pd.Series(oos_pred, index=oos_features.index, name=cfg.prediction_col))

    # Returning an empty-but-typed result avoids downstream special cases and keeps
    # the public API stable even when no predictions can be produced.
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