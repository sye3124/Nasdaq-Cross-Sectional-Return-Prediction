"""Tree-based cross-sectional models for return prediction.

This module mirrors the “one model per date” pattern used by the linear baselines:
for each month (or date) in the panel, fit a model on that date’s cross-section
and immediately score the same cross-section.

That is useful for:
- quick baselines that don’t require rolling time windows
- studying the pure cross-sectional relationship between features and next returns
- producing either level forecasts ("prediction") or cross-sectional ranks ("rank")

Supported estimators:
- RandomForestRegressor (always available via scikit-learn)
- Gradient-boosted trees via either XGBoost or LightGBM (optional dependencies)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import importlib.util
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

PredictionType = Literal["prediction", "rank"]
BoosterType = Literal["xgboost", "lightgbm"]


def _validate_multiindex(panel: pd.DataFrame) -> None:
    """Require a panel indexed by ('ticker', 'date')."""
    # The entire workflow relies on grouping by date and keeping ticker alignment,
    # so we fail early if the indexing contract is not satisfied.
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.names[:2] != ["ticker", "date"]:
        raise ValueError("panel must be indexed by ('ticker', 'date').")


@dataclass
class CrossSectionalModelConfig:
    """Shared configuration for cross-sectional tree models.

    Parameters
    ----------
    prediction_col
        Output column name used in the returned DataFrame.
    prediction_type
        "prediction" returns raw model outputs; "rank" converts them to percentile
        ranks within each date.
    rank_method
        Method passed to pandas rank when ``prediction_type="rank"``.
    """

    prediction_col: str = "prediction"
    prediction_type: PredictionType = "prediction"
    rank_method: str = "average"

    def __post_init__(self) -> None:
        # Restricting to a small set of allowed values prevents silent misconfiguration
        # (e.g., typos that would otherwise fall through and change behavior).
        if self.prediction_type not in {"prediction", "rank"}:
            raise ValueError("prediction_type must be 'prediction' or 'rank'.")


@dataclass
class RandomForestConfig(CrossSectionalModelConfig):
    """Configuration for cross-sectional random forests."""

    n_estimators: int = 200
    max_depth: int | None = None
    min_samples_leaf: int = 1
    bootstrap: bool = True
    n_jobs: int = -1
    random_state: int | None = 0


@dataclass
class GradientBoostingConfig(CrossSectionalModelConfig):
    """Configuration for gradient-boosted tree models (XGBoost/LightGBM)."""

    booster: BoosterType = "xgboost"
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    n_jobs: int = -1
    random_state: int | None = 0


def _build_random_forest(cfg: RandomForestConfig) -> RandomForestRegressor:
    """Instantiate a RandomForestRegressor from a config dataclass."""
    # Centralizing construction keeps estimator parameters consistent across dates
    # and makes it easy to audit/reproduce results.
    return RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        bootstrap=cfg.bootstrap,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )


def _require_xgboost() -> object:
    """Import XGBoost lazily and fail with a clear message if it's missing."""
    # Lazy imports let the project run on minimal dependencies; users only need
    # xgboost when they actually request that booster.
    if importlib.util.find_spec("xgboost") is None:
        raise ImportError("xgboost is required to use booster='xgboost'.")
    from xgboost import XGBRegressor  # type: ignore

    return XGBRegressor


def _require_lightgbm() -> object:
    """Import LightGBM lazily and fail with a clear message if it's missing."""
    # Same rationale as XGBoost: keep the base environment lightweight while
    # providing optional power users a path to stronger learners.
    if importlib.util.find_spec("lightgbm") is None:
        raise ImportError("lightgbm is required to use booster='lightgbm'.")
    from lightgbm import LGBMRegressor  # type: ignore

    return LGBMRegressor


def _build_boosting_model(cfg: GradientBoostingConfig):
    """Instantiate either an XGBoost or LightGBM regressor based on cfg.booster."""
    if cfg.booster == "xgboost":
        model_cls = _require_xgboost()
        # Conservative defaults reduce overfitting risk in noisy financial targets:
        # shallow-ish trees + subsampling typically generalize better cross-sectionally.
        return model_cls(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            n_jobs=cfg.n_jobs,
            random_state=cfg.random_state,
            objective="reg:squarederror",
            # Making regularization explicit avoids ambiguity about hidden defaults,
            # and makes later tuning choices easier to interpret.
            reg_lambda=0.0,
            reg_alpha=0.0,
        )

    model_cls = _require_lightgbm()
    # Keep parameter parity across boosters so comparisons are more “apples-to-apples”.
    return model_cls(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )


def _cross_sectional_predictions(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    target_col: str,
    cfg: CrossSectionalModelConfig,
    estimator_factory,
) -> pd.DataFrame:
    """Fit one model per date and return predictions (or within-date ranks)."""
    _validate_multiindex(panel)

    results: list[pd.Series] = []

    # Fitting separately per date isolates the pure cross-sectional mapping for that month,
    # which is useful for diagnostics and for “instant” baselines without time aggregation.
    for date, df_date in panel.groupby(level="date", sort=True):
        # We only generate outputs for observations with a complete feature vector;
        # otherwise, missingness would dominate the model behavior.
        subset = df_date[[*feature_cols, target_col]].dropna(subset=feature_cols)
        if subset.empty:
            continue

        # Training requires observed targets; this prevents the model from implicitly
        # treating missing targets as zeros (or dropping them later in inconsistent ways).
        train_df = subset.dropna(subset=[target_col])
        if train_df.empty:
            continue

        # Each date gets a fresh estimator to avoid accidental state carryover
        # and to keep the interpretation “one independent model per month”.
        model = estimator_factory()
        train_X = train_df[list(feature_cols)].to_numpy()
        train_y = train_df[target_col].to_numpy()
        model.fit(train_X, train_y)

        # Scoring all tickers with available features makes the output useful for
        # portfolio construction (where the rank/score is needed even if y is missing).
        pred_df = subset
        X_pred = pred_df[list(feature_cols)].to_numpy()
        preds = model.predict(X_pred)

        # Keeping predictions indexed by (ticker, date) ensures frictionless alignment
        # with returns, rankings, and decile portfolio routines downstream.
        pred_series = pd.Series(preds, index=pred_df.index, name=cfg.prediction_col)

        # Ranking is often more stable than levels for cross-sectional finance tasks,
        # and makes signals comparable across dates with different return scales.
        if cfg.prediction_type == "rank":
            pred_series = pred_series.groupby(level="date").rank(pct=True, method=cfg.rank_method)

        results.append(pred_series)

    if not results:
        # Returning a schema-correct empty frame avoids special-casing later
        # steps when the input panel is too sparse.
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])
        return pd.DataFrame(columns=[cfg.prediction_col], index=empty_index)

    output = pd.concat(results).to_frame()

    # Normalizing the date dtype avoids subtle join bugs when upstream code uses
    # Periods, strings, or mixed timestamp conventions.
    output.index = pd.MultiIndex.from_arrays(
        [
            output.index.get_level_values("ticker"),
            pd.to_datetime(output.index.get_level_values("date")),
        ],
        names=["ticker", "date"],
    )
    return output.sort_index()


def cross_sectional_random_forest(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    target_col: str = "next_return",
    config: RandomForestConfig | None = None,
) -> pd.DataFrame:
    """Fit a random forest per date and return predictions or ranks."""
    cfg = config or RandomForestConfig()

    # A factory guarantees a brand-new estimator per date, which prevents any
    # accidental reuse of fitted state across cross-sections.
    return _cross_sectional_predictions(
        panel,
        feature_cols,
        target_col=target_col,
        cfg=cfg,
        estimator_factory=lambda: _build_random_forest(cfg),
    )


def cross_sectional_gradient_boosting(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    target_col: str = "next_return",
    config: GradientBoostingConfig | None = None,
) -> pd.DataFrame:
    """Fit boosted trees per date (XGBoost or LightGBM) and return predictions or ranks."""
    cfg = config or GradientBoostingConfig()

    # Keeping the orchestration identical to the RF path makes it easier to compare
    # model families without introducing pipeline differences.
    return _cross_sectional_predictions(
        panel,
        feature_cols,
        target_col=target_col,
        cfg=cfg,
        estimator_factory=lambda: _build_boosting_model(cfg),
    )


__all__ = [
    "RandomForestConfig",
    "GradientBoostingConfig",
    "cross_sectional_random_forest",
    "cross_sectional_gradient_boosting",
]