"""Tree-based cross-sectional models for return prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import importlib.util
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

PredictionType = Literal["prediction", "rank"]
BoosterType = Literal["xgboost", "lightgbm"]


def _validate_multiindex(panel: pd.DataFrame) -> None:
    """Ensure the DataFrame is indexed by ('ticker', 'date')."""
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.names[:2] != ["ticker", "date"]:
        raise ValueError("panel must be indexed by ('ticker', 'date').")


@dataclass
class CrossSectionalModelConfig:
    """Base configuration shared across tree-based models."""

    prediction_col: str = "prediction"
    prediction_type: PredictionType = "prediction"
    rank_method: str = "average"

    def __post_init__(self) -> None:
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
    """Construct a RandomForestRegressor from the given configuration."""
    return RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        bootstrap=cfg.bootstrap,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )


def _require_xgboost() -> object:
    """Ensure xgboost is installed and return the XGBRegressor class."""
    if importlib.util.find_spec("xgboost") is None:
        raise ImportError("xgboost is required to use booster='xgboost'.")
    from xgboost import XGBRegressor  # type: ignore

    return XGBRegressor


def _require_lightgbm() -> object:
    """Ensure lightgbm is installed and return the LGBMRegressor class."""
    if importlib.util.find_spec("lightgbm") is None:
        raise ImportError("lightgbm is required to use booster='lightgbm'.")
    from lightgbm import LGBMRegressor # type: ignore
    
    return LGBMRegressor


def _build_boosting_model(cfg: GradientBoostingConfig):
    """Construct a gradient boosting model from the given configuration."""
    if cfg.booster == "xgboost":
        model_cls = _require_xgboost()
        return model_cls(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            n_jobs=cfg.n_jobs,
            random_state=cfg.random_state,
            objective="reg:squarederror",
            reg_lambda=0.0,
            reg_alpha=0.0,
        )
    
    model_cls = _require_lightgbm()
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
    """Fit cross-sectional models and return predictions or ranks."""
    _validate_multiindex(panel)
    # Iterate over dates and fit/predict
    results: list[pd.Series] = []
    for date, df_date in panel.groupby(level="date", sort=True):
        subset = df_date[[*feature_cols, target_col]].dropna(subset=feature_cols)
        if subset.empty:
            continue

        # Only train on tickers with non-missing target
        train_df = subset.dropna(subset=[target_col])
        if train_df.empty:
            continue

        model = estimator_factory()
        train_X = train_df[list(feature_cols)].to_numpy()
        train_y = train_df[target_col].to_numpy()
        model.fit(train_X, train_y)

        # Predict for ALL tickers that have features at this date
        pred_df = subset
        X_pred = pred_df[list(feature_cols)].to_numpy()
        preds = model.predict(X_pred)

        # Store predictions in a Series with MultiIndex
        pred_series = pd.Series(preds, index=pred_df.index, name=cfg.prediction_col)
        if cfg.prediction_type == "rank":
            pred_series = pred_series.groupby(level="date").rank(pct=True, method=cfg.rank_method)

        results.append(pred_series)

    # Combine results into a single DataFrame
    if not results:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])
        return pd.DataFrame(columns=[cfg.prediction_col], index=empty_index)

    # Concatenate and sort
    output = pd.concat(results).to_frame()
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
    """Fit a random forest per date and return cross-sectional predictions."""

    cfg = config or RandomForestConfig()
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
    """Fit gradient-boosted trees per date and return predictions or ranks."""

    cfg = config or GradientBoostingConfig()
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