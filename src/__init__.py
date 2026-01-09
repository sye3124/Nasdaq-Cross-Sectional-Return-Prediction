from __future__ import annotations

# Feature engineering
from .features import FeatureConfig, compute_features

# Factor exposures / risk loadings
from .factor_exposures import ExposureConfig, compute_factor_exposures

# Cross-sectional models
from .linear_models import CrossSectionalOLSConfig, cross_sectional_ols
from .tree_models import (
    GradientBoostingConfig,
    RandomForestConfig,
    cross_sectional_gradient_boosting,
    cross_sectional_random_forest,
)

# Prediction workflows
from .oos_predictions import generate_oos_predictions_all_models
from .ranking import RankingConfig, convert_predictions_to_rankings

# Portfolio construction
from .portfolios import compute_decile_portfolio_returns

# Forecast evaluation
from .forecasting_metrics import compute_oos_r2, evaluate_forecasting_accuracy

__all__ = [
    # Features
    "FeatureConfig",
    "compute_features",
    # Factor exposures
    "ExposureConfig",
    "compute_factor_exposures",
    # Linear models
    "CrossSectionalOLSConfig",
    "cross_sectional_ols",
    # Tree models
    "RandomForestConfig",
    "GradientBoostingConfig",
    "cross_sectional_random_forest",
    "cross_sectional_gradient_boosting",
    # OOS prediction + ranking
    "generate_oos_predictions_all_models",
    "RankingConfig",
    "convert_predictions_to_rankings",
    # Portfolios
    "compute_decile_portfolio_returns",
    # Metrics
    "compute_oos_r2",
    "evaluate_forecasting_accuracy",
]