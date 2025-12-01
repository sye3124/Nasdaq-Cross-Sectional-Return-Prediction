# Nasdaq-Cross-Sectional-Return-Prediction
Testing Predictability and Portfolio Performance in Nasdaq Stocks

## Rolling factor exposures
Use `src/factor_exposures.py` to estimate Fama-French three-factor loadings for each stock. The helper uses a 36–60 month rolling window, runs an OLS regression of excess returns on `MKT`, `SMB`, and `HML`, and lags the resulting betas so they are observable at time *t*.

## Comparison tables and plots
`src/reporting.py` stitches together the forecasting, portfolio, and performance helpers to generate common evaluation artifacts:
- Model performance tables with cumulative-return plots via `compute_model_performance_table`
- Diebold–Mariano and Sharpe-significance tables via `compute_dm_significance_table` and `compute_sharpe_significance_table`
- Rolling Sharpe, factor-exposure, and feature-importance plots via `plot_rolling_sharpe`, `plot_factor_exposures`, and `plot_feature_importance`

The convenience wrapper `generate_comparison_report` accepts decile returns (and optional predictions, weights, factor loadings, and feature panels) and writes plots to a target `output_dir`.